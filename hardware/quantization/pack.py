"""Pack fixed-point pursuit parameters into hardware streaming order.

Convolution weights are tiled by output-channel group, input-channel group,
output PE, input lane, and 3x3 tap.  Every tile is exactly 8*4*9 S16 words,
or eighteen 256-bit beats.  Linear weights are stored input-major in groups of
sixteen outputs so one input activation can be reused across sixteen DSPs.
The packer reconstructs every original tensor before writing an immutable,
hash-bound image directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
CONVOLUTION_LAYERS = tuple(f"L{index}" for index in range(1, 9))
LINEAR_LAYERS = ("H1", "H2")
OUTPUT_PES = 8
INPUT_CHANNEL_LANES = 4
KERNEL_TAPS = 9
WEIGHTS_PER_BEAT = 16
LINEAR_OUTPUT_LANES = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_hex_words(path: Path, values: np.ndarray, bits: int) -> None:
    mask = (1 << bits) - 1
    digits = (bits + 3) // 4
    with path.open("w", encoding="ascii") as handle:
        for value in np.asarray(values).reshape(-1):
            handle.write(f"{int(value) & mask:0{digits}x}\n")


def pack_beats(words: np.ndarray, *, word_bits: int, words_per_beat: int) -> np.ndarray:
    flattened = np.asarray(words).reshape(-1)
    if len(flattened) % words_per_beat:
        raise ValueError("word count must be an exact number of beats")
    mask = (1 << word_bits) - 1
    beats = []
    for start in range(0, len(flattened), words_per_beat):
        value = 0
        for lane, word in enumerate(flattened[start : start + words_per_beat]):
            value |= (int(word) & mask) << (lane * word_bits)
        beats.append(value)
    return np.asarray(beats, dtype=object)


def pack_convolution(weight: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    if weight.ndim != 4 or tuple(weight.shape[2:]) != (3, 3):
        raise ValueError(f"expected OIHW 3x3 weight, got {weight.shape}")
    outputs, inputs = weight.shape[:2]
    output_tiles = (outputs + OUTPUT_PES - 1) // OUTPUT_PES
    input_tiles = (inputs + INPUT_CHANNEL_LANES - 1) // INPUT_CHANNEL_LANES
    packed = []
    for output_tile in range(output_tiles):
        for input_tile in range(input_tiles):
            for output_lane in range(OUTPUT_PES):
                output_channel = output_tile * OUTPUT_PES + output_lane
                for input_lane in range(INPUT_CHANNEL_LANES):
                    input_channel = input_tile * INPUT_CHANNEL_LANES + input_lane
                    for kernel_index in range(KERNEL_TAPS):
                        kernel_row, kernel_column = divmod(kernel_index, 3)
                        value = 0
                        if output_channel < outputs and input_channel < inputs:
                            value = weight[
                                output_channel,
                                input_channel,
                                kernel_row,
                                kernel_column,
                            ]
                        packed.append(value)
    packed_array = np.asarray(packed, dtype=np.int16)

    reconstructed = np.zeros_like(weight, dtype=np.int16)
    cursor = 0
    for output_tile in range(output_tiles):
        for input_tile in range(input_tiles):
            for output_lane in range(OUTPUT_PES):
                output_channel = output_tile * OUTPUT_PES + output_lane
                for input_lane in range(INPUT_CHANNEL_LANES):
                    input_channel = input_tile * INPUT_CHANNEL_LANES + input_lane
                    for kernel_index in range(KERNEL_TAPS):
                        kernel_row, kernel_column = divmod(kernel_index, 3)
                        if output_channel < outputs and input_channel < inputs:
                            reconstructed[
                                output_channel,
                                input_channel,
                                kernel_row,
                                kernel_column,
                            ] = packed_array[cursor]
                        cursor += 1
    if not np.array_equal(reconstructed, weight.astype(np.int16)):
        raise AssertionError("convolution pack/unpack mismatch")
    return packed_array, {
        "output_channels": int(outputs),
        "input_channels": int(inputs),
        "output_tiles": int(output_tiles),
        "input_tiles": int(input_tiles),
        "tiles": int(output_tiles * input_tiles),
        "words_per_tile": OUTPUT_PES * INPUT_CHANNEL_LANES * KERNEL_TAPS,
        "beats_per_tile": (
            OUTPUT_PES * INPUT_CHANNEL_LANES * KERNEL_TAPS // WEIGHTS_PER_BEAT
        ),
    }


def pack_linear(weight: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    if weight.ndim != 2:
        raise ValueError(f"expected OI linear weight, got {weight.shape}")
    outputs, inputs = weight.shape
    output_groups = (outputs + LINEAR_OUTPUT_LANES - 1) // LINEAR_OUTPUT_LANES
    packed = []
    for input_index in range(inputs):
        for output_group in range(output_groups):
            for output_lane in range(LINEAR_OUTPUT_LANES):
                output_index = output_group * LINEAR_OUTPUT_LANES + output_lane
                packed.append(
                    weight[output_index, input_index] if output_index < outputs else 0
                )
    packed_array = np.asarray(packed, dtype=np.int16)

    reconstructed = np.zeros_like(weight, dtype=np.int16)
    cursor = 0
    for input_index in range(inputs):
        for output_group in range(output_groups):
            for output_lane in range(LINEAR_OUTPUT_LANES):
                output_index = output_group * LINEAR_OUTPUT_LANES + output_lane
                if output_index < outputs:
                    reconstructed[output_index, input_index] = packed_array[cursor]
                cursor += 1
    if not np.array_equal(reconstructed, weight.astype(np.int16)):
        raise AssertionError("linear pack/unpack mismatch")
    return packed_array, {
        "output_features": int(outputs),
        "input_features": int(inputs),
        "output_groups": int(output_groups),
        "words_per_input": int(output_groups * LINEAR_OUTPUT_LANES),
        "beats_per_input": int(output_groups),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    parameters_path = args.parameters.resolve()

    args.output_dir.mkdir(parents=True)
    with np.load(parameters_path) as parameters:
        convolution_words = []
        convolution_layout: dict[str, object] = {}
        convolution_word_cursor = 0
        convolution_beat_cursor = 0
        for name in CONVOLUTION_LAYERS:
            packed, description = pack_convolution(parameters[f"{name}.weight_integer"])
            convolution_layout[name] = {
                **description,
                "word_offset": convolution_word_cursor,
                "beat_offset": convolution_beat_cursor,
            }
            convolution_words.append(packed)
            convolution_word_cursor += len(packed)
            convolution_beat_cursor += len(packed) // WEIGHTS_PER_BEAT
        convolution_words_array = np.concatenate(convolution_words)
        convolution_beats = pack_beats(
            convolution_words_array,
            word_bits=16,
            words_per_beat=WEIGHTS_PER_BEAT,
        )

        linear_words = []
        linear_layout: dict[str, object] = {}
        linear_word_cursor = 0
        linear_beat_cursor = 0
        for name in (*LINEAR_LAYERS, "readout"):
            packed, description = pack_linear(parameters[f"{name}.weight_integer"])
            linear_layout[name] = {
                **description,
                "word_offset": linear_word_cursor,
                "beat_offset": linear_beat_cursor,
            }
            linear_words.append(packed)
            linear_word_cursor += len(packed)
            linear_beat_cursor += len(packed) // WEIGHTS_PER_BEAT
        linear_words_array = np.concatenate(linear_words)
        linear_beats = pack_beats(
            linear_words_array,
            word_bits=16,
            words_per_beat=WEIGHTS_PER_BEAT,
        )

        parameter_records: dict[str, object] = {}
        parameter_arrays = []
        parameter_cursor = 0
        for name in (*CONVOLUTION_LAYERS, *LINEAR_LAYERS):
            constants = parameters[f"{name}.constant_integer"].astype(np.int32)
            fractions = parameters[
                f"{name}.current_and_state_fraction_bits"
            ].astype(np.uint8)
            padded = ((len(constants) + OUTPUT_PES - 1) // OUTPUT_PES) * OUTPUT_PES
            records = np.zeros((padded, 2), dtype=np.int64)
            records[: len(constants), 0] = constants
            records[: len(fractions), 1] = fractions
            parameter_arrays.append(records)
            parameter_records[name] = {
                "record_offset": parameter_cursor,
                "records": int(padded),
                "valid_records": int(len(constants)),
            }
            parameter_cursor += padded
        channel_parameters = np.concatenate(parameter_arrays, axis=0)
        readout_fraction_bits = parameters[
            "readout.weight_fraction_bits"
        ].astype(np.uint8)

    paths = {
        "convolution_words": args.output_dir / "conv_weights_tiled_s16.mem",
        "convolution_beats": args.output_dir / "conv_weights_tiled_256.mem",
        "linear_words": args.output_dir / "linear_weights_input_major_s16.mem",
        "linear_beats": args.output_dir / "linear_weights_input_major_256.mem",
        "constants": args.output_dir / "layer_constants_s24.mem",
        "fractions": args.output_dir / "layer_fraction_bits_u5.mem",
        "readout_fractions": args.output_dir / "readout_fraction_bits_u5.mem",
    }
    write_hex_words(paths["convolution_words"], convolution_words_array, 16)
    write_hex_words(paths["convolution_beats"], convolution_beats, 256)
    write_hex_words(paths["linear_words"], linear_words_array, 16)
    write_hex_words(paths["linear_beats"], linear_beats, 256)
    write_hex_words(paths["constants"], channel_parameters[:, 0], 24)
    write_hex_words(paths["fractions"], channel_parameters[:, 1], 5)
    write_hex_words(paths["readout_fractions"], readout_fraction_bits, 5)

    artifacts = {}
    for name, path in paths.items():
        artifacts[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "lines": sum(1 for _ in path.open("r", encoding="ascii")),
        }
    layout = {
        "schema_version": SCHEMA_VERSION,
        "scope": "pursuit_fixed_point_hardware_streaming_parameter_image",
        "packer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "source": {
            "parameters_npz": {
                "path": str(parameters_path),
                "sha256": sha256_file(parameters_path),
            },
        },
        "convolution": {
            "order": "layer, output_tile, input_tile, output_lane, input_lane, kernel_row_major",
            "output_pes": OUTPUT_PES,
            "input_channel_lanes": INPUT_CHANNEL_LANES,
            "kernel_taps": KERNEL_TAPS,
            "weight_bits": 16,
            "weights_per_beat": WEIGHTS_PER_BEAT,
            "beat_bits": 256,
            "layers": convolution_layout,
        },
        "linear": {
            "order": "layer, input_index, output_group, output_lane",
            "output_lanes": LINEAR_OUTPUT_LANES,
            "weight_bits": 16,
            "weights_per_beat": WEIGHTS_PER_BEAT,
            "beat_bits": 256,
            "layers": linear_layout,
        },
        "channel_parameters": {
            "order": "L1..L8,H1,H2; each padded to a multiple of 8 outputs",
            "layers": parameter_records,
            "constant_bits": 24,
            "fraction_bits": 5,
        },
        "readout": {
            "fraction_bits_artifact": "readout_fractions",
            "normalization": "raw_integer / 2^(fraction_bits + 1)",
            "outputs": int(len(readout_fraction_bits)),
        },
        "artifacts": artifacts,
        "self_check": "every packed convolution and linear tensor reconstructed exactly",
    }
    layout_path = args.output_dir / "layout.json"
    layout_path.write_text(
        json.dumps(layout, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(layout_path)


if __name__ == "__main__":
    main()
