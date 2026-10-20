"""Freeze a trained continuous-output model and exact reference inputs for RTL."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _shared import LAYERS, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--task', choices=('pursuit', 'saccade'), required=True)
    parser.add_argument('--depth', type=int, choices=range(5), required=True)
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from hardware.quantization.fixed_point import load_model, RuntimeCounters
    from hardware.quantization.formal import CONTRACT, ExactFixedPointReference
    from hardware.quantization.integer_reference import reference as independent_reference
    from hardware.quantization.pack import write_hex_words

    torch.set_num_threads(2)
    source = args.model_dir.resolve()
    data_path = args.data.resolve()
    config = json.loads((source/'configuration.json').read_text())
    expected_mask = {f'L{i+1}': list(range((8,16,24,32)[i])) for i in range(args.depth)}
    assert config['model']['stateless_channels'] == expected_mask
    assert config['model']['alpha'] == 0.75
    assert config['input']['temporal_steps'] == 2
    assert config['training']['seed'] == 2021
    paths = [source/name for name in ('best_checkpoint.pt', 'configuration.json', 'summary.json')]
    paths += [data_path, data_path.parent/'manifest.json']
    hashes = {str(path): sha(path) for path in paths}
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'model').mkdir()
    for path in paths[:3]:
        shutil.copy2(path, out/'model'/path.name)
    model, checkpoint = load_model(out/'model/best_checkpoint.pt', config)
    stateless = tuple(expected_mask)
    reference = ExactFixedPointReference(model, **CONTRACT, stateless_layer_names=stateless)
    with np.load(data_path, allow_pickle=False) as data:
        frames = data['event_frames'][:args.samples].copy()
        labels = data['labels'][:args.samples].copy()
    assert frames.shape == (args.samples, 2, 96, 128)
    assert np.all(frames.sum(axis=(1,2,3), dtype=np.int64) == 400)
    counters = RuntimeCounters.empty()
    trace = {}
    with torch.inference_mode():
        reference.predict(torch.from_numpy(frames).float(), counters, trace)
    parameters = reference.parameter_arrays()
    independent = independent_reference(parameters, frames, stateless)
    for key, values in independent.items():
        np.testing.assert_array_equal(values, trace[key], err_msg=key)
    for layer in stateless:
        for quantity in ('current', 'spike', 'membrane'):
            np.testing.assert_array_equal(trace[f't0.{layer}.{quantity}_integer'],
                                          trace[f't1.{layer}.{quantity}_integer'])
    trace['input.event_count_integer'] = frames
    np.savez_compressed(out/'trace.npz', **trace)
    np.savez_compressed(out/'parameters.npz', **parameters)
    np.savez_compressed(out/'inputs.npz', event_frames=frames, labels=labels)
    subprocess.run([sys.executable, '-m', 'hardware.quantization.pack',
                    '--parameters', str(out/'parameters.npz'),
                    '--output-dir', str(out/'hardware_image')], cwd=ROOT, check=True)
    mem_hashes = {}
    for sample in range(args.samples):
        folder = out/f'sample_{sample:02d}'
        folder.mkdir()
        fields = [('input', frames[sample], 9)]
        for quantity, bits in (('current',24), ('spike',8), ('membrane',16)):
            values = np.concatenate([trace[f't{t}.{layer}.{quantity}_integer'][sample].reshape(-1)
                                     for t in (0,1) for layer in LAYERS])
            fields.append((quantity, values, bits))
        fields.append(('result',trace['readout.output_integer_before_temporal_divide'][sample],64))
        for name, values, bits in fields:
            target = folder/f'{name}.mem'
            write_hex_words(target, values, bits)
            mem_hashes[str(target.relative_to(out))] = sha(target)
    assert hashes == {str(path): sha(path) for path in paths}
    record = dict(task=args.task, depth=args.depth, seed=2021,
                  checkpoint_epoch=int(checkpoint['epoch']), samples=args.samples,
                  source=str(source), data=str(data_path), source_sha256=hashes,
                  source_unchanged=True, arithmetic=CONTRACT,
                  independent_reference_arrays=len(independent),
                  independent_reference_values=sum(v.size for v in independent.values()),
                  runtime_counters=asdict(counters), mem_sha256=mem_hashes,
                  model_parameter_shapes={k:list(v.shape) for k,v in parameters.items()})
    (out/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('task','depth','samples','checkpoint_epoch',
                                          'independent_reference_arrays','independent_reference_values')}))


if __name__ == '__main__':
    main()
