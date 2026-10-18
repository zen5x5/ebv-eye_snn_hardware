"""Independent int64 reference with an explicitly specified stateless prefix."""
import numpy as np

LAYERS = [f"L{i}" for i in range(1, 9)] + ["H1", "H2"]


def reference(parameters, frames, stateless_layers):
    states = {}
    trace = {}
    h2 = []
    for timestep in range(2):
        values = frames.astype(np.int64)
        for name in LAYERS:
            weight = parameters[f"{name}.weight_integer"].astype(np.int64)
            fractions = parameters[f"{name}.current_and_state_fraction_bits"].astype(np.int64)
            np.testing.assert_array_equal(fractions, parameters[f"{name}.weight_fraction_bits"])
            if weight.ndim == 4:
                padded = np.pad(values, ((0, 0), (0, 0), (1, 1), (1, 1)))
                windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(2, 3))
                current = np.einsum("nchwkl,ockl->nohw", windows, weight, optimize=True)
            else:
                current = values @ weight.T
            channel_shape = (1, len(fractions)) + (1,)*(current.ndim-2)
            current += parameters[f"{name}.constant_integer"].astype(np.int64).reshape(channel_shape)
            assert current.min() >= -(1 << 39) and current.max() < (1 << 39)
            current = current.clip(-(1 << 23), (1 << 23)-1)
            if name in ("L1", "L2", "L3", "L4", "L5"):
                n, c, h, w = current.shape
                current = current.reshape(n, c, h//2, 2, w//2, 2).max(axis=(3, 5))
            previous = np.zeros_like(current) if timestep == 0 or (name in stateless_layers) else states[name]
            scale = (1 << fractions).reshape((1, len(fractions))+(1,)*(current.ndim-2))
            pre_reset = (3*previous)//4 + current
            spikes_unclipped = np.maximum(pre_reset, 0)//scale
            assert spikes_unclipped.max() <= 255, (name, timestep, "spike range")
            spikes = spikes_unclipped.clip(0, 255)
            membrane = np.maximum(pre_reset-spikes*scale, -scale)
            assert membrane.min() >= -32768 and membrane.max() <= 32767, (name, timestep, "state range")
            states[name] = membrane
            trace[f"t{timestep}.{name}.current_integer"] = current.astype(np.int32)
            trace[f"t{timestep}.{name}.spike_integer"] = spikes.astype(np.uint8)
            trace[f"t{timestep}.{name}.membrane_integer"] = membrane.astype(np.int32)
            values = spikes
            if name == "L8":
                values = values.reshape(len(frames), 480)
        h2.append(values)
    trace["readout.output_integer_before_temporal_divide"] = (h2[0]+h2[1]) @ parameters["readout.weight_integer"].astype(np.int64).T
    return trace

