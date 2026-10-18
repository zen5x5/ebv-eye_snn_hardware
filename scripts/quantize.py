"""Evaluate a retained trained model on the included user10 test data."""
import argparse
import json
from pathlib import Path
import sys
from _shared import ROOT, VARIANTS
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--test-data', type=Path, default=ROOT/'fixtures/user10/test.npz')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from hardware.quantization.fixed_point import load_model, RuntimeCounters
    from hardware.quantization.formal import CONTRACT, ExactFixedPointReference, errors, means, normalized_to_pixels
    torch.set_num_threads(4)
    model_dir = ROOT/'models'/args.variant
    config = json.loads((model_dir/'configuration.json').read_text())
    model, checkpoint = load_model(model_dir/'best_checkpoint.pt', config)
    stateless = tuple(f'L{i+1}' for i in range(VARIANTS.index(args.variant)))
    reference = ExactFixedPointReference(model, **CONTRACT, stateless_layer_names=stateless)
    with np.load(args.test_data, allow_pickle=False) as data:
        frames, labels = data['event_frames'], data['labels']
    prediction = []
    with torch.inference_mode():
        for start in range(0, len(frames), 16):
            prediction.append(reference.predict(torch.from_numpy(frames[start:start+16]).float(), RuntimeCounters.empty()).numpy())
    prediction = np.concatenate(prediction)
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir/'parameters.npz', **reference.parameter_arrays())
    np.savez_compressed(args.output_dir/'predictions.npz', prediction=prediction, labels=labels)
    report = dict(variant=args.variant, checkpoint_epoch=int(checkpoint['epoch']), samples=len(frames),
                  fixed_test=means(errors(normalized_to_pixels(prediction), labels)))
    (args.output_dir/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
