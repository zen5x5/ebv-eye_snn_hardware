import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts._shared import ROOT, VARIANTS, sample_files
from hardware.quantization.integer_reference import reference


class RetainedFixtures(unittest.TestCase):
    def test_all_mem_files_reconstructed_exactly(self):
        with tempfile.TemporaryDirectory(prefix='ebv-fixture-check-') as folder:
            for variant in VARIANTS:
                for sample in range(16):
                    sample_files(variant, sample, Path(folder)/variant/f'sample_{sample:02d}')

    def test_independent_integer_reference(self):
        for depth, variant in enumerate(VARIANTS):
            with np.load(ROOT/'models'/variant/'parameters.npz', allow_pickle=False) as parameters:
                params = dict(parameters)
            with np.load(ROOT/'fixtures'/f'{variant}.npz', allow_pickle=False) as trace:
                frames = trace['input.event_count_integer'][:1]
                actual = reference(params, frames, tuple(f'L{i+1}' for i in range(depth)))
                for name, value in actual.items():
                    np.testing.assert_array_equal(value, trace[name][:1], err_msg=variant+' '+name)


if __name__ == '__main__':
    unittest.main()
