"""Repository-local assets shared by the explicit reproduction commands."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('full_state', 'l1_stateless', 'l1_l2_stateless', 'l1_l2_l3_stateless',
            'l1_l2_l3_l4_stateless')
CYCLES = (105305, 90525, 78817, 72101, 68293)
LAYERS = tuple(f'L{i}' for i in range(1, 9)) + ('H1', 'H2')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tool(name):
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f'{name} is required; source Vivado settings64.sh first')
    return Path(path)


def run(command, log, cwd):
    with log.open('w') as stream:
        subprocess.run(list(map(str, command)), cwd=cwd, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def sample_files(variant, sample, directory):
    import numpy as np
    directory.mkdir(parents=True)
    manifest = json.loads((ROOT / 'results/quantization/user10' / f'{variant}.json').read_text())
    with np.load(ROOT / 'fixtures' / f'{variant}.npz', allow_pickle=False) as trace:
        fields = [('input', trace['input.event_count_integer'][sample], 9)]
        for quantity, bits in (('current', 24), ('spike', 8), ('membrane', 16)):
            values = np.concatenate([trace[f't{t}.{layer}.{quantity}_integer'][sample].reshape(-1)
                                     for t in (0, 1) for layer in LAYERS])
            fields.append((quantity, values, bits))
        fields.append(('result', trace['readout.output_integer_before_temporal_divide'][sample], 64))
        for name, values, bits in fields:
            path = directory / f'{name}.mem'
            mask = (1 << bits) - 1
            path.write_text(''.join(f'{int(value) & mask:0{(bits+3)//4}x}\n' for value in values.reshape(-1)))
            assert sha(path) == manifest['mem_sha256'][f'sample_{sample:02d}/{name}.mem'], path
    return directory
