"""Replay retained fixtures through the unchanged four-variant RTL."""
import argparse
import json
import re
from _shared import ROOT, VARIANTS, CYCLES, sample_files, tool, run
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--samples', nargs='+', type=int, default=list(range(16)))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    sources = sorted((ROOT / 'hardware/rtl').glob('*.sv'))
    run([tool('xvlog'), '--sv', *sources, ROOT / 'tests/rtl/tb_pursuit_state_depth_timing_top.sv'], out/'compile.log', out)
    glbl = tool('xvlog').resolve().parents[1] / 'data/verilog/src/glbl.v'
    run([tool('xvlog'), glbl], out/'compile_glbl.log', out)
    depth = VARIANTS.index(args.variant)
    run([tool('xelab'), 'work.tb_pursuit_state_depth_timing_top', 'work.glbl', '-L', 'unisims_ver',
         '-generic_top', f'STATELESS_DEPTH={depth}', '-s', 'replay', '-timescale', '1ns/1ps', '-mt', '4'], out/'elaborate.log', out)
    records = []
    for sample in args.samples:
        directory = sample_files(args.variant, sample, out / f'sample_{sample:02d}')
        log = directory / 'simulation.log'
        run([tool('xsim'), 'replay', '-tclbatch', ROOT/'hardware/vivado/run_simulation.tcl',
             '-testplusarg', f'IMAGE_DIRECTORY={ROOT / "models" / args.variant / "hardware_image"}',
             '-testplusarg', f'SAMPLE_DIRECTORY={directory}', '-log', log], directory/'console.log', out)
        match = re.search(r'PASS: state-depth register-bus replay transactions=(\d+) values=(\d+) core_cycles=(\d+)', log.read_text())
        assert match is not None, log
        transactions, values, cycles = map(int, match.groups())
        assert cycles == CYCLES[depth]
        assert transactions == 11242 - sum((3072, 1536, 576)[:depth])
        assert values == 270000 - sum((73728, 36864, 13824)[:depth])
        records.append(dict(variant=args.variant, sample=sample, values=values, cycles=cycles, mismatches=0))
    (out/'summary.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(records))


if __name__ == '__main__':
    main()
