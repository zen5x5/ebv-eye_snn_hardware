"""Replay independently prepared task/model cases through the same RTL."""
import argparse
import json
from pathlib import Path
import re
import shutil
import time
from _shared import ROOT, CYCLES, sha, tool, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = sorted((ROOT/'hardware/rtl').glob('*.sv'))
    sources.append(ROOT/'tests/rtl/tb_pursuit_state_depth_timing_top.sv')
    snapshot = out/'source_snapshot'
    snapshot.mkdir()
    for path in sources:
        shutil.copy2(path, snapshot/path.name)
    source_hashes = {path.name:sha(path) for path in sources}
    run([tool('xvlog'), '--sv', *[snapshot/path.name for path in sources]], out/'compile.log', out)
    glbl = tool('xvlog').resolve().parents[1]/'data/verilog/src/glbl.v'
    run([tool('xvlog'), glbl], out/'compile_glbl.log', out)
    cases = [(path.resolve(), json.loads((path/'summary.json').read_text())) for path in args.cases]
    for depth in sorted({record['depth'] for _,record in cases}):
        run([tool('xelab'), 'work.tb_pursuit_state_depth_timing_top', 'work.glbl', '-L','unisims_ver',
             '-generic_top',f'STATELESS_DEPTH={depth}', '-s',f'depth_{depth}',
             '-timescale','1ns/1ps','-mt','2'], out/f'elaborate_depth_{depth}.log', out)
    records = []
    for folder, case in cases:
        depth = case['depth']
        case_records = []
        for sample in range(case['samples']):
            started = time.monotonic()
            target = out/folder.name/f'sample_{sample:02d}'
            target.mkdir(parents=True)
            log = target/'simulation.log'
            run([tool('xsim'),f'depth_{depth}', '-tclbatch',ROOT/'hardware/vivado/run_simulation.tcl',
                 '-testplusarg',f'IMAGE_DIRECTORY={folder / "hardware_image"}',
                 '-testplusarg',f'SAMPLE_DIRECTORY={folder / f"sample_{sample:02d}"}',
                 '-log',log], target/'console.log', out)
            text = log.read_text()
            match = re.search(r'PASS: state-depth register-bus replay transactions=(\d+) values=(\d+) core_cycles=(\d+)',text)
            assert match is not None, log
            transactions,values,cycles = map(int,match.groups())
            assert transactions == 11242-sum((3072,1536,576,192)[:depth])
            assert values == 270000-sum((73728,36864,13824,4608)[:depth])
            if depth < len(CYCLES):
                assert cycles == CYCLES[depth]
            layers = [dict(zip(('layer','cycles','state_write_groups','state_read_groups','mac_issue_groups'),map(int,match)))
                      for match in re.findall(r'LAYER layer=(\d+) cycles=(\d+) state_write_groups=(\d+) state_read_groups=(\d+) mac_issue_groups=(\d+)',text)]
            assert len(layers) == 8
            assert all(layer['state_write_groups']==layer['state_read_groups']==0 for layer in layers[:depth])
            row = dict(case=folder.name,task=case['task'],depth=depth,sample=sample,
                       values_checked=values,transactions=transactions,core_cycles=cycles,
                       latency_ms_at_160mhz=cycles/160000, mismatches=0,layers=layers,
                       seconds=time.monotonic()-started)
            case_records.append(row)
            records.append(row)
            print(json.dumps({k:row[k] for k in ('case','sample','core_cycles','values_checked','seconds')}),flush=True)
        assert len({r['core_cycles'] for r in case_records}) == 1
        (out/folder.name/'summary.json').write_text(json.dumps(case_records,indent=2)+'\n')
    assert source_hashes == {path.name:sha(path) for path in sources}
    report = dict(status='passed',records=records,source_sha256=source_hashes,
                  cases={str(folder):sha(folder/'summary.json') for folder,_ in cases})
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
