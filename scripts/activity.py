"""Capture inference activity from a retained post-route functional netlist."""
import argparse
import gzip
import json
from pathlib import Path
import re
import shutil
from _shared import ROOT, VARIANTS, CYCLES, sample_files, tool, run, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--samples', nargs='+', type=int, default=list(range(16)))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    implementation = ROOT/'results/implementation'/args.variant
    netlist = out/'post_route_funcsim.v'
    with gzip.open(implementation/'post_route_funcsim.v.gz', 'rb') as incoming, netlist.open('wb') as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    shutil.copy2(implementation/'netlist_signal_scopes.txt', out/'netlist_signal_scopes.txt')
    run([tool('xvlog'), netlist], out/'compile_netlist.log', out)
    run([tool('xvlog'), '--sv', ROOT/'tests/rtl/tb_state_depth_power.sv'], out/'compile_tb.log', out)
    run([tool('xvlog'), tool('xvlog').resolve().parents[1]/'data/verilog/src/glbl.v'], out/'compile_glbl.log', out)
    run([tool('xelab'), 'work.tb_state_depth_power', 'work.glbl', '-L', 'unisims_ver', '-L', 'secureip',
         '-s', 'activity', '-debug', 'typical', '-timescale', '1ns/1ps', '-mt', '4'], out/'elaborate.log', out)
    records = []
    for sample in args.samples:
        folder = sample_files(args.variant, sample, out/f'sample_{sample:02d}')
        log = folder/'simulation.log'
        run([tool('xsim'), 'activity', '-tclbatch', ROOT/'hardware/vivado/record_activity.tcl',
             '-testplusarg', f'IMAGE_DIRECTORY={ROOT/"models"/args.variant/"hardware_image"}',
             '-testplusarg', f'SAMPLE_DIRECTORY={folder}',
             '-testplusarg', f'EXPECTED_CYCLES={CYCLES[VARIANTS.index(args.variant)]}', '-log', log], folder/'console.log', out)
        assert 'PASS: post-route power replay' in log.read_text(), log
        activity = folder/'inference.saif'
        shutil.move(out/'inference.saif', activity)
        window = re.search(r'POWER_END time_ns=([\d.]+) duration_ns=([\d.]+)', log.read_text()).groups()
        record = dict(variant=args.variant, sample=sample, mode='functional', saif_sha256=sha(activity),
                      netlist_sha256=sha(netlist), power_window=window)
        (folder/'summary.json').write_text(json.dumps(record, indent=2)+'\n')
        records.append(record)
    (out/'summary.json').write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
