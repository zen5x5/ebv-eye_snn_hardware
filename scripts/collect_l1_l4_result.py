"""Collect the validated depth-4 implementation without rerunning FPGA tools."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import re
import shutil
from _shared import ROOT, VARIANTS, CYCLES, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory',type=Path,required=True)
    args=parser.parse_args()
    run=args.run_directory.resolve()
    name=VARIANTS[4]
    synthesis=run/'synthesis'
    implementation=run/'implementation'
    replay=json.loads((run/'rtl_pursuit/summary.json').read_text())
    assert replay['status']=='passed' and len(replay['records'])==16
    assert all(r['core_cycles']==CYCLES[4] and r['mismatches']==0 for r in replay['records'])
    timing=(implementation/'reports/post_route_timing.rpt').read_text()
    match=re.search(r'WNS\(ns\).*?\n[^\n]*\n\s*([^\n]+)',timing,re.S)
    values=match[1].split()
    setup,hold,pulse=map(float,(values[0],values[4],values[8]))
    assert min(setup,hold,pulse)>=0
    assert all(float(values[i])==0 for i in (1,2,5,6,9,10))
    checks=dict((name,int(count)) for name,count in re.findall(r'checking (\w+) \((\d+)\)',(implementation/'reports/post_route_check_timing.rpt').read_text()))
    assert len(checks)==12 and all(value==0 for value in checks.values())
    route=(implementation/'reports/post_route_status.rpt').read_text()
    assert re.search(r'# of nets with routing errors\.+\s*:\s*0\s*:',route)
    run_summary=dict(line.split('=',1) for line in (implementation/'run_summary.txt').read_text().splitlines())
    assert float(run_summary['target_clock_period_ns'])==6.25
    assert run_summary['critical_warning_count']=='0'
    resource_rows={parts[1].strip():parts[2].strip() for line in (implementation/'reports/post_route_utilization.rpt').read_text().splitlines() if len(parts:=line.split('|'))>=4}
    resources=dict(lut=int(resource_rows['CLB LUTs']),ff=int(resource_rows['CLB Registers']),
                   dsp=int(resource_rows['DSPs']),bram36_equivalent=float(resource_rows['Block RAM Tile']),
                   ramb36=int(resource_rows['RAMB36/FIFO*']),ramb18=int(resource_rows['RAMB18']))
    assert resources['bram36_equivalent']==resources['ramb36']+resources['ramb18']/2
    source_hashes={}
    for path in sorted((synthesis/'source_snapshot').glob('*.sv')):
        assert sha(path)==sha(ROOT/'hardware/rtl'/path.name)
        source_hashes[path.name]=sha(path)
    for stage,source in (('synthesis',synthesis),('implementation',implementation)):
        target=ROOT/'results'/stage/name
        target.mkdir(parents=True,exist_ok=False)
        shutil.copytree(source/'reports',target/'reports')
        shutil.copytree(source/'source_snapshot',target/'source_snapshot')
        shutil.copy2(source/'run_summary.txt',target/'run_summary.txt')
        checkpoint='post_synth.dcp' if stage=='synthesis' else 'post_route.dcp'
        shutil.copy2(source/checkpoint,target/checkpoint)
    target=ROOT/'results/implementation'/name
    for filename in ('physical_audit.txt','netlist_signal_scopes.txt','implemented_constraints.xdc'):
        shutil.copy2(implementation/filename,target/filename)
    original_netlist=implementation/'post_route_funcsim.v'
    with original_netlist.open('rb') as src,gzip.open(target/'post_route_funcsim.v.gz','wb',compresslevel=9) as dest:
        shutil.copyfileobj(src,dest)
    rtl_target=ROOT/'results/rtl'/name
    rtl_target.mkdir()
    shutil.copytree(run/'rtl_pursuit/source_snapshot',rtl_target/'source_snapshot')
    for sample in range(16):
        folder=rtl_target/f'sample_{sample:02d}'
        folder.mkdir()
        shutil.copy2(run/f'rtl_pursuit/pursuit_l1_l4/sample_{sample:02d}/simulation.log',folder/'simulation.log')
    row=dict(variant=name,depth=4,user=10,seed=2021,core_cycles=CYCLES[4],
             frequency_mhz=160,core_latency_ms=CYCLES[4]/160000,
             retained_cross_step_states=1992,resources=resources,
             timing=dict(setup_slack_ns=setup,hold_slack_ns=hold,pulse_width_slack_ns=pulse,checks=checks),
             rtl_samples=16,rtl_values_checked=sum(r['values_checked'] for r in replay['records']),
             rtl_mismatches=0,source_sha256=source_hashes,
             report_sha256={p.name:sha(p) for p in (target/'reports').glob('*')},
             implementation_checkpoint_sha256=sha(target/'post_route.dcp'),
             uncompressed_funcsim_sha256=sha(original_netlist),
             power_evaluated=False)
    (target/'summary.json').write_text(json.dumps(row,indent=2)+'\n')
    (rtl_target/'report.json').write_text(json.dumps(replay,indent=2)+'\n')
    rows=[]
    for depth,item in enumerate(json.loads((ROOT/'results/summary.json').read_text())['variants']):
        rows.append(dict(variant=item['variant'],stateless_depth=depth,
                         core_cycles=item['performance']['core_cycles'],frequency_mhz=160,
                         core_latency_ms=item['performance']['core_latency_ms'],
                         bram36_equivalent=item['resources']['bram36_equivalent']['used']))
    rows.append(dict(variant=name,stateless_depth=4,core_cycles=CYCLES[4],frequency_mhz=160,
                     core_latency_ms=row['core_latency_ms'],bram36_equivalent=resources['bram36_equivalent']))
    result=dict(scope='five matched 400-event, T2 state-depth implementations',
                timing_passed_for_all_variants=True,latency_scope='core computation after inputs and weights are on chip',
                variants=rows,l1_l4_evidence=str((target/'summary.json').relative_to(ROOT)))
    (ROOT/'results/latency_bram.json').write_text(json.dumps(result,indent=2)+'\n')
    with (ROOT/'results/latency_bram.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(row))


if __name__=='__main__':
    main()
