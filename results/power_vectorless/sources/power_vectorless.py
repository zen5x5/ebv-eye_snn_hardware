"""Estimate default-activity power for all five retained routed designs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from time import monotonic

from _shared import ROOT, VARIANTS, CYCLES, run, sha, tool


def report_value(report, label):
    return re.search(r'^\|\s*' + re.escape(label) + r'\s*\|\s*([^|]+)',
                     report, re.MULTILINE).group(1).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    sources = out / 'sources'
    sources.mkdir()
    for relative in ('scripts/power_vectorless.py', 'scripts/_shared.py',
                     'hardware/vivado/power_vectorless.tcl'):
        shutil.copy2(ROOT / relative, sources / Path(relative).name)
    executable = tool('vivado')
    started_at = datetime.now(timezone.utc).isoformat()
    started = monotonic()

    def estimate(item):
        depth, variant, cycles = item
        folder = out / variant
        folder.mkdir()
        checkpoint = ROOT / 'results/implementation' / variant / 'post_route.dcp'
        checkpoint_sha = sha(checkpoint)
        task_started = monotonic()
        print(f'START {variant}', flush=True)
        run([executable, '-mode', 'batch', '-nojournal', '-log', folder / 'vivado.log',
             '-source', sources / 'power_vectorless.tcl', '-tclargs', checkpoint,
             folder / 'reports'], folder / 'console.log', folder)
        assert sha(checkpoint) == checkpoint_sha, checkpoint
        reports = folder / 'reports'
        report = (reports / 'power.rpt').read_text()
        settings = dict(line.split('=', 1) for line in
                        (reports / 'settings.txt').read_text().splitlines())
        assert settings['part'] == 'xczu3eg-sfvc784-2-e', settings
        assert float(settings['clock_period_ns']) == 6.25, settings
        total = float(report_value(report, 'Total On-Chip Power (W)'))
        dynamic = float(report_value(report, 'Dynamic (W)'))
        static = float(report_value(report, 'Device Static (W)'))
        row = dict(variant=variant, stateless_depth=depth, core_cycles=cycles,
                   core_ms=cycles / 160000, total_w=total, dynamic_w=dynamic,
                   device_static_w=static, total_energy_uj=total * cycles / 160,
                   junction_temperature_c=float(report_value(report, 'Junction Temperature (C)')),
                   confidence=report_value(report, 'Confidence Level'),
                   checkpoint_sha256=checkpoint_sha,
                   settings=settings,
                   reports_sha256={p.name: sha(p) for p in sorted(reports.iterdir())},
                   elapsed_seconds=monotonic() - task_started)
        assert row['junction_temperature_c'] == 25, row
        (folder / 'result.json').write_text(json.dumps(row, indent=2) + '\n')
        print(f'DONE {variant}: {total:.3f} W, {row["elapsed_seconds"]:.1f} s', flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(estimate, zip(range(5), VARIANTS, CYCLES)))
    summary = dict(method='vectorless_default', frequency_mhz=160,
                   process='typical', junction_temperature_c=25,
                   voltage='nominal', simulation_activity_file=None,
                   tasks=['pursuit', 'saccade_continuous_xy'],
                   task_equivalence='Identical runtime-programmed circuit at matched depth; '
                                    'no task-specific activity is supplied.',
                   energy_scope='Total on-chip estimate multiplied by measured core cycles / '
                                '160 MHz; excludes host and board peripherals.',
                   started_at_utc=started_at, elapsed_seconds=monotonic() - started,
                   sources_sha256={p.name: sha(p) for p in sorted(sources.iterdir())},
                   variants=rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    columns = ('variant', 'stateless_depth', 'core_cycles', 'core_ms', 'total_w',
               'dynamic_w', 'device_static_w', 'total_energy_uj', 'confidence')
    with (out / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f'COMPLETE: {out / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
