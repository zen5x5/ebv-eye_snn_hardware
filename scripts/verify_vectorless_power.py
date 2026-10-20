"""Check saved default-activity power, XML agreement and unchanged routed designs."""
import argparse
import csv
from decimal import Decimal
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from _shared import ROOT, VARIANTS, CYCLES, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['method'] == 'vectorless_default'
    assert summary['simulation_activity_file'] is None
    assert summary['frequency_mhz'] == 160
    assert summary['tasks'] == ['pursuit', 'saccade_continuous_xy']
    assert len(summary['variants']) == 5
    for name, expected in summary['sources_sha256'].items():
        assert sha(directory / 'sources' / name) == expected, name
    with (directory / 'summary.csv').open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 5
    for depth, (name, cycles, row, csv_row) in enumerate(zip(
            VARIANTS, CYCLES, summary['variants'], csv_rows)):
        assert row['variant'] == csv_row['variant'] == name
        assert row['stateless_depth'] == depth and row['core_cycles'] == cycles
        assert sha(ROOT / 'results/implementation' / name / 'post_route.dcp') == row['checkpoint_sha256']
        folder = directory / name
        assert json.loads((folder / 'result.json').read_text()) == row
        for filename, expected in row['reports_sha256'].items():
            assert sha(folder / 'reports' / filename) == expected, (name, filename)
        xml = ET.parse(folder / 'reports/power.xml').getroot()
        table = xml.find("./section[@title='Summary']/table")
        cells = [[c.attrib['contents'].strip() for c in r.findall('tablecell')]
                 for r in table.findall('tablerow')]
        values = {cells[0]: cells[1] for cells in cells if len(cells) == 2}
        for label, key in (('Total On-Chip Power (W)', 'total_w'),
                           ('Dynamic (W)', 'dynamic_w'), ('Device Static (W)', 'device_static_w')):
            assert float(values[label]) == row[key] == float(csv_row[key]), (name, label)
        assert values['Simulation Activity File'] == '---'
        assert float(values['Junction Temperature (C)']) == 25
        assert values['Confidence Level'] == row['confidence']
        difference = Decimal(values['Total On-Chip Power (W)']) - (
            Decimal(values['Dynamic (W)']) + Decimal(values['Device Static (W)']))
        # Each reported power is rounded by Vivado to 0.001 W.
        assert abs(difference) <= Decimal('0.001'), name
        assert row['core_ms'] == cycles / 160000
        assert row['total_energy_uj'] == row['total_w'] * cycles / 160
        activity = (folder / 'reports/input_activity.rpt').read_text()
        assert '(S)' not in activity and '(U)' not in activity, name
        assert 'clk:  static probability = 0.5 (C)  signal rate = 320 (C)' in activity
        assert 'host_write_enable:  static probability = 0.5 (D)  signal rate = 20 (D)' in activity
        assert 'rst:  static probability = 0 (D)  signal rate = 0 (D)' in activity
        log = (folder / 'console.log').read_text()
        assert 'ERROR:' not in log and 'CRITICAL WARNING:' not in log, name
        assert 'read_saif' not in log, name
        assert re.search(r'(?m)^(?:# )?set_switching_activity\b', log) is None, name
        assert 'report_power completed successfully' in log, name
    result = dict(status='passed', variants=5, represented_tasks=2,
                  report_xml_agreement=True, default_activity_verified=True,
                  routed_checkpoints_unchanged=True,
                  confidence_levels=[r['confidence'] for r in summary['variants']])
    (directory / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
