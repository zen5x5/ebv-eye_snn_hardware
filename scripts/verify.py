"""Verify retained evidence and paper metrics without simulation or training."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
from _shared import ROOT, VARIANTS, CYCLES


def digest(stream):
    value = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024*1024), b''):
        value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='also decompress and hash all retained compressed originals')
    args = parser.parse_args()
    checked = 0
    for line in (ROOT/'MANIFEST.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        with (ROOT/name).open('rb') as stream:
            assert digest(stream) == expected, name
        checked += 1
    provenance = json.loads((ROOT/'results/provenance/files.json').read_text())
    original = 0
    for record in provenance['files']:
        path = ROOT/record['path']
        if record['encoding'] == 'original':
            with path.open('rb') as stream:
                assert digest(stream) == record['source_sha256'], path
            original += 1
        elif record['encoding'] == 'gzip' and args.full:
            with gzip.open(path, 'rb') as stream:
                assert digest(stream) == record['source_sha256'], path
            original += 1
    summary = json.loads((ROOT/'results/summary.json').read_text())
    replay = json.loads((ROOT/'results/rtl/report.json').read_text())
    assert len(summary['variants']) == 4
    assert sum(record['values_checked'] for record in replay['variants'].values()) == 12340224
    for depth, item in enumerate(summary['variants']):
        name = item['variant']
        assert name == VARIANTS[depth]
        assert item['implementation_checks_passed']
        assert item['performance']['core_cycles'] == CYCLES[depth]
        assert all(item['timing'][key] >= 0 for key in ('setup_slack_ns', 'hold_slack_ns', 'pulse_width_slack_ns'))
        assert item['power']['user'] == 10 and item['power']['samples'] == 16
        assert not item['power']['board_power_measured']
        assert replay['variants'][name]['mismatches'] == 0
        assert replay['variants'][name]['samples'] == 16
        quant = json.loads((ROOT/'results/quantization/user10'/f'{name}.json').read_text())
        with (ROOT/'models'/name/'best_checkpoint.pt').open('rb') as stream:
            assert digest(stream) == quant['checkpoint_sha256']
        for filename, expected in item['source_sha256'].items():
            if filename.endswith('.sv'):
                with (ROOT/'hardware/rtl'/filename).open('rb') as stream:
                    assert digest(stream) == expected
        for name_report, expected in item['report_sha256'].items():
            with (ROOT/'results/implementation'/name/'reports'/name_report).open('rb') as stream:
                assert digest(stream) == expected
        energies = []
        for sample in range(16):
            folder = ROOT/'results/power'/name/f'sample_{sample:02d}'
            log = (folder/'simulation.log').read_text()
            assert 'PASS: post-route power replay' in log
            record = json.loads((folder/'summary.json').read_text())
            duration = float(record['power_window'][1])
            assert duration == CYCLES[depth]*6.25
            text = (folder/'power.rpt').read_text()
            power = float(re.search(r'^\|\s*Total On-Chip Power \(W\)\s*\|\s*([\d.]+)', text, re.M)[1])
            energies.append(power*duration/1000)
        assert abs(sum(energies)/16-item['power']['metrics']['total_energy_uj']['mean']) < 1e-9
    quant = json.loads((ROOT/'results/quantization/multiuser/results.json').read_text())
    assert len(quant['subjects']) == 48
    assert quant['aggregates']['pursuit']['subjects'] == 24
    print(json.dumps(dict(status='passed', manifest_files=checked, original_payloads_verified=original,
                          variants=4, rtl_samples=64, power_samples=64, full_decompression=args.full)))


if __name__ == '__main__':
    main()
