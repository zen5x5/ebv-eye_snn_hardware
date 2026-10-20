"""Independently check saved 24-user state-depth quantization groups."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

METRICS = ('mean_angular_error_deg', 'mean_pixel_error')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--check-original-inputs',action='store_true')
    args = parser.parse_args()
    root = args.directory.resolve()
    result = json.loads((root/'results.json').read_text())
    manifest = json.loads((root/'input_manifest.json').read_text())
    configuration = json.loads((root/'configuration.json').read_text())
    expected = {(t,v,u) for t in configuration['tasks'] for v in configuration['variants'] for u in range(4,28)}
    assert len(result['subjects'])==len(expected) and len(manifest)==len(expected)
    assert {(r['task'],r['variant'],r['user']) for r in result['subjects']} == expected
    checked_inputs = {}
    new = 0
    retained = 0
    tested_samples = 0
    for row, record in zip(result['subjects'],manifest):
        folder = root/row['output']
        assert json.loads((folder/'summary.json').read_text()) == row
        assert record['input_sha256'] == row['input_sha256']
        source = root/row['model']
        for name in ('configuration.json','summary.json','best_checkpoint.pt'):
            original = next(p for p in row['input_sha256'] if p.endswith('/'+name))
            assert sha(source/name) == row['input_sha256'][original]
        test = root/row['test']
        original = next(p for p in row['input_sha256'] if p.endswith('/test.npz'))
        if test not in checked_inputs:
            checked_inputs[test] = sha(test)
        assert checked_inputs[test] == row['input_sha256'][original]
        if args.check_original_inputs:
            for original, expected_hash in row['input_sha256'].items():
                path = Path(original)
                if path not in checked_inputs:
                    checked_inputs[path] = sha(path)
                assert checked_inputs[path] == expected_hash
        if row['execution']=='retained':
            assert sha(folder/'retained_source.json') == row['retained_sha256']
            retained += 1
            continue
        assert row['execution']=='new_cpu' and row['independent_probe_arrays']==61
        with np.load(folder/'predictions_test.npz',allow_pickle=False) as data:
            normalized = data['prediction_normalized_xy'].astype(np.float64)
            labels = data['target_row_col_px']
            pixels = np.column_stack((normalized[:,1]*540+540, normalized[:,0]*960+960))
            np.testing.assert_array_equal(pixels,data['prediction_row_col_px'])
            with np.load(test,allow_pickle=False) as cache:
                np.testing.assert_array_equal(labels,cache['labels'].astype(np.float64))
            np.testing.assert_array_equal(data['representation_index'],np.arange(len(labels)))
            mm_per_px = 40*25.4/(1920**2+1080**2)**0.5
            def rays(points):
                return np.column_stack(((points[:,1]-960)*mm_per_px,
                                        (points[:,0]-540)*mm_per_px,np.full(len(points),400)))
            a,b = rays(pixels),rays(labels)
            angular = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),(a*b).sum(axis=1)))
            distance = np.linalg.norm(pixels-labels,axis=1)
            np.testing.assert_allclose(angular,data['angular_error_deg'],rtol=0,atol=1e-12)
            np.testing.assert_array_equal(distance,data['pixel_error'])
            assert abs(float(angular.mean())-row['fixed_point'][METRICS[0]])<1e-12
            assert abs(float(distance.mean())-row['fixed_point'][METRICS[1]])<1e-12
            assert len(labels)==row['representations']
            tested_samples += len(labels)
        new += 1
    for aggregate in result['aggregates']:
        group=[r for r in result['subjects'] if r['task']==aggregate['task'] and r['variant']==aggregate['variant']]
        assert len(group)==24
        assert sum(r['representations'] for r in group)==aggregate['representations']
        assert aggregate['representations']=={'pursuit':25716,'saccade':28389}[aggregate['task']]
        for precision in ('official_float','fixed_point'):
            for metric in METRICS:
                assert statistics.mean(r[precision][metric] for r in group)==aggregate[precision][metric]
    assert new==result['new_models'] and retained==result['retained_models']
    assert len(result['aggregates'])==len(configuration['tasks'])*len(configuration['variants'])
    print(json.dumps(dict(status='passed',models=len(expected),new_cpu_models=new,retained_models=retained,
                          new_test_predictions=tested_samples,aggregates=len(result['aggregates']),
                          original_inputs_checked=args.check_original_inputs)),flush=True)


if __name__=='__main__':
    main()
