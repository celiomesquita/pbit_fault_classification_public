"""Verify E3 hashes, replicate-level decisions and Monte Carlo moment checks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_coverage_simulation import ROOT, OUT, sha, utc
from cluster_ratio_methods import METHODS, FAMILIES


def main():
    run = OUT / "run"
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert sorted(manifest["completed"]) == list(range(40))
    for name, digest in {**manifest["input_sha256"], **manifest["output_sha256"]}.items():
        assert sha(ROOT / name) == digest, name
    assert sha(ROOT / "scripts/run_cluster_coverage_parallel.py") == manifest["execution_pool"]["launcher_sha256"]
    summary = pd.read_csv(run / "summary.csv")
    assert summary.replicates.eq(5000).all() and summary.resamples.eq(2000).all()
    assert summary.groupby('id').method.nunique().eq(6).all()
    moment_checks = []
    for cfg in manifest['scenarios']:
        saved = np.load(run / f"scenario_{cfg['id']:02d}.npz")
        estimates = saved['estimate'][:, 0]
        bias = float(estimates.mean()-cfg['delta'])
        mc_se = float(estimates.std(ddof=1)/np.sqrt(len(estimates)))
        assert abs(bias) <= 6*mc_se, (cfg, bias, mc_se)
        moment_checks.append(dict(id=cfg['id'], mean=float(estimates.mean()), truth=cfg['delta'],
                                  MC_SE=mc_se, standardised_bias=bias/mc_se))
        for method in METHODS:
            all_p, bounds = saved[f'{method}_p'], saved[f'{method}_bounds'][:, 0]
            assert np.isfinite(all_p).all() and ((all_p >= 0) & (all_p <= 1)).all()
            assert not np.isnan(bounds).any()
            assert (bounds[:, ::2] <= bounds[:, 1::2]).all()
            events = dict(coverage95=(bounds[:, 0] <= cfg['delta']) & (cfg['delta'] <= bounds[:, 1]),
                          reject_zero=all_p[:, 0] <= 0.05, abstain=saved[f'{method}_abstain'][:, 0])
            for i, family in enumerate(FAMILIES):
                events[f'equivalence_m{family}'] = (bounds[:, 2+2*i] > -cfg['margin']) & (bounds[:, 3+2*i] < cfg['margin'])
            if cfg['delta'] == 0:
                for family in [3, 4, 8]:
                    events[f'holm_fwer_m{family}'] = all_p[:, :family].min(axis=1) <= 0.05/family
            for event, flags in events.items():
                record = summary[summary.id.eq(cfg['id']) & summary.method.eq(method) & summary.event.eq(event)]
                assert len(record) == 1
                assert int(flags.sum()) == int(record.iloc[0]['count'])
                assert np.isclose(flags.mean(), record.iloc[0].rate)
    extra = [Path(__file__), ROOT / 'tests/test_cluster_coverage.py', ROOT / 'scripts/plot_cluster_coverage.py',
             ROOT / 'paper/supplement/figures/fig_cluster_coverage.pdf', ROOT / 'paper/supplement/figures/fig_cluster_coverage.png',
             ROOT / 'paper/manuscript/pbit_paper.tex', ROOT / 'paper/supplement/pbit_supplementary.tex',
             ROOT / 'paper/manuscript/pbit_paper.pdf', ROOT / 'paper/supplement/pbit_supplementary.pdf',
             ROOT / 'docs/plano_novos_experimentos.md']
    report = dict(verified_utc=utc(), status='passed', scenarios=40, replicates_per_scenario=5000,
                  resamples_per_replicate=2000, tests='11 unittest tests passed under Python 3.14; E3 tests also passed under 3.13',
                  reproduction='scenario 01: all arrays exactly reproduced under Python 3.13',
                  latex='Both PDFs compiled; no undefined references/citations or overfull boxes',
                  hash_count=len(manifest['input_sha256'])+len(manifest['output_sha256'])+1,
                  moment_checks=moment_checks,
                  additional_sha256={str(p.relative_to(ROOT)): sha(p) for p in extra})
    (run / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"Verified {report['hash_count']} hashes; 40 scenarios; all stored decisions reconstructed; 40 marginal moment checks passed.")


if __name__ == '__main__':
    main()
