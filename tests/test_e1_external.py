from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e1_external_analysis as e1

SEV = dict(zip(e1.FAULT_ORDER, e1.SEVERITY))


def synthetic(seed=0, cycles=400, units=30):
    rng = np.random.default_rng(seed)
    truth = rng.choice(e1.FAULT_ORDER, size=cycles, p=[.3, .05, .1, .2, .15, .07, .08, .05])
    pred = np.where(rng.random(cycles) < 0.5, truth, rng.choice(e1.FAULT_ORDER, size=cycles))
    sn = rng.integers(0, units, size=cycles)
    return truth.tolist(), pred.tolist(), [f"SN{u:03d}" for u in sn]


def scalar_metrics(truth, pred, sn):
    st = np.array([SEV[t] for t in truth])
    sp = np.array([SEV[p] for p in pred])
    cmr = sum(1 for a, b in zip(st, sp) if a == 3 and b == 0)/sum(1 for a in st if a == 3)
    fa0 = sum(1 for a, b in zip(st, sp) if a == 0 and b >= 1)/sum(1 for a in st if a == 0)
    dr = sum(1 for a, b in zip(st, sp) if b < a)/len(st)
    r3 = sum(1 for a, b in zip(st, sp) if a == 3 and b == 3)/sum(1 for a in st if a == 3)
    losses, alarms = [], []
    for u in sorted(set(sn)):
        idx = [i for i, x in enumerate(sn) if x == u]
        crit = [i for i in idx if st[i] == 3]
        norm = [i for i in idx if st[i] == 0]
        if crit:
            losses.append(sum(sp[i] == 0 for i in crit)/len(crit))
        if norm:
            alarms.append(sum(sp[i] >= 1 for i in norm)/len(norm))
    labels = sorted(set(truth) | set(pred))
    f1 = f1_score(truth, pred, labels=labels, average="macro", zero_division=0)
    return dict(CMR=cmr, Lbar=np.mean(losses), FA0=fa0, FA0_unit=np.mean(alarms), DR=dr, R3=r3, macro_F1=f1)


class MetricsTests(unittest.TestCase):
    def test_metrics_match_scalar_loops_and_sklearn(self):
        for seed in range(3):
            truth, pred, sn = synthetic(seed)
            tensor, order = e1.confusion_by_unit(truth, pred, sn)
            got = e1.metrics_from(tensor, np.ones((1, len(order))))
            want = scalar_metrics(truth, pred, sn)
            for name in e1.METRICS:
                self.assertAlmostEqual(float(got[name][0]), float(want[name]), places=10, msg=name)

    def test_bootstrap_weights_match_explicit_resampling(self):
        truth, pred, sn = synthetic(5)
        tensor, order = e1.confusion_by_unit(truth, pred, sn)
        rng = np.random.default_rng(3)
        picks = rng.integers(0, len(order), size=len(order))
        weights = np.bincount(picks, minlength=len(order)).astype(float)[None, :]
        got = e1.metrics_from(tensor, weights)
        rows = [i for u in picks for i in range(len(sn)) if sn[i] == order[u]]
        resampled_sn = [f"{sn[i]}#{k}" for k, u in enumerate(picks) for i in range(len(sn)) if sn[i] == order[u]]
        want = scalar_metrics([truth[i] for i in rows], [pred[i] for i in rows], resampled_sn)
        for name in ("CMR", "FA0", "DR", "R3", "macro_F1"):
            self.assertAlmostEqual(float(got[name][0]), float(want[name]), places=10, msg=name)


class LabelTests(unittest.TestCase):
    def test_every_combination_has_the_protocol_label(self):
        melhoria = "Melhoria aparente do compromisso (não confirmada)"
        expected = {
            ("menor", "maior"): "Deslocamento ao longo do compromisso",
            ("menor", "semelhante"): melhoria, ("menor", "menor"): melhoria,
            ("semelhante", "menor"): melhoria,
            ("semelhante", "semelhante"): "Sem diferença apreciável",
            ("semelhante", "maior"): "Piora aparente", ("maior", "maior"): "Piora aparente",
            ("maior", "menor"): "Deslocamento inverso", ("maior", "semelhante"): "Deslocamento inverso"}
        value = {"menor": {"m": -0.05, "f": -0.06}, "semelhante": {"m": 0.0, "f": 0.0},
                 "maior": {"m": 0.05, "f": 0.06}}
        for (m, f), label in expected.items():
            got, mc, fc = e1.classify(value[m]["m"], value[f]["f"])
            self.assertEqual((got, mc, fc), (label, m, f))

    def test_thresholds_are_inclusive(self):
        self.assertEqual(e1.classify(-0.02, 0.0)[1], "menor")
        self.assertEqual(e1.classify(0.02, 0.0)[1], "maior")
        self.assertEqual(e1.classify(-0.0199, 0.03)[1:], ("semelhante", "maior"))
        self.assertEqual(e1.classify(0.0, -0.03)[2], "menor")


class RetrospectiveCurveTests(unittest.TestCase):
    def test_curve_matches_exhaustive_scalar_search(self):
        rng = np.random.default_rng(2)
        n = 120
        probs = rng.dirichlet(np.ones(8), size=n)
        truth = rng.choice(e1.FAULT_ORDER, size=n, p=[.3, .05, .1, .2, .15, .07, .08, .05])
        base = np.array(e1.FAULT_ORDER)[probs.argmax(axis=1)]
        data = pd.DataFrame({"Fault_Mode": truth, "untuned_base": base})
        for k, c in enumerate(e1.FAULT_ORDER):
            data[f"p_{c}"] = probs[:, k]
        score = probs[:, 4]+probs[:, 7]
        for limit in (0.0, 0.3, 0.5, 1.0):
            best = np.inf
            for tau in [np.inf, 0.0, *score]:
                pred = [3 if s >= tau else SEV[b] for s, b in zip(score, base)]
                st = [SEV[t] for t in truth]
                norm = [i for i, a in enumerate(st) if a == 0]
                crit = [i for i, a in enumerate(st) if a == 3]
                fa0 = sum(pred[i] >= 1 for i in norm)/len(norm)
                if fa0 <= limit+1e-12:
                    best = min(best, sum(pred[i] == 0 for i in crit)/len(crit))
            self.assertAlmostEqual(e1.retrospective_curve_cmr(data, limit), best, places=12)


class GuardTests(unittest.TestCase):
    def test_refuses_when_output_exists(self):
        original = e1.OUT
        try:
            e1.OUT = ROOT
            with self.assertRaises(AssertionError):
                e1.refuse_if_computed()
        finally:
            e1.OUT = original


if __name__ == "__main__":
    unittest.main()
