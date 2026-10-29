from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pbit.augmentation import augment_training_set  # noqa: E402
from pbit.data_validation import load_long_dataset  # noqa: E402
from pbit.features import extract_features  # noqa: E402
from pbit.modeling import (  # noqa: E402
    FAULT_ORDER,
    META_COLS,
    SEVERITY_MAP,
    compute_safety_metrics,
    compute_sample_weights,
    evaluate_predictions,
)

DATA = PROJECT_ROOT / "data"
SEV_LABELS = [0, 1, 2, 3]
EXPECTED = {
    "F1-tuned": {"cmr_count": 6, "macro_f1": 0.3898},
    "CMR-tuned": {"cmr_count": 0, "macro_f1": 0.2704},
}


def load_params():
    f1 = json.loads((DATA / "best_lgbm_params.json").read_text())
    cmr = json.loads((DATA / "best_lgbm_cmr_params.json").read_text())
    f1_f05 = f1.pop("f05_boost")
    cmr_f05 = cmr.pop("f05_boost")
    cmr_sev3 = cmr.pop("sev3_boost")
    return f1, f1_f05, cmr, cmr_f05, cmr_sev3


def train_models(train_df, aug_feat, feat_cols, params):
    f1_params, f1_f05, cmr_params, cmr_f05, cmr_sev3 = params
    full = pd.concat([train_df, aug_feat], ignore_index=True) if len(aug_feat) else train_df
    X = full[feat_cols].values.astype(float)
    y = full["Fault_Mode"].values

    sw1 = compute_sample_weights(y, "freq_sev")
    sw1[y == "F05"] *= f1_f05
    m1 = lgb.LGBMClassifier(**f1_params).fit(X, y, sample_weight=sw1)

    sw2 = compute_sample_weights(y, "freq_sev")
    sw2[y == "F05"] *= cmr_f05
    sw2[np.isin(y, ["F03", "F06"])] *= cmr_sev3
    m2 = lgb.LGBMClassifier(**cmr_params).fit(X, y, sample_weight=sw2)
    return {"F1-tuned": m1, "CMR-tuned": m2}, full["Fault_Mode"].value_counts()


def sev(arr):
    return np.array([SEVERITY_MAP[c] for c in arr])


def direction_counts(y_true, y_pred):
    st, sp = sev(y_true), sev(y_pred)
    s3 = st == 3
    return {
        "sev3_total": int(s3.sum()),
        "sev3_correct_class": int((y_true[s3] == y_pred[s3]).sum()),
        "sev3_lateral_same_severity": int(((sp == 3) & (y_true != y_pred) & s3).sum()),
        "sev3_down_to_sev0": int(((sp == 0) & s3).sum()),
        "sev3_down_to_sev1_2": int(((sp > 0) & (sp < 3) & s3).sum()),
        "R3": float(((sp == 3) & s3).sum() / s3.sum()),
        "D3": float(((sp < 3) & s3).sum() / s3.sum()),
    }


def md_table(df: pd.DataFrame) -> str:
    df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def main() -> None:
    train_df = pd.read_csv(DATA / "train_real.csv")
    test_df = pd.read_csv(DATA / "test_real.csv")
    long_df = load_long_dataset(DATA / "PBIT_atuador_DESCARAC_r1_long_clean.csv")
    feat_cols = [c for c in train_df.columns if c not in META_COLS]
    params = load_params()
    X_test = test_df[feat_cols].values.astype(float)
    y_test = test_df["Fault_Mode"].values

    ids = set(train_df["Experiment_Index"].tolist())
    aug_all = extract_features(
        augment_training_set(long_df, ids, target_per_class=160, noise_scale=0.01, random_state=42)
    )
    aug_no_f06 = extract_features(
        augment_training_set(long_df, ids, target_per_class={"F04": 160, "F05": 160},
                             noise_scale=0.01, random_state=42)
    )

    variants = {
        "main (F04/F05/F06 aug 160)": aug_all,
        "control (F06 not augmented)": aug_no_f06,
    }
    preds = test_df[["Experiment_Index", "Serial_Number", "Fault_Mode"]].copy()
    out = ["# Reanálise do teste final por severidade", ""]
    summary_rows, dir_rows, sn_rows = [], [], []
    cms = {}

    for vname, aug in variants.items():
        models, counts = train_models(train_df, aug, feat_cols, params)
        out += [f"## Variante: {vname}", "",
                "Contagem de treino por classe: " + ", ".join(f"{c} = {counts.get(c, 0)}" for c in FAULT_ORDER), ""]
        for mname, model in models.items():
            yp = model.predict(X_test)
            tag = f"{'main' if vname.startswith('main') else 'ctrl'}_{mname}"
            preds[f"pred_{tag}"] = yp
            s = compute_safety_metrics(y_test, yp)
            labels = [c for c in FAULT_ORDER if c in set(y_test)]
            mf1 = evaluate_predictions(y_test, yp)["macro_f1"]
            mf1_present = f1_score(y_test, yp, average="macro", labels=labels, zero_division=0)
            summary_rows.append({
                "variante": vname.split(" ")[0], "modelo": mname,
                "CMR": round(s["cmr"], 4), "CMR_contagem": f"{s['cmr_count']}/{s['crit_total']}",
                "DR": round(s["down_risk"], 4), "SE": round(s["sev_err"], 4),
                "macro_F1": round(mf1, 4), "macro_F1_7_classes_com_N02_ausente": round(mf1_present, 4),
            })
            d = direction_counts(y_test, yp)
            dir_rows.append({"variante": vname.split(" ")[0], "modelo": mname, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}})
            cms[(vname.split(" ")[0], mname)] = (
                confusion_matrix(sev(y_test), sev(yp), labels=SEV_LABELS),
                confusion_matrix(y_test, yp, labels=FAULT_ORDER),
            )
            s3 = np.isin(y_test, ["F03", "F06"])
            sub = preds.loc[s3, ["Serial_Number", "Fault_Mode"]].copy()
            sub["pred"] = yp[s3]
            for (sn, fm), g in sub.groupby(["Serial_Number", "Fault_Mode"]):
                sn_rows.append({
                    "variante": vname.split(" ")[0], "modelo": mname, "SN": sn, "classe": fm,
                    "ciclos": len(g), "previsao": ", ".join(f"{k} = {v}" for k, v in g["pred"].value_counts().items()),
                })

    check = []
    sm = {(r["variante"], r["modelo"]): r for r in summary_rows}
    for mname, exp in EXPECTED.items():
        r = sm[("main", mname)]
        check.append({
            "modelo": mname,
            "CMR_contagem_obtida": r["CMR_contagem"], "CMR_contagem_artigo": f"{exp['cmr_count']}/74",
            "macroF1_obtido": r["macro_F1"], "macroF1_artigo": exp["macro_f1"],
            "reproduzido": bool(r["CMR_contagem"].split("/")[0] == str(exp["cmr_count"])
                                and abs(r["macro_F1"] - exp["macro_f1"]) <= 0.001),
        })
    out += ["## Verificação de reprodução", "", md_table(pd.DataFrame(check).set_index("modelo")), ""]
    out += ["## Métricas de segurança", "", md_table(pd.DataFrame(summary_rows).set_index("variante")), ""]
    out += ["## Direção dos erros de Sev3", "", md_table(pd.DataFrame(dir_rows).set_index("variante")), ""]
    out += ["## Previsões para ciclos de Sev3 por SN", "", md_table(pd.DataFrame(sn_rows).set_index("variante")), ""]
    for (v, m), (cs, cc) in cms.items():
        out += [f"## Matriz de confusão por severidade (linhas = verdadeiro, colunas = previsto): {v}, {m}", "",
                md_table(pd.DataFrame(cs, index=[f"Sev{i}" for i in SEV_LABELS], columns=[f"Sev{i}" for i in SEV_LABELS])), ""]
        out += [f"## Matriz de confusão por classe: {v}, {m}", "",
                md_table(pd.DataFrame(cc, index=FAULT_ORDER, columns=FAULT_ORDER)), ""]

    preds.to_csv(DATA / "final_test_predictions.csv", index=False)
    (DATA / "final_test_reanalysis.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
