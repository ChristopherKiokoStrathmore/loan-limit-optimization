"""
ML layer: PD, offer acceptance, and utilization response models
with train / out-of-time-style holdout validation on synthetic data.
MSc DS&BA framing — predictive models feed OR allocation.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from loan_limit_engine import (
    DATA,
    FIG,
    RNG,
    SCENARIOS,
    TIERS,
    acceptance_prob,
    build_synthetic_portfolio,
    pd_after_increase,
    utilization_response,
)

ROOT = Path(__file__).resolve().parent


def simulate_labeled_offers(df: pd.DataFrame, n_offers_per: int = 1) -> pd.DataFrame:
    """Generate synthetic historical offer outcomes for supervised learning."""
    rng = RNG
    rows = []
    for _, r in df.iterrows():
        for _ in range(n_offers_per):
            tier = float(rng.choice(TIERS[1:]))  # historical offers were non-zero
            p_acc = acceptance_prob(tier, r.utilization, r.base_pd, 1.0)
            accepted = bool(rng.random() < p_acc)
            util_n = utilization_response(r.utilization, tier, accepted, 1.0)
            # observation noise
            util_n = float(np.clip(util_n + rng.normal(0, 0.03), 0.05, 0.99))
            pd_n = pd_after_increase(r.base_pd, tier, accepted, util_n, r.utilization)
            # binary default label over horizon (Bernoulli)
            defaulted = int(rng.random() < pd_n)
            rows.append(
                {
                    "borrower_id": r.borrower_id,
                    "product": r["product"],
                    "tenure_m": r.tenure_m,
                    "tenure_bucket": r.tenure_bucket,
                    "pd_band": r.pd_band,
                    "util_bucket": r.util_bucket,
                    "current_limit": r.current_limit,
                    "utilization": r.utilization,
                    "monthly_income": r.monthly_income,
                    "dti": r.dti,
                    "quality": r.quality,
                    "offer_tier": tier,
                    "accepted": int(accepted),
                    "util_after": util_n,
                    "base_pd": r.base_pd,
                    "defaulted": defaulted,
                    # pseudo time split key: earlier accounts = train
                    "vintage_score": r.tenure_m + rng.normal(0, 2),
                }
            )
    return pd.DataFrame(rows)


def _preprocess(cat_cols, num_cols):
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )


def train_and_validate(offers: pd.DataFrame) -> dict:
    """Out-of-time style split: lower vintage_score → train, higher → test."""
    offers = offers.sort_values("vintage_score")
    cut = int(0.7 * len(offers))
    train, test = offers.iloc[:cut].copy(), offers.iloc[cut:].copy()

    cat = ["product", "tenure_bucket", "pd_band", "util_bucket"]
    num_pd = ["tenure_m", "utilization", "current_limit", "monthly_income", "dti", "offer_tier"]
    num_acc = num_pd
    num_util = ["tenure_m", "utilization", "current_limit", "dti", "offer_tier"]

    results = {"n_train": len(train), "n_test": len(test), "models": {}}

    # --- PD / default classifier (interpretable logit + GBM) ---
    Xtr, Xte = train[cat + num_pd], test[cat + num_pd]
    ytr, yte = train["defaulted"], test["defaulted"]

    logit = Pipeline(
        [
            ("prep", _preprocess(cat, num_pd)),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            ),
        ]
    )
    gbm_pd = Pipeline(
        [
            ("prep", _preprocess(cat, num_pd)),
            (
                "clf",
                GradientBoostingClassifier(random_state=42, max_depth=3, n_estimators=80),
            ),
        ]
    )
    logit.fit(Xtr, ytr)
    gbm_pd.fit(Xtr, ytr)

    def clf_metrics(model, name):
        proba = model.predict_proba(Xte)[:, 1]
        return {
            "name": name,
            "auc": float(roc_auc_score(yte, proba)),
            "brier": float(brier_score_loss(yte, proba)),
            "ap": float(average_precision_score(yte, proba)),
            "proba_test": proba,
        }

    m_logit = clf_metrics(logit, "logistic_pd")
    m_gbm = clf_metrics(gbm_pd, "gbm_pd")
    results["models"]["pd_logistic"] = {k: v for k, v in m_logit.items() if k != "proba_test"}
    results["models"]["pd_gbm"] = {k: v for k, v in m_gbm.items() if k != "proba_test"}

    # ROC curve
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for m, color in [(m_logit, "#2980b9"), (m_gbm, "#27ae60")]:
        fpr, tpr, _ = roc_curve(yte, m["proba_test"])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{m['name']} AUC={m['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Out-of-time PD Model ROC (Synthetic Offers)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "pd_roc_oot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Acceptance classifier ---
    Xtr_a, Xte_a = train[cat + num_acc], test[cat + num_acc]
    ytr_a, yte_a = train["accepted"], test["accepted"]
    acc_model = Pipeline(
        [
            ("prep", _preprocess(cat, num_acc)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    acc_model.fit(Xtr_a, ytr_a)
    proba_a = acc_model.predict_proba(Xte_a)[:, 1]
    results["models"]["acceptance_logit"] = {
        "auc": float(roc_auc_score(yte_a, proba_a)),
        "brier": float(brier_score_loss(yte_a, proba_a)),
    }

    # --- Utilization regressor (among accepted) ---
    tr_acc = train[train.accepted == 1]
    te_acc = test[test.accepted == 1]
    util_model = Pipeline(
        [
            ("prep", _preprocess(cat, num_util)),
            (
                "reg",
                GradientBoostingRegressor(random_state=42, max_depth=3, n_estimators=80),
            ),
        ]
    )
    util_model.fit(tr_acc[cat + num_util], tr_acc["util_after"])
    pred_u = util_model.predict(te_acc[cat + num_util])
    results["models"]["utilization_gbm"] = {
        "mae": float(mean_absolute_error(te_acc["util_after"], pred_u)),
        "r2": float(r2_score(te_acc["util_after"], pred_u)),
        "n_test_accepted": int(len(te_acc)),
    }

    # Permutation importance on PD logit (interpretability)
    # Use a sample for speed
    samp = Xte.sample(n=min(800, len(Xte)), random_state=42)
    y_samp = yte.loc[samp.index]
    r = permutation_importance(logit, samp, y_samp, n_repeats=8, random_state=42, scoring="roc_auc")
    # Map back roughly via transformer feature names is messy; report raw column shuffle importance
    # Recompute by shuffling original columns
    base_auc = roc_auc_score(yte, logit.predict_proba(Xte)[:, 1])
    importances = []
    for col in cat + num_pd:
        X_perm = Xte.copy()
        X_perm[col] = RNG.permutation(X_perm[col].values)
        auc_p = roc_auc_score(yte, logit.predict_proba(X_perm)[:, 1])
        importances.append({"feature": col, "auc_drop": float(base_auc - auc_p)})
    imp_df = pd.DataFrame(importances).sort_values("auc_drop", ascending=False)
    imp_df.to_csv(DATA / "pd_feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(imp_df["feature"], imp_df["auc_drop"], color="#8e44ad")
    ax.invert_yaxis()
    ax.set_xlabel("AUC drop under permutation")
    ax.set_title("PD Logistic — Permutation Importance (OOT)")
    fig.tight_layout()
    fig.savefig(FIG / "pd_permutation_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    results["pd_feature_importance"] = imp_df.to_dict(orient="records")
    results["chosen_pd_model"] = (
        "logistic_pd"
        if m_logit["auc"] >= m_gbm["auc"] - 0.02
        else "gbm_pd"
    )  # prefer interpretable if close
    results["split"] = "out_of_time_vintage_70_30"

    with open(DATA / "ml_validation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Calibration plot for chosen PD
    proba = m_logit["proba_test"]
    bins = pd.qcut(proba, q=10, duplicates="drop")
    cal = pd.DataFrame({"p": proba, "y": yte.values, "bin": bins})
    cal_g = cal.groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.scatter(cal_g["pred"], cal_g["obs"], s=cal_g["n"], alpha=0.7, c="#c0392b")
    ax.set_xlabel("Predicted PD")
    ax.set_ylabel("Observed default rate")
    ax.set_title("PD Calibration (OOT deciles)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "pd_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return results, logit, acc_model, util_model


def main():
    print("ML behavioral models: building offers + OOT validation...")
    borrowers = build_synthetic_portfolio(4000)  # always regenerate for consistent schema
    offers = simulate_labeled_offers(borrowers, n_offers_per=1)
    offers.to_csv(DATA / "synthetic_offers.csv", index=False)
    results, *_ = train_and_validate(offers)
    print(json.dumps({k: v for k, v in results.items() if k != "pd_feature_importance"}, indent=2))
    print("ML done.")
    return results


if __name__ == "__main__":
    main()
