"""
Loan Limit Increase Optimization Engine
Senior Data Scientist Take-Home Assessment
Author: Christopher Nguu
Synthetic portfolio; African/fintech-style unsecured lending assumptions.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
DATA = ROOT / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Assumptions (stated explicitly — synthetic, not proprietary lender data)
# ---------------------------------------------------------------------------
ASSUMPTIONS = {
    "market": "East Africa unsecured digital / mobile-money-adjacent lending",
    "currency": "KES (Kenya Shilling equivalent units)",
    "horizon_months": 12,
    "annual_interest_rate": 0.36,  # ~3% flat/mo digital lending range (illustrative)
    "fee_rate_on_draw": 0.02,
    "lgd": 0.65,  # unsecured recovery ~35%
    "cost_of_capital": 0.12,  # annual
    "operating_cost_per_account": 150,  # KES/year admin
    "max_portfolio_el_ratio": 0.090,  # EL/EAD ceiling (synthetic book baseline ~8.4%)
    "max_cvar_proxy_ratio": 0.12,  # stress EL proxy
    "max_share_high_risk": 0.45,  # exposure in PD bands D+E (synthetic book is risk-heavy)
    "budget_increase_kes": 50_000_000,  # total incremental limit budget
    "n_cohorts": None,  # set after build
}


@dataclass
class MacroScenario:
    name: str
    pd_multiplier: float
    lgd_multiplier: float
    demand_multiplier: float  # utilization / acceptance soft effect
    description: str


SCENARIOS = [
    MacroScenario("baseline", 1.00, 1.00, 1.00, "Stable employment & remittances"),
    MacroScenario("mild_recession", 1.35, 1.10, 0.92, "Inflation + mild unemployment rise"),
    MacroScenario("severe", 1.90, 1.25, 0.80, "Sharp FX/commodity shock, liquidity stress"),
]


# Discrete limit-increase tiers (fraction of current limit)
TIERS = [0.0, 0.10, 0.25, 0.50, 1.00]
TIER_LABELS = ["0%", "10%", "25%", "50%", "100%"]


def logistic(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def build_synthetic_portfolio(n_borrowers: int = 4000) -> pd.DataFrame:
    """Synthetic African fintech-style unsecured book."""
    rng = RNG
    products = rng.choice(["personal", "salary_advance"], size=n_borrowers, p=[0.62, 0.38])
    tenure_m = np.clip(rng.lognormal(mean=2.8, sigma=0.55, size=n_borrowers), 1, 60).astype(int)
    tenure_bucket = pd.cut(
        tenure_m, bins=[0, 6, 18, 61], labels=["new", "established", "loyal"]
    ).astype(str)

    # Latent credit quality
    quality = rng.normal(0, 1, n_borrowers)
    quality += np.where(products == "salary_advance", 0.35, 0.0)
    quality += np.where(tenure_m > 18, 0.4, 0.0) - np.where(tenure_m < 6, 0.35, 0.0)

    base_pd = logistic(-1.8 + 0.85 * (-quality) + rng.normal(0, 0.15, n_borrowers))
    base_pd = np.clip(base_pd, 0.015, 0.35)

    pd_band = pd.cut(
        base_pd,
        bins=[0, 0.04, 0.08, 0.12, 0.18, 1.0],
        labels=["A", "B", "C", "D", "E"],
    ).astype(str)

    current_limit = rng.choice(
        [3000, 5000, 8000, 12000, 20000, 35000, 50000],
        size=n_borrowers,
        p=[0.12, 0.22, 0.25, 0.18, 0.12, 0.07, 0.04],
    ).astype(float)
    # Correlate limit with quality
    current_limit *= np.clip(1.0 + 0.25 * quality, 0.55, 1.6)
    current_limit = np.round(current_limit / 500) * 500

    util = np.clip(
        logistic(0.2 + 0.5 * base_pd - 0.15 * quality + rng.normal(0, 0.4, n_borrowers)),
        0.05,
        0.98,
    )
    util_bucket = pd.cut(util, bins=[0, 0.35, 0.70, 1.01], labels=["low", "med", "high"]).astype(
        str
    )

    balance = current_limit * util
    monthly_income = rng.lognormal(mean=10.2, sigma=0.55, size=n_borrowers)  # ~KES
    dti = balance / np.maximum(monthly_income * 12, 1)

    df = pd.DataFrame(
        {
            "borrower_id": [f"B{i:05d}" for i in range(n_borrowers)],
            "product": products,
            "tenure_m": tenure_m,
            "tenure_bucket": tenure_bucket,
            "pd_band": pd_band,
            "base_pd": base_pd,
            "current_limit": current_limit,
            "utilization": util,
            "util_bucket": util_bucket,
            "balance": balance,
            "monthly_income": monthly_income,
            "dti": dti,
            "quality": quality,
        }
    )
    return df


def cohort_key(row) -> str:
    # Use [] access: row.product collides with Series.prod in pandas
    return f"{row['pd_band']}|{row['util_bucket']}|{row['tenure_bucket']}|{row['product']}"


def aggregate_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cohort"] = df.apply(cohort_key, axis=1)
    g = (
        df.groupby("cohort", as_index=False)
        .agg(
            n=("borrower_id", "count"),
            pd_band=("pd_band", "first"),
            util_bucket=("util_bucket", "first"),
            tenure_bucket=("tenure_bucket", "first"),
            product=("product", "first"),
            base_pd=("base_pd", "mean"),
            current_limit=("current_limit", "mean"),
            utilization=("utilization", "mean"),
            balance=("balance", "sum"),
            total_limit=("current_limit", "sum"),
            mean_income=("monthly_income", "mean"),
        )
    )
    return g


# ---------------------------------------------------------------------------
# Behavioral response functions
# ---------------------------------------------------------------------------
def acceptance_prob(delta_frac: float, util: float, pd: float, demand_mult: float = 1.0) -> float:
    """P(accept limit increase offer). High util & lower PD → higher accept."""
    if delta_frac <= 0:
        return 0.0
    x = -0.4 + 4.5 * delta_frac + 1.8 * util - 2.0 * pd
    return float(np.clip(logistic(x) * demand_mult, 0.0, 0.98))


def utilization_response(
    util0: float, delta_frac: float, accepted: bool, demand_mult: float = 1.0
) -> float:
    """New steady-state utilization after offer outcome."""
    if not accepted or delta_frac <= 0:
        # under-limit churn pressure slightly raises util of remaining balance use
        return float(np.clip(util0 * (1.0 + 0.02 * (delta_frac == 0)), 0.05, 0.99))
    # Marginal propensity to spend new limit (diminishing)
    uplift = 0.35 * delta_frac * (1.0 - util0) * demand_mult
    return float(np.clip(util0 * (1 - 0.15 * delta_frac) + uplift + util0 * 0.15 * delta_frac, 0.05, 0.99))


def pd_after_increase(pd0: float, delta_frac: float, accepted: bool, new_util: float, util0: float) -> float:
    """PD uplift from higher limit / leverage; small if not accepted."""
    if not accepted or delta_frac <= 0:
        return float(pd0)
    leverage = delta_frac * (0.4 + 0.6 * max(0.0, new_util - util0))
    # Relative PD increase capped
    mult = 1.0 + 0.55 * leverage
    return float(np.clip(pd0 * mult, 0.01, 0.55))


def churn_prob(delta_frac: float, util: float, pd: float) -> float:
    """Annual churn if under-limited (high util, no increase)."""
    if delta_frac > 0:
        # generous offers reduce churn
        return float(np.clip(0.08 - 0.06 * delta_frac + 0.05 * pd, 0.02, 0.25))
    # refused / zero increase: high-util borrowers shop around
    return float(np.clip(0.12 + 0.22 * util + 0.08 * pd, 0.05, 0.55))


def expected_profit_cohort(
    n: float,
    limit0: float,
    util0: float,
    pd0: float,
    delta_frac: float,
    scenario: MacroScenario,
    lgd_base: float = ASSUMPTIONS["lgd"],
    r: float = ASSUMPTIONS["annual_interest_rate"],
    fee: float = ASSUMPTIONS["fee_rate_on_draw"],
    coc: float = ASSUMPTIONS["cost_of_capital"],
    opex: float = ASSUMPTIONS["operating_cost_per_account"],
) -> Dict[str, float]:
    """Expected 12-month economics for one cohort under a single tier decision."""
    p_acc = acceptance_prob(delta_frac, util0, pd0, scenario.demand_multiplier)
    # Mixture: accept vs decline (decline → treat as 0% increase path)
    outcomes = []
    for accepted, w in [(True, p_acc), (False, 1.0 - p_acc)]:
        d_eff = delta_frac if accepted else 0.0
        util_n = utilization_response(util0, d_eff, accepted, scenario.demand_multiplier)
        pd_n = pd_after_increase(pd0, d_eff, accepted, util_n, util0) * scenario.pd_multiplier
        pd_n = float(np.clip(pd_n, 0.01, 0.6))
        lgd = float(np.clip(lgd_base * scenario.lgd_multiplier, 0.3, 0.95))
        new_limit = limit0 * (1.0 + d_eff)
        ead = new_limit * util_n  # expected exposure at default proxy
        interest = ead * r
        fees = ead * fee * 0.5  # fees on incremental draws (half of book turns)
        el = ead * pd_n * lgd
        capital_charge = new_limit * coc * 0.5  # average drawn capital approx
        churn = churn_prob(d_eff, util0, pd0)
        # Lost franchise value if churn (simple): 0.25 * interest of survivors foregone
        churn_cost = churn * 0.25 * interest
        profit = interest + fees - el - capital_charge - opex - churn_cost
        outcomes.append(
            {
                "w": w,
                "profit": profit,
                "el": el,
                "ead": ead,
                "pd": pd_n,
                "limit": new_limit,
                "churn": churn,
                "util": util_n,
                "accepted": float(accepted),
            }
        )

    def mix(key):
        return sum(o["w"] * o[key] for o in outcomes)

    inc_limit = limit0 * delta_frac * p_acc  # expected incremental limit
    return {
        "exp_profit_per_acct": mix("profit"),
        "exp_el_per_acct": mix("el"),
        "exp_ead_per_acct": mix("ead"),
        "exp_pd": mix("pd"),
        "exp_limit": mix("limit"),
        "exp_churn": mix("churn"),
        "exp_util": mix("util"),
        "p_accept": p_acc,
        "inc_limit_per_acct": inc_limit,
        "exp_profit_cohort": mix("profit") * n,
        "exp_el_cohort": mix("el") * n,
        "exp_ead_cohort": mix("ead") * n,
        "inc_limit_cohort": inc_limit * n,
    }


# ---------------------------------------------------------------------------
# MILP: portfolio allocation of increase tiers across cohorts
# ---------------------------------------------------------------------------
def solve_milp(
    cohorts: pd.DataFrame,
    scenario: MacroScenario,
    budget: float | None = None,
    max_el_ratio: float | None = None,
    max_high_risk_share: float | None = None,
    force_tier: Dict[str, float] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    budget = budget if budget is not None else ASSUMPTIONS["budget_increase_kes"]
    max_el_ratio = max_el_ratio if max_el_ratio is not None else ASSUMPTIONS["max_portfolio_el_ratio"]
    max_high_risk_share = (
        max_high_risk_share if max_high_risk_share is not None else ASSUMPTIONS["max_share_high_risk"]
    )

    # Precompute economics
    econ = {}
    for _, row in cohorts.iterrows():
        for t in TIERS:
            econ[(row.cohort, t)] = expected_profit_cohort(
                row.n, row.current_limit, row.utilization, row.base_pd, t, scenario
            )

    prob = pulp.LpProblem("loan_limit_increase", pulp.LpMaximize)
    x = pulp.LpVariable.dicts(
        "tier",
        ((c, t) for c in cohorts.cohort for t in TIERS),
        cat="Binary",
    )

    # One tier per cohort
    for c in cohorts.cohort:
        if force_tier and c in force_tier:
            for t in TIERS:
                if abs(t - force_tier[c]) < 1e-9:
                    prob += x[(c, t)] == 1
                else:
                    prob += x[(c, t)] == 0
        else:
            prob += pulp.lpSum(x[(c, t)] for t in TIERS) == 1

    # Objective: total expected profit
    prob += pulp.lpSum(
        econ[(c, t)]["exp_profit_cohort"] * x[(c, t)] for c in cohorts.cohort for t in TIERS
    )

    # Budget on expected incremental limits
    prob += (
        pulp.lpSum(econ[(c, t)]["inc_limit_cohort"] * x[(c, t)] for c in cohorts.cohort for t in TIERS)
        <= budget,
        "budget",
    )

    # Portfolio EL / EAD
    total_el = pulp.lpSum(
        econ[(c, t)]["exp_el_cohort"] * x[(c, t)] for c in cohorts.cohort for t in TIERS
    )
    total_ead = pulp.lpSum(
        econ[(c, t)]["exp_ead_cohort"] * x[(c, t)] for c in cohorts.cohort for t in TIERS
    )
    # EL <= max_el_ratio * EAD  → EL - ratio*EAD <= 0
    prob += total_el - max_el_ratio * total_ead <= 0, "el_ratio"

    # Concentration: high-risk (D,E) exposure share
    high = cohorts[cohorts.pd_band.isin(["D", "E"])].cohort.tolist()
    if high:
        high_ead = pulp.lpSum(
            econ[(c, t)]["exp_ead_cohort"] * x[(c, t)] for c in high for t in TIERS
        )
        prob += high_ead - max_high_risk_share * total_ead <= 0, "high_risk_conc"

    # CVaR proxy: under severe PD mult on same allocation, EL ratio soft constraint
    # Implemented as: baseline solution's stress EL <= max_cvar * stress EAD using severe mult on PD in econ
    # For tractability we apply a scaled EL penalty constraint using mild stress on selected tiers
    stress = MacroScenario("stress_proxy", 1.5, 1.1, 0.9, "inline stress")
    econ_s = {}
    for _, row in cohorts.iterrows():
        for t in TIERS:
            econ_s[(row.cohort, t)] = expected_profit_cohort(
                row.n, row.current_limit, row.utilization, row.base_pd, t, stress
            )
    total_el_s = pulp.lpSum(
        econ_s[(c, t)]["exp_el_cohort"] * x[(c, t)] for c in cohorts.cohort for t in TIERS
    )
    total_ead_s = pulp.lpSum(
        econ_s[(c, t)]["exp_ead_cohort"] * x[(c, t)] for c in cohorts.cohort for t in TIERS
    )
    prob += total_el_s - ASSUMPTIONS["max_cvar_proxy_ratio"] * total_ead_s <= 0, "cvar_proxy"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))
    status_name = pulp.LpStatus[status]
    if status_name != "Optimal":
        # Relax concentration then CVaR proxy; keep EL & budget (common in stressed books)
        if "high_risk_conc" in prob.constraints:
            del prob.constraints["high_risk_conc"]
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))
        status_name = pulp.LpStatus[status] + "_relaxed_conc"
        if not status_name.startswith("Optimal"):
            if "cvar_proxy" in prob.constraints:
                del prob.constraints["cvar_proxy"]
            status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))
            status_name = pulp.LpStatus[status] + "_relaxed_risk"

    rows = []
    for _, row in cohorts.iterrows():
        chosen = None
        for t in TIERS:
            if pulp.value(x[(row.cohort, t)]) is not None and pulp.value(x[(row.cohort, t)]) > 0.5:
                chosen = t
                break
        if chosen is None:
            chosen = 0.0
        e = econ[(row.cohort, chosen)]
        rows.append(
            {
                "cohort": row.cohort,
                "pd_band": row.pd_band,
                "util_bucket": row.util_bucket,
                "tenure_bucket": row.tenure_bucket,
                "product": row.product,
                "n": row.n,
                "tier": chosen,
                "tier_label": f"{int(chosen*100)}%",
                **{f"m_{k}": v for k, v in e.items()},
            }
        )
    alloc = pd.DataFrame(rows)
    metrics = summarize_allocation(alloc, status=status_name, scenario=scenario.name, budget=budget)
    return alloc, metrics


def summarize_allocation(
    alloc: pd.DataFrame, status: str = "Heuristic", scenario: str = "baseline", budget: float = 0.0
) -> Dict[str, float]:
    profit = alloc["m_exp_profit_cohort"].sum()
    el = alloc["m_exp_el_cohort"].sum()
    ead = alloc["m_exp_ead_cohort"].sum()
    inc = alloc["m_inc_limit_cohort"].sum()
    n = alloc["n"].sum()
    # Portfolio default rate proxy: exposure-weighted PD
    pd_w = (alloc["m_exp_pd"] * alloc["m_exp_ead_cohort"]).sum() / max(ead, 1)
    churn_w = (alloc["m_exp_churn"] * alloc["n"]).sum() / max(n, 1)
    high_ead = alloc.loc[alloc.pd_band.isin(["D", "E"]), "m_exp_ead_cohort"].sum()
    return {
        "status": status,
        "scenario": scenario,
        "n_accounts": float(n),
        "total_profit": float(profit),
        "total_el": float(el),
        "total_ead": float(ead),
        "el_ratio": float(el / max(ead, 1)),
        "default_rate_proxy": float(pd_w),
        "incremental_limit": float(inc),
        "budget_used_pct": float(100 * inc / max(budget, 1)),
        "avg_churn": float(churn_w),
        "high_risk_ead_share": float(high_ead / max(ead, 1)),
        "profit_per_account": float(profit / max(n, 1)),
    }


def strategy_force_map(cohorts: pd.DataFrame, strategy: str) -> Dict[str, float] | None:
    """Return forced tier map for non-optimized strategies; None → free MILP."""
    force = {}
    if strategy == "no_increase":
        for c in cohorts.cohort:
            force[c] = 0.0
        return force
    if strategy == "uniform_25":
        for c in cohorts.cohort:
            force[c] = 0.25
        return force
    if strategy == "risk_based":
        # A/B generous, C moderate, D small, E none — util modulates
        for _, row in cohorts.iterrows():
            base = {"A": 0.50, "B": 0.25, "C": 0.10, "D": 0.10, "E": 0.0}[row.pd_band]
            if row.util_bucket == "high" and base > 0:
                # bump one tier if possible
                idx = TIERS.index(base)
                base = TIERS[min(idx + 1, len(TIERS) - 1)]
            if row.util_bucket == "low" and base >= 0.25:
                idx = TIERS.index(base)
                base = TIERS[max(idx - 1, 0)]
            force[row.cohort] = base
        return force
    if strategy == "greedy_high_util":
        for _, row in cohorts.iterrows():
            if row.util_bucket == "high" and row.pd_band in ["A", "B", "C"]:
                force[row.cohort] = 0.50
            elif row.util_bucket == "med" and row.pd_band in ["A", "B"]:
                force[row.cohort] = 0.25
            else:
                force[row.cohort] = 0.0
        return force
    if strategy == "bandit_heuristic":
        # Contextual bandit-style: score = expected profit uplift vs 0%, pick best tier greedily
        # then lightly explore by boosting high-uncertainty (new tenure) one tier
        for _, row in cohorts.iterrows():
            best_t, best_score = 0.0, -1e18
            base = expected_profit_cohort(
                row.n, row.current_limit, row.utilization, row.base_pd, 0.0, SCENARIOS[0]
            )["exp_profit_per_acct"]
            for t in TIERS:
                e = expected_profit_cohort(
                    row.n, row.current_limit, row.utilization, row.base_pd, t, SCENARIOS[0]
                )
                # UCB-like bonus for new tenure
                bonus = 80.0 if row.tenure_bucket == "new" else 0.0
                score = e["exp_profit_per_acct"] - base + bonus * t
                if score > best_score:
                    best_score, best_t = score, t
            force[row.cohort] = best_t
        return force
    return None  # optimized


def run_strategy(
    cohorts: pd.DataFrame, strategy: str, scenario: MacroScenario, budget: float | None = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    budget = budget if budget is not None else ASSUMPTIONS["budget_increase_kes"]
    force = strategy_force_map(cohorts, strategy)
    if strategy == "optimized_milp":
        return solve_milp(cohorts, scenario, budget=budget)
    # For forced strategies, still use MILP shell to evaluate under constraints when possible;
    # if infeasible, evaluate unconstrained economics of forced map.
    try:
        alloc, metrics = solve_milp(cohorts, scenario, budget=budget * 5, force_tier=force)
        # Recompute with true budget note
        metrics = summarize_allocation(alloc, status="Forced", scenario=scenario.name, budget=budget)
        metrics["strategy"] = strategy
        return alloc, metrics
    except Exception:
        pass
    # Manual evaluation
    rows = []
    for _, row in cohorts.iterrows():
        t = force[row.cohort] if force else 0.0
        e = expected_profit_cohort(
            row.n, row.current_limit, row.utilization, row.base_pd, t, scenario
        )
        rows.append(
            {
                "cohort": row.cohort,
                "pd_band": row.pd_band,
                "util_bucket": row.util_bucket,
                "tenure_bucket": row.tenure_bucket,
                "product": row.product,
                "n": row.n,
                "tier": t,
                "tier_label": f"{int(t*100)}%",
                **{f"m_{k}": v for k, v in e.items()},
            }
        )
    alloc = pd.DataFrame(rows)
    metrics = summarize_allocation(alloc, status="Evaluated", scenario=scenario.name, budget=budget)
    metrics["strategy"] = strategy
    return alloc, metrics


STRATEGIES = [
    "no_increase",
    "uniform_25",
    "risk_based",
    "greedy_high_util",
    "bandit_heuristic",
    "optimized_milp",
]


def run_all_simulations(cohorts: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    records = []
    detail = {}
    for scen in SCENARIOS:
        for strat in STRATEGIES:
            alloc, metrics = run_strategy(cohorts, strat, scen)
            metrics["strategy"] = strat
            records.append(metrics)
            detail[(scen.name, strat)] = alloc
    summary = pd.DataFrame(records)
    # Profit lift vs no_increase within scenario
    lifts = []
    for scen in summary.scenario.unique():
        base = summary[(summary.scenario == scen) & (summary.strategy == "no_increase")][
            "total_profit"
        ].iloc[0]
        base_dr = summary[(summary.scenario == scen) & (summary.strategy == "no_increase")][
            "default_rate_proxy"
        ].iloc[0]
        for _, r in summary[summary.scenario == scen].iterrows():
            lifts.append(
                {
                    "scenario": r.scenario,
                    "strategy": r.strategy,
                    "profit_lift_pct": 100.0 * (r.total_profit - base) / abs(base) if base != 0 else 0.0,
                    "default_rate_delta_pp": 100.0 * (r.default_rate_proxy - base_dr),
                }
            )
    lift_df = pd.DataFrame(lifts)
    summary = summary.merge(lift_df, on=["scenario", "strategy"])
    return summary, detail


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_strategy_comparison(summary: pd.DataFrame) -> Path:
    base = summary[summary.scenario == "baseline"].copy()
    order = STRATEGIES
    base["strategy"] = pd.Categorical(base["strategy"], order, ordered=True)
    base = base.sort_values("strategy")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(base)))
    axes[0].barh(base["strategy"].astype(str), base["total_profit"] / 1e6, color=colors)
    axes[0].set_xlabel("Expected profit (KES millions, 12m)")
    axes[0].set_title("Baseline: Profit by Strategy")
    axes[0].invert_yaxis()

    axes[1].barh(base["strategy"].astype(str), 100 * base["default_rate_proxy"], color=colors)
    axes[1].set_xlabel("Exposure-weighted PD proxy (%)")
    axes[1].set_title("Baseline: Default Rate Proxy")
    axes[1].invert_yaxis()
    fig.tight_layout()
    path = FIG / "strategy_comparison_baseline.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_macro_stress(summary: pd.DataFrame) -> Path:
    pivot = summary.pivot(index="strategy", columns="scenario", values="total_profit")
    pivot = pivot.reindex(STRATEGIES)
    pivot = pivot[["baseline", "mild_recession", "severe"]] / 1e6
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, color=["#2ecc71", "#f39c12", "#e74c3c"])
    ax.set_ylabel("Expected profit (KES millions)")
    ax.set_title("Profit under Macro Scenarios")
    ax.legend(title="Scenario")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    path = FIG / "macro_stress_profit.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_efficient_frontier(cohorts: pd.DataFrame) -> Path:
    """Vary EL constraint; record profit for optimized MILP under baseline.

    Unique (realized EL, profit) points are plotted once and connected in
    increasing-EL order. All EL-ceiling detail lives in a right-hand legend,
    which avoids overlapping on-plot annotations.
    """
    el_targets = [0.025, 0.035, 0.045, 0.055, 0.065, 0.08]
    profits, els, statuses = [], [], []
    for el_t in el_targets:
        alloc, m = solve_milp(cohorts, SCENARIOS[0], max_el_ratio=el_t)
        profits.append(m["total_profit"] / 1e6)
        els.append(100 * m["el_ratio"])
        statuses.append(m["status"])

    groups: Dict[Tuple[float, float], List[int]] = {}
    for i, (x, y) in enumerate(zip(els, profits)):
        key = (round(float(x), 4), round(float(y), 4))
        groups.setdefault(key, []).append(i)

    # Sort unique points by realized EL so the drawn frontier does not zigzag
    unique_keys = sorted(groups.keys(), key=lambda k: (k[0], -k[1]))
    xs = [k[0] for k in unique_keys]
    ys = [k[1] for k in unique_keys]

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(xs, ys, "-", color="#5d6d7e", lw=1.8, alpha=0.9, zorder=2)

    colors = ["#1abc9c", "#3498db", "#9b59b6", "#e67e22", "#c0392b"]
    handles = []
    legend_labels = []
    for gi, key in enumerate(unique_keys):
        idxs = groups[key]
        x, y = key
        c = colors[gi % len(colors)]
        h = ax.scatter([x], [y], s=75, color=c, zorder=4, edgecolors="white", linewidths=0.8)
        handles.append(h)
        bits = []
        for i in idxs:
            st = "Optimal" if "Optimal" in str(statuses[i]) else "relaxed"
            bits.append(f"EL≤{el_targets[i]:.1%} ({st})")
        legend_labels.append(", ".join(bits) + f"\nπ={y:.2f}m · EL/EAD={x:.2f}%")

    ax.legend(
        handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7.5,
        frameon=True,
        title="EL ceiling → outcome",
        title_fontsize=8.5,
        borderpad=0.6,
        labelspacing=0.9,
    )

    ax.set_xlabel("Realized EL / EAD (%)")
    ax.set_ylabel("Expected profit (KES millions)")
    ax.set_title("Risk–Return Frontier (Optimized MILP, Baseline)")
    ax.grid(True, alpha=0.3)
    xpad = (max(xs) - min(xs)) * 0.08 or 0.15
    ypad = (max(ys) - min(ys)) * 0.10 or 0.4
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad)
    fig.tight_layout()
    path = FIG / "efficient_frontier.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path



def plot_tier_heatmap(alloc: pd.DataFrame) -> Path:
    """Heatmap of average tier by PD band x util."""
    tmp = alloc.copy()
    tmp["w_tier"] = tmp["tier"] * tmp["n"]
    piv = tmp.pivot_table(
        index="pd_band",
        columns="util_bucket",
        values="w_tier",
        aggfunc="sum",
    )
    cnt = tmp.pivot_table(index="pd_band", columns="util_bucket", values="n", aggfunc="sum")
    avg = (piv / cnt).reindex(index=["A", "B", "C", "D", "E"], columns=["low", "med", "high"])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(avg.values, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["low", "med", "high"])
    ax.set_yticks(range(5))
    ax.set_yticklabels(["A", "B", "C", "D", "E"])
    ax.set_xlabel("Utilization bucket")
    ax.set_ylabel("PD band")
    ax.set_title("Optimized MILP: Mean Limit Increase Fraction")
    for i in range(avg.shape[0]):
        for j in range(avg.shape[1]):
            v = avg.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Increase fraction")
    fig.tight_layout()
    path = FIG / "tier_heatmap_optimized.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_behavior_curves() -> Path:
    deltas = np.linspace(0, 1, 21)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for util, style in [(0.3, "-"), (0.6, "--"), (0.9, ":")]:
        axes[0].plot(deltas, [acceptance_prob(d, util, 0.08) for d in deltas], style, label=f"util={util}")
        axes[1].plot(
            deltas,
            [utilization_response(util, d, True) for d in deltas],
            style,
            label=f"util0={util}",
        )
        axes[2].plot(
            deltas,
            [
                pd_after_increase(0.08, d, True, utilization_response(util, d, True), util)
                for d in deltas
            ],
            style,
            label=f"util0={util}",
        )
    axes[0].set_title("Offer Acceptance")
    axes[1].set_title("Utilization Response")
    axes[2].set_title("PD after Increase (base PD=8%)")
    for ax in axes:
        ax.set_xlabel("Limit increase fraction")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Probability / rate")
    fig.tight_layout()
    path = FIG / "behavior_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    print("Building synthetic portfolio...")
    borrowers = build_synthetic_portfolio(4000)
    borrowers.to_csv(DATA / "synthetic_borrowers.csv", index=False)
    cohorts = aggregate_cohorts(borrowers)
    cohorts.to_csv(DATA / "cohorts.csv", index=False)
    ASSUMPTIONS["n_cohorts"] = int(len(cohorts))
    print(f"Borrowers={len(borrowers)}, Cohorts={len(cohorts)}")

    print("Running strategy simulations across macro scenarios...")
    summary, detail = run_all_simulations(cohorts)
    summary.to_csv(DATA / "strategy_summary.csv", index=False)

    # Save optimized baseline allocation
    opt_alloc = detail[("baseline", "optimized_milp")]
    opt_alloc.to_csv(DATA / "optimized_allocation_baseline.csv", index=False)

    print("Generating figures...")
    paths = [
        plot_strategy_comparison(summary),
        plot_macro_stress(summary),
        plot_efficient_frontier(cohorts),
        plot_tier_heatmap(opt_alloc),
        plot_behavior_curves(),
    ]

    # Key metrics JSON
    baseline = summary[summary.scenario == "baseline"].set_index("strategy")
    key = {
        "assumptions": ASSUMPTIONS,
        "n_borrowers": int(len(borrowers)),
        "n_cohorts": int(len(cohorts)),
        "baseline_metrics": baseline[
            [
                "total_profit",
                "profit_lift_pct",
                "default_rate_proxy",
                "el_ratio",
                "incremental_limit",
                "avg_churn",
                "high_risk_ead_share",
                "status",
            ]
        ].to_dict(orient="index"),
        "figures": [str(p) for p in paths],
    }
    with open(DATA / "key_metrics.json", "w") as f:
        json.dump(key, f, indent=2, default=str)

    print("\n=== BASELINE RESULTS ===")
    cols = [
        "strategy",
        "total_profit",
        "profit_lift_pct",
        "default_rate_proxy",
        "el_ratio",
        "incremental_limit",
        "avg_churn",
    ]
    print(summary[summary.scenario == "baseline"][cols].to_string(index=False))
    print("\nFigures:", paths)
    print("Done.")
    return summary, detail, key


if __name__ == "__main__":
    main()
