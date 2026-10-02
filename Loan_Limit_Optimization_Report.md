# Optimal Loan Limit Increases under Profitability–Risk Trade-offs
## A Predict-then-Optimize Study on Synthetic East-African Unsecured Lending Data

**Author:** Christopher Nguu  
**Companion code:** `loan_limit_optimization.ipynb`, `loan_limit_engine.py`, `ml_behavioral_models.py`

---

## Abstract

This report develops a transparent predict-then-optimize framework for retail loan-limit increases. Supervised models estimate probability of default (PD), offer acceptance, and utilization response on an out-of-time-style holdout; a mixed-integer linear program (MILP) then allocates discrete increase tiers across risk cohorts to maximize expected profit subject to budget, expected-loss (EL), concentration, and stress-loss proxies.

All borrower-level inputs are synthetic: no proprietary lender portfolios, rates, or performance figures are used or claimed. On this East-African unsecured book ($N=4000$), the constrained MILP lifts 12-month expected profit by about 149% versus a no-increase baseline while reducing the exposure-weighted PD proxy (12.9% → 11.0%) and EL/EAD (8.4% → 7.2%). Unconstrained bandit-style heuristics can post higher raw profit but with materially worse default rates, which is a useful reminder that portfolio constraints belong in the decision layer. Macro stress scenarios compress attainable profit and motivate scenario-conditioned policies. I discuss limitations of synthetic design, parametric behavior, and certainty-equivalent optimization explicitly.

Keywords: credit limit optimization; predict-then-optimize; MILP; PD modelling; utilization elasticity; stress testing; fintech lending

---

## 1. Introduction

Credit limit increases are a central lever in revolving and digital unsecured products. They can deepen relationships and interest income, but they also expand exposure at default (EAD) and may worsen leverage-driven default risk. In emerging-market digital lending, thin files, volatile incomes, and macroeconomic sensitivity make naive rules—uniform percentage increases, or always raising high utilizers—particularly fragile.

The questions I set out to answer (mapped to the application form) are:

1. What limit-increase strategy balances profitability and risk?
2. Which operations-research (OR) techniques are appropriate?
3. How do acceptance, utilization, repayment/PD uplift, and churn reshape the optimum?
4. How do external economic conditions change the solution?
5. What innovative methods (RL, dynamic pricing, nudges) can extend the baseline?

The contribution of this project is an end-to-end, reproducible pipeline that (i) states assumptions, (ii) validates ML predictors out-of-time, (iii) formulates and solves a cohort MILP, (iv) benchmarks heuristics under macro stress, and (v) discusses limitations and operationalization.

---

## 2. Related context and conceptual background

The problem sits at the intersection of several streams:

- Credit risk modelling — PD, LGD, EAD and expected loss $\mathrm{EL}=\mathrm{EAD}\times\mathrm{PD}\times\mathrm{LGD}$, as in IRB / IFRS 9 style decompositions.
- Customer response / limit elasticity — whether an offer is accepted and how much of the new limit is drawn.
- Portfolio decision science — constrained optimization with risk measures (EL caps, CVaR or stress proxies), and increasingly predict-then-optimize designs where ML supplies parameters to an OR model.
- Sequential decision making — MDPs / reinforcement learning for timing and personalization, typically requiring strong offline evaluation and safety constraints before production use.

I privilege interpretability and constraint fidelity (logistic PD + MILP) over black-box end-to-end policies. That choice fits regulated lending practice and the learning outcomes of an MSc programme in business analytics.

---

## 3. Data and assumptions

### 3.1 Synthetic portfolio design

Because this case study cannot use live portfolio data, I build a synthetic East-African digital / mobile-money-adjacent unsecured book. Currency units are KES-equivalent. The sample has 4,000 borrowers, aggregated into 77 risk cohorts. Products are `personal` (62%) and `salary_advance` (38%). Segmentation cuts PD band (A–E), utilization (low/med/high), tenure (new/established/loyal), and product. The planning horizon is 12 months.

| Item | Choice |
|------|--------|
| Market analogue | East Africa digital / mobile-money-adjacent unsecured lending |
| Currency unit | KES-equivalent |
| Sample size | 4,000 synthetic borrowers → 77 risk cohorts |
| Products | `personal` (62%), `salary_advance` (38%) |
| Segmentation | PD band (A–E), utilization (low/med/high), tenure (new/established/loyal), product |
| Horizon | 12 months |

### 3.2 Economic assumptions (illustrative, not proprietary)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Annual interest rate $r$ | 36% | Illustrative digital-lending order of magnitude |
| Fee rate on draw | 2% | Origination / facility fees (partial turnover) |
| LGD | 65% | Unsecured recovery ~35% |
| Cost of capital | 12% | Funding / economic capital proxy |
| Opex per account | KES 150 / year | Admin |
| Incremental limit budget $B$ | KES 50,000,000 | Portfolio programme budget |
| Max EL/EAD $\rho$ | 9.0% | Feasible relative to synthetic baseline ~8.4% |
| Max high-risk EAD share | 45% | Book is intentionally risk-heterogeneous |
| Stress EL proxy | 12% under inline stress multipliers | CVaR-style stand-in |

Increase tiers: $\mathcal{T}=\{0\%,10\%,25\%,50\%,100\%\}$ of current limit.

---

## 4. Methodology

### 4.1 Machine learning layer (predict)

From synthetic historical offers I train three models. Models: logistic regression vs GBM for PD; logistic regression for acceptance; gradient boosting for utilization among accepted offers.

Validation uses accounts sorted by a vintage/tenure score: first 70% train, last 30% test (out-of-time style). Metrics are AUC, Brier, and average precision for classifiers, plus MAE and $R^2$ for utilization. For interpretability I report permutation importance and reliability (calibration) plots. Preference rule: choose logistic PD when OOT AUC is within about 0.02 of GBM.

### 4.2 Behavioral response layer

For increase fraction $\delta$:

- Acceptance $p^{\mathrm{acc}}=\sigma(\beta_0+\beta_\delta\delta+\beta_u u_0+\beta_{\mathrm{PD}}\mathrm{PD}_0)$ scaled by macro demand.
- Utilization response with diminishing marginal draw on new limit.
- PD uplift from leverage when the offer is accepted.
- Churn higher when high-utilization accounts are denied increases.

Expected profit mixes accept and decline paths.

### 4.3 Optimization layer (optimize)

Decision variables: $x_{c,t}\in\{0,1\}$ — cohort $c$ assigned to tier $t$.

\begin{align}
\max_x &\sum_{c,t}\Pi_{c,t}\,x_{c,t} \\
\text{s.t.} &\sum_t x_{c,t}=1 &&\forall c \\
&\sum_{c,t}\Delta L_{c,t}\,x_{c,t}\le B \\
&\sum_{c,t}\mathrm{EL}_{c,t}\,x_{c,t}\le\rho\sum_{c,t}\mathrm{EAD}_{c,t}\,x_{c,t} \\
&\sum_{c\in\mathcal{H},t}\mathrm{EAD}_{c,t}\,x_{c,t}\le\gamma\sum_{c,t}\mathrm{EAD}_{c,t}\,x_{c,t} \\
&\text{stress-EL}\le\rho^{s}\times\text{stress-EAD}
\end{align}

$\Pi_{c,t}$, $\mathrm{EL}_{c,t}$, and $\Delta L_{c,t}$ are precomputed from the behavioral model (lookup linearization). That keeps the MILP tractable and auditable; I solve it with CBC.

### 4.4 Benchmark strategies

| Strategy | Description |
|----------|-------------|
| `no_increase` | All tiers 0% |
| `uniform_25` | Flat +25% |
| `risk_based` | Ladder by PD band, modulated by utilization |
| `greedy_high_util` | Large increases only for high util × better PD |
| `bandit_heuristic` | Per-cohort best uplift + UCB-style exploration bonus for new tenure |
| `optimized_milp` | Constrained MILP |

### 4.5 Macro scenarios

| Scenario | PD mult. | LGD mult. | Demand mult. |
|----------|----------|-----------|--------------|
| Baseline | 1.00 | 1.00 | 1.00 |
| Mild recession | 1.35 | 1.10 | 0.92 |
| Severe | 1.90 | 1.25 | 0.80 |

---

## 5. Results

### 5.1 ML validation (OOT)

| Model | Metric | Value |
|-------|--------|------:|
| PD logistic | AUC | 0.732 |
| PD logistic | Brier | 0.182 |
| PD GBM | AUC | 0.702 |
| Acceptance logit | AUC | 0.762 |
| Utilization GBM | MAE | 0.025 |
| Utilization GBM | $R^2$ | 0.917 |

Chosen PD model: logistic (better OOT AUC and stronger interpretability). Feature permutation importance highlights utilization, offer tier, leverage (DTI), and tenure-related signals — consistent with domain expectation.

### 5.2 Baseline strategy comparison

| Strategy | Profit (KES m) | Lift vs no-increase | PD proxy (%) | EL/EAD (%) | Avg churn |
|----------|---------------:|--------------------:|-------------:|-----------:|----------:|
| no_increase | 4.11 | 0% | 12.89 | 8.38 | 0.254 |
| uniform_25 | 5.88 | +43.1% | 13.37 | 8.71 | 0.107 |
| risk_based | 5.73 | +39.5% | 11.97 | 7.78 | 0.160 |
| greedy_high_util | 5.07 | +23.3% | 12.26 | 7.97 | 0.209 |
| bandit_heuristic | 11.82 | +187.6% | 16.07 | 10.47 | 0.053 |
| optimized_milp | 10.25 | +149.5% | 11.04 | 7.19 | 0.148 |

The MILP is not the raw-profit maximizer; the bandit is. The MILP is the risk-constrained maximizer: lower PD proxy and EL ratio than both the baseline and the bandit, with a large profit lift. That is the academically and operationally relevant notion of “optimal” under a risk appetite.

MILP solution status: Optimal; budget utilization ≈ 58%; high-risk EAD share ≈ 23%.

### 5.3 Allocation pattern

The optimized heatmap (PD band × utilization) concentrates larger increase fractions on stronger PD bands with meaningful utilization — rationing E/D segments — aligning with the behavioral elasticities in §4.2.

### 5.4 Macro stress

Under mild and severe scenarios, expected profit falls for all strategies. Policies that ignore stress EL look more fragile. This supports either scenario-weighted stochastic programs or indicator-triggered playbooks (tighten tiers for PD C–E when leading indicators deteriorate).

### 5.5 Efficient frontier

Varying the EL/EAD ceiling traces a risk–return curve for the MILP: tighter risk appetite reduces attainable profit, making the trade-off explicit for credit committees.

---

## 6. Discussion

### 6.1 Answering the form questions (synthesis)

1. Strategy: Predict-then-optimize with discrete tiers, cohort MILP, and hard risk/budget constraints; generous to low-PD × high-util × longer-tenure; freeze weak risks.
2. OR toolkit: MILP primary; stochastic programming / CVaR / robust OR / DP as extensions.
3. Behavior: Acceptance, utilization, PD uplift, and churn jointly determine $\Pi_{c,t}$; miscalibration rotates the frontier.
4. Macro: PD/LGD/demand multipliers shift the feasible set; maintain stress-conditioned policies.
5. Innovation: Constrained bandits / RL, risk-based pricing co-moved with limits, and behavioral nudges — always measured against the MILP baseline under the same constraints.

### 6.2 Interpretability and governance

Logistic PD, precomputed cohort tables, and an explicit MIP are committee-friendly. Overrides remain possible without retraining a monolithic policy network.

---

## 7. Limitations

1. Synthetic data — external validity to any live lender portfolio is unproven.
2. Parametric behavior — true elasticities need A/B offer experiments.
3. Certainty equivalence — MILP uses expectations; default clustering and fat tails are only proxied via stress EL.
4. Fairness — no audit across protected attributes or adverse geographic concentration beyond PD bands.
5. Pricing held fixed — joint limit–APR optimization is left to future work.
6. Single-period — multi-period MDP value of relationship not fully capitalized.

---

## 8. Recommendations to operationalize

1. Champion–challenger: shadow the MILP against current policy for 1–2 cycles; compare profit, EL, and complaints/churn.
2. Monthly re-optimization with refreshed PD and utilization; freeze list from early-warning scores.
3. Model risk management: document assumptions, monitoring (AUC/calibration drift), and override logs.
4. Experimentation: stratified A/B on tier size within MILP-eligible segments; estimate acceptance and PD uplift.
5. Macro playbooks: pre-approve mild/severe tier caps; trigger on FX, collections cure, and mobility/income proxies.
6. Innovation sandbox: offline bandit evaluation with action masks; couple limit changes to risk-based fees where RAROC turns negative.
7. Data stack: Python/SQL feature store; reproducible notebooks; solver (CBC/Gurobi) in a controlled batch job.

---

## 9. Conclusion

A literature-aware, assumption-explicit, validated predict-then-optimize pipeline shows that constrained MILP allocation of limit-increase tiers can substantially improve expected profitability while improving—not merely “accepting”—portfolio risk metrics relative to naive and unconstrained heuristics. The accompanying notebook makes the results reproducible for assessment and for adaptation once live offer and performance data are available.

---

## Appendix A — Repository map

| Path | Role |
|------|------|
| `form_answers.md` | Concise web-form responses (Q1–Q5) |
| `Loan_Limit_Optimization_Report.md` | This report |
| `Loan_Limit_Optimization_Report.pdf` | Polished PDF with embedded figures |
| `loan_limit_optimization.ipynb` | Executable notebook |
| `loan_limit_engine.py` | Synthetic data, behavior, MILP, simulation |
| `ml_behavioral_models.py` | OOT ML training & interpretability |
| `requirements.txt` | Python dependencies (`pulp<3`) |
| `data/` | CSV / JSON metrics |
| `figures/` | PNG charts |

## Appendix B — Key numeric results (baseline)

- No-increase profit: KES 4.11m; PD proxy 12.89%; EL/EAD 8.38%
- Optimized MILP profit: KES 10.25m (+149.5%); PD proxy 11.04%; EL/EAD 7.19%
- Bandit (unconstrained local scores): profit KES 11.82m but PD proxy 16.07%
- PD logistic OOT AUC 0.732; acceptance AUC 0.762; utilization $R^2$ 0.917
