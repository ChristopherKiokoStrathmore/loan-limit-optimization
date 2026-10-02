#!/usr/bin/env python3
"""Build Loan_Limit_Optimization_Report.pdf from revised content + figures."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
OUT = ROOT / "Loan_Limit_Optimization_Report.pdf"

PAGE_W, PAGE_H = A4
LEFT = RIGHT = 16 * mm
TOP = BOTTOM = 16 * mm


def styles():
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle(
            "TitleC",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            alignment=TA_CENTER,
            spaceAfter=2 * mm,
            textColor=colors.HexColor("#1a252f"),
        ),
        "subtitle": ParagraphStyle(
            "SubC",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=9.5,
            leading=12,
            alignment=TA_CENTER,
            spaceAfter=3 * mm,
            textColor=colors.HexColor("#34495e"),
        ),
        "meta": ParagraphStyle(
            "MetaC",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            spaceAfter=2 * mm,
            textColor=colors.HexColor("#2c3e50"),
        ),
        "h1": ParagraphStyle(
            "H1C",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            spaceBefore=3.5 * mm,
            spaceAfter=1.5 * mm,
            textColor=colors.HexColor("#1a252f"),
        ),
        "h2": ParagraphStyle(
            "H2C",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12.5,
            spaceBefore=2.5 * mm,
            spaceAfter=1.2 * mm,
            textColor=colors.HexColor("#2c3e50"),
        ),
        "body": ParagraphStyle(
            "BodyC",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            alignment=TA_JUSTIFY,
            spaceAfter=1.6 * mm,
        ),
        "note": ParagraphStyle(
            "NoteC",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=11,
            alignment=TA_LEFT,
            spaceBefore=1 * mm,
            spaceAfter=2.5 * mm,
            textColor=colors.HexColor("#4a5568"),
            leftIndent=2 * mm,
            rightIndent=2 * mm,
            borderPadding=3,
        ),
        "caption": ParagraphStyle(
            "CapC",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            spaceBefore=0.5 * mm,
            spaceAfter=2 * mm,
            textColor=colors.HexColor("#5d6d7e"),
        ),
        "bullet": ParagraphStyle(
            "BulC",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=11.5,
            leftIndent=5 * mm,
            spaceAfter=0.8 * mm,
        ),
        "footer": ParagraphStyle(
            "FootC",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#7f8c8d"),
        ),
        "th": ParagraphStyle(
            "ThC",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
        ),
        "td": ParagraphStyle(
            "TdC",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            alignment=TA_LEFT,
        ),
        "tdr": ParagraphStyle(
            "TdR",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
        ),
    }
    return s


def fig(name: str, width_mm: float = 140, caption: str | None = None):
    path = FIG / name
    if not path.exists():
        return [Paragraph(f"[Missing figure: {name}]", styles()["note"])]
    # Preserve aspect
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    aspect = h / w
    w_pt = width_mm * mm
    h_pt = w_pt * aspect
    # Cap height so pages don't blow up
    max_h = 85 * mm
    if h_pt > max_h:
        h_pt = max_h
        w_pt = h_pt / aspect
    img = Image(str(path), width=w_pt, height=h_pt)
    img.hAlign = "CENTER"
    out = [Spacer(1, 1.5 * mm), img]
    if caption:
        out.append(Paragraph(caption, styles()["caption"]))
    else:
        out.append(Spacer(1, 1.5 * mm))
    return out


def table(data, col_widths=None):
    S = styles()
    styled = []
    for r, row in enumerate(data):
        styled_row = []
        for cell in row:
            style = S["th"] if r == 0 else S["tdr"]
            styled_row.append(Paragraph(str(cell), style))
        styled.append(styled_row)
    t = Table(styled, colWidths=col_widths, hAlign="CENTER")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1a252f")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b0bec5")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
            ]
        )
    )
    return t


def add_page_number(canvas, doc):
    canvas.saveState()
    page = canvas.getPageNumber()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#7f8c8d"))
    canvas.drawCentredString(PAGE_W / 2, 10 * mm, f"{page}")
    canvas.restoreState()


def build():
    S = styles()
    story = []

    story.append(Paragraph(
        "Optimal Loan Limit Increases under Profitability–Risk Trade-offs",
        S["title"],
    ))
    story.append(Paragraph(
        "A Predict-then-Optimize Study on Synthetic East-African Unsecured Lending Data",
        S["subtitle"],
    ))
    story.append(Paragraph("Author: Christopher Nguu", S["meta"]))
    story.append(Paragraph(
        "Companion code: loan_limit_optimization.ipynb · loan_limit_engine.py · ml_behavioral_models.py",
        S["meta"],
    ))
    story.append(Spacer(1, 1.5 * mm))
    story.append(Paragraph(
        "Data note. All borrower-level inputs in this study are synthetic. "
        "No proprietary lender portfolios, rates, or performance figures are used or claimed.",
        S["note"],
    ))

    # Abstract
    story.append(Paragraph("Abstract", S["h1"]))
    story.append(Paragraph(
        "This report develops a transparent predict-then-optimize framework for retail loan-limit "
        "increases. Supervised models estimate PD, offer acceptance, and utilization on an "
        "out-of-time-style holdout; a mixed-integer linear program (MILP) allocates discrete increase "
        "tiers across risk cohorts to maximize expected profit subject to budget, expected-loss (EL), "
        "concentration, and stress-loss proxies. On a synthetic East-African unsecured book (N=4,000), "
        "the constrained MILP lifts 12-month expected profit by about 149% versus a no-increase baseline "
        "while reducing the exposure-weighted PD proxy (12.9% → 11.0%) and EL/EAD (8.4% → 7.2%). "
        "Unconstrained bandit-style heuristics can post higher raw profit but with worse default rates—"
        "showing why portfolio constraints belong in the decision layer. Macro stress compresses "
        "attainable profit and motivates scenario-conditioned policies. Limitations are discussed explicitly.",
        S["body"],
    ))

    story.append(Paragraph("1. Introduction", S["h1"]))
    story.append(Paragraph(
        "Credit limit increases deepen relationships and interest income but expand EAD and may worsen "
        "leverage-driven default risk. In emerging-market digital lending, thin files and macro sensitivity "
        "make naive rules fragile. This project answers five assessment questions with an end-to-end "
        "reproducible pipeline: stated assumptions, OOT-validated ML predictors, cohort MILP, heuristic "
        "benchmarks under macro stress, and operational recommendations.",
        S["body"],
    ))

    story.append(Paragraph("2. Related context", S["h1"]))
    story.append(Paragraph(
        "The problem intersects credit scoring (PD/LGD/EAD), customer response / limit elasticity, "
        "constrained portfolio optimization (EL caps, CVaR/stress proxies), and predict-then-optimize "
        "designs. Sequential methods (MDP/RL) are discussed as extensions requiring offline evaluation "
        "and safety constraints. I privilege interpretability (logistic PD + auditable MILP), which fits "
        "regulated lending practice and MSc business-analytics learning outcomes.",
        S["body"],
    ))

    story.append(Paragraph("3. Data and assumptions", S["h1"]))
    story.append(Paragraph(
        "Because live portfolio data are unavailable for this case study, I use a synthetic East-Africa "
        "digital unsecured analogue with KES-equivalent units: 4,000 borrowers aggregated into 77 cohorts "
        "by PD band × utilization × tenure × product (personal vs salary_advance). Horizon 12 months. "
        "Illustrative economics: r=36% interest, fees=2%, LGD=65%, cost of capital=12%, incremental budget "
        "B=KES 50,000,000, max EL/EAD=9%. Tiers T={0%, 10%, 25%, 50%, 100%}.",
        S["body"],
    ))
    story.extend(fig("portfolio_composition.png", 115, "Figure 1. Synthetic portfolio composition."))

    story.append(Paragraph("4. Methodology", S["h1"]))
    story.append(Paragraph("4.1 Machine learning (predict)", S["h2"]))
    story.append(Paragraph(
        "Simulated historical offers; vintage/tenure-sorted 70/30 out-of-time-style split. "
        "Models: logistic regression vs GBM for PD; logistic regression for acceptance; "
        "GBM utilization among acceptors. Metrics: AUC, Brier, AP; MAE/R². Interpretability via "
        "permutation importance and calibration. Prefer logistic PD when OOT AUC is within ~0.02 of GBM.",
        S["body"],
    ))
    story.append(table(
        [
            ["Model", "Metric", "Value"],
            ["PD logistic", "AUC / Brier", "0.732 / 0.182"],
            ["PD GBM", "AUC / Brier", "0.702 / 0.119"],
            ["Acceptance logit", "AUC", "0.762"],
            ["Utilization GBM", "MAE / R²", "0.025 / 0.917"],
        ],
        col_widths=[50 * mm, 45 * mm, 40 * mm],
    ))
    story.append(Spacer(1, 2 * mm))
    story.extend(fig("pd_roc_oot.png", 105, "Figure 2. PD ROC (out-of-time)."))
    story.extend(fig("pd_calibration.png", 105, "Figure 3. PD calibration."))
    story.extend(fig("pd_permutation_importance.png", 110, "Figure 4. PD permutation importance."))

    story.append(Paragraph("4.2 Behavioral responses &amp; profit", S["h2"]))
    story.append(Paragraph(
        "Acceptance rises with tier size and utilization and falls with PD; utilization response has "
        "diminishing marginal draw; PD uplifts with leverage when accepted; churn rises when high-util "
        "accounts are denied. Account profit (12m) mixes interest, fees, expected loss, capital, opex, "
        "and churn cost over accept/decline paths.",
        S["body"],
    ))
    story.extend(fig("behavior_curves.png", 155, "Figure 5. Acceptance, utilization, and PD response curves."))

    story.append(Paragraph("4.3 MILP formulation", S["h2"]))
    story.append(Paragraph(
        "Decision x<sub>c,t</sub> ∈ {0,1}: assign cohort c to tier t. Maximize Σ Π<sub>c,t</sub> x<sub>c,t</sub> "
        "subject to one tier per cohort, incremental-limit budget B, EL ≤ ρ·EAD, high-risk concentration, "
        "and a stress-EL proxy. Cohort economics are precomputed (lookup linearization) so the MIP remains "
        "tractable and auditable (CBC).",
        S["body"],
    ))

    story.append(Paragraph("4.4 Benchmarks &amp; macros", S["h2"]))
    story.append(Paragraph(
        "Strategies: no_increase, uniform_25, risk_based, greedy_high_util, bandit_heuristic, optimized_milp. "
        "Macros: baseline; mild recession (PD×1.35, LGD×1.10, demand×0.92); severe (PD×1.90, LGD×1.25, demand×0.80).",
        S["body"],
    ))

    story.append(Paragraph("5. Results", S["h1"]))
    story.append(Paragraph("5.1 Baseline strategy comparison", S["h2"]))
    story.append(table(
        [
            ["Strategy", "Profit (KES m)", "Lift %", "PD proxy %", "EL/EAD %", "Churn"],
            ["no_increase", "4.11", "0.0", "12.89", "8.38", "0.254"],
            ["uniform_25", "5.88", "43.1", "13.37", "8.71", "0.107"],
            ["risk_based", "5.73", "39.5", "11.97", "7.78", "0.160"],
            ["greedy_high_util", "5.07", "23.3", "12.26", "7.97", "0.209"],
            ["bandit_heuristic", "11.82", "187.6", "16.07", "10.47", "0.053"],
            ["optimized_milp", "10.25", "149.5", "11.04", "7.19", "0.148"],
        ],
        col_widths=[32 * mm, 28 * mm, 18 * mm, 22 * mm, 22 * mm, 18 * mm],
    ))
    story.append(Spacer(1, 1.5 * mm))
    story.append(Paragraph(
        "Insight. MILP status=Optimal. Profit lift ≈ 149.5% with PD proxy 11.04% (vs 12.89% baseline). "
        "The bandit posts higher raw profit but worse default metrics—an unconstrained local envelope, "
        "not a deployable policy under the stated risk appetite.",
        S["body"],
    ))
    story.extend(fig("strategy_comparison_baseline.png", 145, "Figure 6. Baseline strategy comparison."))
    story.extend(fig("tier_heatmap_optimized.png", 100, "Figure 7. Optimized MILP: mean increase fraction by PD band × utilization."))

    story.append(Paragraph("5.2 Macro stress &amp; frontier", S["h2"]))
    story.append(Paragraph(
        "Under mild and severe macros, expected profit falls for all strategies. Stress-aware constraints "
        "and scenario playbooks are therefore first-class design requirements for emerging-market books. "
        "Varying the EL/EAD ceiling traces a risk–return curve for the MILP.",
        S["body"],
    ))
    story.extend(fig("macro_stress_profit.png", 145, "Figure 8. Profit under macro scenarios."))
    story.extend(fig("efficient_frontier.png", 155, "Figure 9. Risk–return frontier (optimized MILP, baseline)."))

    story.append(Paragraph("6. Discussion", S["h1"]))
    for line in [
        "Q1 Strategy: Predict-then-optimize with discrete tiers and hard risk/budget constraints; concentrate increases on low-PD × high-util × longer-tenure cohorts.",
        "Q2 OR: MILP primary; stochastic programming, CVaR, robust OR, and DP/MDP as extensions.",
        "Q3 Behavior: Acceptance, utilization, PD uplift, and churn jointly determine cohort economics; miscalibration rotates the frontier.",
        "Q4 Macro: PD/LGD/demand multipliers shrink the feasible set; maintain stress-conditioned policies.",
        "Q5 Innovation: Constrained bandits/RL, risk-based pricing co-moved with limits, and behavioral nudges—always measured against the MILP baseline under identical constraints.",
    ]:
        story.append(Paragraph("• " + line, S["bullet"]))

    story.append(Paragraph("7. Limitations", S["h1"]))
    story.append(Paragraph(
        "(1) Synthetic external validity unproven for any live lender portfolio. (2) Parametric elasticities "
        "need A/B offer logs. (3) Certainty-equivalent MILP; clustering/fat tails only via stress proxy. "
        "(4) Fairness not audited. (5) APR held fixed—no joint limit–price optimization. "
        "(6) Single-period; multi-period franchise value incomplete.",
        S["body"],
    ))

    story.append(Paragraph("8. Recommendations to operationalize", S["h1"]))
    for i, line in enumerate([
        "Champion–challenger shadow vs current policy.",
        "Monthly re-optimization with early-warning freezes.",
        "Model-risk documentation (assumptions, drift monitors, overrides).",
        "Stratified A/B within MILP-eligible segments.",
        "Macro playbooks with pre-approved tier caps.",
        "Offline bandit sandbox with action masks; couple limits to risk-based fees when RAROC turns negative.",
        "Python/SQL feature store + reproducible batch solve (CBC/Gurobi).",
    ], 1):
        story.append(Paragraph(f"{i}. {line}", S["bullet"]))

    story.append(Paragraph("9. Conclusion", S["h1"]))
    story.append(Paragraph(
        "A literature-aware, assumption-explicit, validated predict-then-optimize pipeline shows that "
        "constrained MILP allocation of limit-increase tiers can substantially improve expected profitability "
        "while improving—not merely tolerating—portfolio risk metrics relative to naive and unconstrained "
        "heuristics. The companion notebook makes results reproducible for assessment and adaptable once "
        "live offer and performance data are available.",
        S["body"],
    ))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "Companion files: form_answers.md · loan_limit_optimization.ipynb · loan_limit_engine.py · "
        "ml_behavioral_models.py · data/ · figures/",
        S["caption"],
    ))

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=LEFT,
        rightMargin=RIGHT,
        topMargin=TOP,
        bottomMargin=BOTTOM + 4 * mm,
        title="Optimal Loan Limit Increases — Christopher Nguu",
        author="Christopher Nguu",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
