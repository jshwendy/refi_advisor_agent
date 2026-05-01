"""
utils/breakeven_calc.py

Closed-form mortgage refinancing breakeven analysis.

Primary reference:
    Agarwal, S., Driscoll, J. C., & Laibson, D. I. (2013).
    "Optimal Mortgage Refinancing: A Closed-Form Solution."
    Journal of Money, Credit and Banking, 45(4), 591–622.
    https://doi.org/10.1111/jmcb.12017

Closing cost estimation:
    Brueggeman, W. B., & Fisher, J. D. (2019).
    Real Estate Finance and Investments (15th ed.), p. 312.
    Midpoint of CFPB Closing Disclosure survey range (2.0–3.0%).

Discount rate assumption:
    Homeowner's opportunity cost of capital is their mortgage rate
    (Agarwal et al. 2013, Assumption A2: no arbitrage on home equity).
"""

import math
from typing import Optional


# ── LLPA credit score adjustment ──────────────────────────────────────────────
# Source: FHFA Loan-Level Price Adjustment (LLPA) Matrix (2024).
# Applied as an additive rate spread over the base conforming rate.
# Full matrix also conditions on LTV — omitted here (known simplification).
# https://www.fanniemae.com/media/9391/display

_LLPA_TIERS: list[tuple[int, float]] = [
    (760, 0.00),   #   0 bps — best-execution tier
    (740, 0.13),   # 12.5 bps
    (720, 0.25),   #  25 bps
    (700, 0.50),   #  50 bps
    (680, 0.75),   #  75 bps
    (660, 1.00),   # 100 bps
    (620, 1.50),   # 150 bps
    (0,   2.00),   # 200 bps — sub-620, near non-conforming boundary
]


def llpa_adjustment(credit_score: int) -> float:
    """
    Return the LLPA rate spread (percentage points) for a given FICO score.

    Args:
        credit_score: Borrower FICO score.

    Returns:
        Additive spread to apply to the base conforming rate.
    """
    for min_score, spread in _LLPA_TIERS:
        if credit_score >= min_score:
            return spread
    return _LLPA_TIERS[-1][1]


# ── Itemized closing cost estimate ────────────────────────────────────────────
# Sources:
#   Origination (1.0%):  CFPB Closing Disclosure Survey (2023), lender median
#   Appraisal ($550):    CoreLogic / CFPB survey median, single-family (2023)
#   Title (0.5%):        HUD Settlement Statement avg; ALTA 2022 market survey
#   Recording ($125):    County recorder median (CFPB data); range $25–$250
#   Prepaid interest:    15-day stub — expected-value midpoint of close date
#                        Formula: balance × (new_rate / 12) × (15 / 30)

def itemized_closing_costs(balance: float, new_rate: float) -> dict:
    """
    Bottom-up closing cost estimate with per-item sourcing.

    Args:
        balance:  Current loan principal.
        new_rate: Estimated new annual interest rate (percentage, not decimal).

    Returns:
        Dict with individual line items and a "closing_costs" total.
    """
    origination = balance * 0.010
    appraisal   = 550.0
    title       = balance * 0.005
    recording   = 125.0
    prepaid_int = balance * (new_rate / 1200) * 0.5

    total = origination + appraisal + title + recording + prepaid_int

    return {
        "closing_origination": round(origination, 2),
        "closing_appraisal":   round(appraisal,   2),
        "closing_title":       round(title,        2),
        "closing_recording":   round(recording,    2),
        "closing_prepaid_int": round(prepaid_int,  2),
        "closing_costs":       round(total,        2),
    }


# ── Core helpers ──────────────────────────────────────────────────────────────

def _monthly_payment(annual_rate: float, principal: float, n_months: int) -> float:
    """
    Standard fixed-rate amortization payment formula.

        P * r / (1 - (1+r)^-n),   r = annual_rate / 1200

    Args:
        annual_rate: Annual interest rate (percentage, not decimal).
        principal:   Loan balance.
        n_months:    Remaining term in months.

    Returns:
        Monthly payment amount. Handles zero-rate edge case (straight-line payoff).
    """
    r = annual_rate / 1200
    if r == 0:
        return principal / n_months
    return principal * r / (1.0 - (1.0 + r) ** -n_months)


def _breakeven_months(
    delta_payment: float, closing_costs: float, new_rate: float
) -> Optional[float]:
    """
    Closed-form breakeven horizon T* (months) where NPV of refinancing = 0.

    Derivation (Agarwal et al. 2013, eq. 3):
        NPV(T) = ΔP * [1 - (1+r)^(-T)] / r  -  C,   r = new_rate / 1200
        Set NPV = 0, solve for T:

            T* = -ln(1 - C*r / ΔP) / ln(1 + r)

    Args:
        delta_payment: Monthly savings (old_payment - new_payment).
        closing_costs: Total upfront refinancing costs.
        new_rate:      Estimated new annual interest rate (percentage, not decimal).

    Returns:
        Breakeven in months, or None when closing costs exceed the present value
        of all future savings (i.e. refinancing is never profitable at any horizon).
    """
    r = new_rate / 1200
    if delta_payment <= 0:
        return None   # new payment is higher; refi never beneficial

    inner = 1.0 - (closing_costs * r / delta_payment)
    if inner <= 0:
        return None   # savings can never recover closing costs

    return -math.log(inner) / math.log(1.0 + r)


# ── Smoke test — exercises pure math helpers only ─────────────────────────────
# (node logic lives in agent.nodes.cost_breakeven_heuristic)

if __name__ == "__main__":
    scenarios = [
        dict(label="A — 760 FICO, 112bps drop",
             credit_score=760, base_rate=6.45, balance=400_000,
             remaining_months=324, old_payment=2_782.63),
        dict(label="B — 700 FICO, LLPA narrows spread",
             credit_score=700, base_rate=6.45, balance=400_000,
             remaining_months=324, old_payment=2_782.63),
        dict(label="C — 650 FICO, LLPA wipes out savings",
             credit_score=650, base_rate=6.45, balance=400_000,
             remaining_months=324, old_payment=2_782.63),
    ]

    for s in scenarios:
        new_rate      = s["base_rate"] + llpa_adjustment(s["credit_score"])
        closing_items = itemized_closing_costs(s["balance"], new_rate)
        new_pmt       = _monthly_payment(new_rate, s["balance"], s["remaining_months"])
        delta         = s["old_payment"] - new_pmt
        t_star        = _breakeven_months(delta, closing_items["closing_costs"], new_rate)

        print(f"\n{'=' * 58}")
        print(f"  {s['label']}")
        print(f"{'=' * 58}")
        print(f"  llpa_spread_bps     {llpa_adjustment(s['credit_score']) * 10_000:.1f}")
        print(f"  estimated_new_rate  {new_rate}")
        print(f"  closing_costs       ${closing_items['closing_costs']:,.2f}")
        print(f"  new_payment         ${new_pmt:,.2f}")
        print(f"  monthly_savings     ${delta:,.2f}")
        print(f"  breakeven_months    {round(t_star, 1) if t_star else 'None (never)'}")
