"""
agent/state.py

Shared LangGraph state for the mortgage refinancing pipeline.
Every node function reads from and writes back to this TypedDict.
"""

from typing import Annotated, Literal, TypedDict


class RefinanceState(TypedDict):
    # ── User inputs ───────────────────────────────────────────────────────────
    last_finance_date:  str
    property_value:     float
    loan_balance:       float
    current_rate:       float
    monthly_payment:    float
    remaining_years:    int
    credit_score:       int
    loan_type:          Literal["FIXED", "ARM"]

    # ── rate_agent outputs ────────────────────────────────────────────────────
    mortgage_source:        str
    mortgage_as_of:         str
    mortgage_rate_today:    float
    mortgage_trend_recent:  str
    mortgage_hist_1yr:      list[float]
    mortgage_refi_avg:      float
    mortgage_current_avg:   float
    mortgage_refi_delta:    float

    fed_source:             str
    fed_as_of:              str
    fed_rate_today:         float
    fed_trend_recent:       str
    fed_hist_1yr:           list[float]
    fed_refi_avg:           float
    fed_current_avg:        float
    fed_refi_delta:         float

    # ── rate_trend_predictor outputs ──────────────────────────────────────────
    mortgage_predicted_3m_avg:  float
    mortgage_predicted_delta:   float
    fed_predicted_3m_avg:       float
    fed_predicted_delta:        float

    # ── news_agent + news_summary_agent outputs ───────────────────────────────
    articles:       list
    raw_text:       str          # raw Tavily dump
    news_summary:   str          # condensed signal (≤5 sentences)

    # ── cost_breakeven_heuristic outputs ─────────────────────────────────────
    estimated_new_rate: float
    closing_costs:      float
    new_payment:        float
    monthly_savings:    float
    breakeven_months:   float
    heuristic_signal:   Literal["REFINANCE NOW", "WAIT", "DON'T REFINANCE"]

    # ── analysis_agent outputs ────────────────────────────────────────────────
    recommendation:     Literal["REFINANCE NOW", "WAIT", "DON'T REFINANCE"]
    analysis_reasoning: str          # 2–3 sentence rationale passed to critic/report

    # ── Critic / refiner loop ─────────────────────────────────────────────────
    critic_score:      int | None    # 1–10; routes to refiner if < 7
    critic_notes:      str           # specific objections from critic
    critic_iterations: Annotated[int, lambda a, b: b]   # last-write-wins; hard cap at 3

    # ── report_generator output ───────────────────────────────────────────────
    report: str

    # ── validation_agent outputs ──────────────────────────────────────────────
    validation_passed:    bool
    validation_notes:     str        # empty string if all checks pass
    metrics:              dict       # latency, critic scores, validation, heuristic match

    # ── Pipeline timing ───────────────────────────────────────────────────────
    pipeline_start_time:  float      # unix timestamp recorded at rate_agent entry

    # ── Audit trail ───────────────────────────────────────────────────────────
    memory: Annotated[dict, lambda a, b: {**a, **b}]
