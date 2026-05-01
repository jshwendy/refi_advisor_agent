"""
agent/nodes.py

LangGraph node functions for the mortgage refinancing pipeline.
Each function receives the full RefinanceState and returns a partial dict
of fields to merge back into state.
"""

import json
import time

from statsmodels.tsa.arima.model import ARIMA

from agent.llm import claude_chat, strip_json_fences
from agent.state import RefinanceState
from utils.breakeven_calc import (
    _breakeven_months,
    _monthly_payment,
    itemized_closing_costs,
    llpa_adjustment,
)
from utils.news_pull import NewsPull
from utils.rate_pull import RatePull


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def memory_update(state: RefinanceState, mem_json: dict) -> dict:
    """
    Append new values to the audit-trail memory dict.

    Each key in mem_json is versioned as {"0": v0, "1": v1, ...} so that
    multiple writes across critic iterations are all preserved.

    Args:
        state:    Current pipeline state.
        mem_json: Dict of field_name → new_value to record.

    Returns:
        {"memory": updated_memory_dict}
    """
    curr_mem = state.get("memory", {})
    for mem_item, new_mem in mem_json.items():
        if curr_mem.get(mem_item):
            idx = len(curr_mem.get(mem_item))
            curr_mem[mem_item][f"{idx}"] = new_mem
        else:
            curr_mem[mem_item] = {"0": new_mem}
    return {"memory": curr_mem}


def _format_breakeven_display(state: RefinanceState) -> str:
    """Return a human-readable breakeven string for use in LLM prompts."""
    return (
        f"{state['breakeven_months']} months"
        if state["breakeven_months"]
        else "Never (savings insufficient to recover closing costs in NPV terms)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# NODE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def rate_agent(state: RefinanceState) -> dict:
    """
    Fetch current mortgage and Fed funds rate data from FRED.

    Reads:  state.last_finance_date
    Writes: mortgage_source, mortgage_as_of, mortgage_rate_today,
            mortgage_trend_recent, mortgage_hist_1yr, mortgage_refi_avg,
            mortgage_current_avg, mortgage_refi_delta,
            fed_source, fed_as_of, fed_rate_today, fed_trend_recent,
            fed_hist_1yr, fed_refi_avg, fed_current_avg, fed_refi_delta
    """
    start = time.time()
    try:
        client = RatePull(last_finance_date=state["last_finance_date"])
        return {"pipeline_start_time": start, **client.fetch_all()}
    except:
        raise RuntimeError(
            "We use FRED API to read interest rates, but it is temporarily unavailable. "
            "Please try again in a few minutes."
        )


def rate_trend_predictor(state: RefinanceState) -> dict:
    """
    Fit ARIMA(2,1,2) on 1-year history for mortgage and Fed series and
    forecast 3 months forward (13 weekly / 90 daily steps respectively).

    Reads:  state.mortgage_hist_1yr, state.mortgage_current_avg,
            state.fed_hist_1yr, state.fed_current_avg
    Writes: mortgage_predicted_3m_avg, mortgage_predicted_delta,
            fed_predicted_3m_avg, fed_predicted_delta
    """
    def _arima_forecast(history: list[float], steps: int) -> list[float]:
        model  = ARIMA(history, order=(2, 1, 2))
        result = model.fit()
        return result.forecast(steps=steps).tolist()

    m_forecast = _arima_forecast(state["mortgage_hist_1yr"], steps=13)
    f_forecast = _arima_forecast(state["fed_hist_1yr"], steps=90)

    m_predicted_avg = round(sum(m_forecast) / len(m_forecast), 4)
    f_predicted_avg = round(sum(f_forecast) / len(f_forecast), 4)

    rate_trend_predictor_rslt = {
        "mortgage_predicted_3m_avg": round(m_predicted_avg, 6),
        "mortgage_predicted_delta":  round(m_predicted_avg - state["mortgage_current_avg"], 6),
        "fed_predicted_3m_avg":      round(f_predicted_avg, 6),
        "fed_predicted_delta":       round(f_predicted_avg - state["fed_current_avg"], 6),
    }
    memory_update(state, rate_trend_predictor_rslt)
    return rate_trend_predictor_rslt


def news_agent(state: RefinanceState) -> dict:
    """
    Fetch mortgage market news via the Tavily search API.

    Runs 3 targeted queries, deduplicates by URL, and concatenates
    snippets into a raw text blob for downstream summarization.

    Reads:  nothing from state (pure data fetch)
    Writes: articles, raw_text
    """
    client = NewsPull()
    return client.fetch_mortgage_news()


def news_summary_agent(state: RefinanceState) -> dict:
    """
    Summarize raw news articles into ≤5 sentences via LLM.

    Reads:  state.raw_text
    Writes: news_summary
    """
    prompt = f"""
Summarize the following articles in 5 sentences or less, focusing on mortgage rate trends and refinancing timing.

{state["raw_text"]}

Respond in valid JSON only. No preamble, no markdown, no explanation outside the JSON.
{{
  "news_summary": "<5 sentences or less>"
}}
"""
    response = claude_chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    parsed = json.loads(strip_json_fences(response))

    news_summary_agent_rslt = {"news_summary": parsed["news_summary"]}
    memory_update(state, news_summary_agent_rslt)
    return news_summary_agent_rslt


def cost_breakeven_heuristic(state: RefinanceState) -> dict:
    """
    Compute refinancing economics using closed-form formulas — no LLM.

    Steps:
      1. Estimate new rate: adjust mortgage_rate_today by LLPA credit-score tier.
      2. Calculate itemized closing costs (origination, appraisal, title,
         recording, prepaid interest).
      3. Calculate new_payment via standard amortization formula.
      4. Calculate monthly_savings = current_payment - new_payment.
      5. Calculate breakeven_months (Agarwal, Driscoll & Laibson 2013).
      6. Classify heuristic_signal:
           REFINANCE NOW   → savings > 0 AND breakeven < 36 months
           WAIT            → savings > 0 AND breakeven 36–60 months
           DON'T REFINANCE → savings ≤ 0 OR breakeven > 60 months
                              (WAIT instead if rates are forecast lower)

    Reads:  state.current_rate, loan_balance, monthly_payment, remaining_years,
            credit_score, mortgage_rate_today, mortgage_predicted_3m_avg
    Writes: estimated_new_rate, closing_costs, new_payment,
            monthly_savings, breakeven_months, heuristic_signal
    """
    balance          = state["loan_balance"]
    remaining_months = int(state["remaining_years"] * 12)
    base_rate        = state["mortgage_rate_today"]
    predicted_rate   = state["mortgage_predicted_3m_avg"]

    # LLPA-adjusted new rate — spread from FHFA LLPA Matrix (2024), credit-score-only tiers
    estimated_new_rate = base_rate + llpa_adjustment(state["credit_score"])

    # Itemized closing costs — sources: CFPB (2023), ALTA (2022), CoreLogic (2023)
    closing_items = itemized_closing_costs(balance, estimated_new_rate)
    closing_costs = closing_items["closing_costs"]

    new_payment     = _monthly_payment(estimated_new_rate, balance, remaining_months)
    monthly_savings = state["monthly_payment"] - new_payment

    # Closed-form breakeven: T* = -ln(1 - C·r / ΔP) / ln(1 + r),  r = new_rate / 12
    breakeven = _breakeven_months(monthly_savings, closing_costs, estimated_new_rate)

    rates_dropping = predicted_rate < base_rate

    MIN_MONTHLY_SAVINGS = state["monthly_payment"] * 0.05

    if monthly_savings <= 0:
        signal = "DON'T REFINANCE"
    elif monthly_savings < MIN_MONTHLY_SAVINGS:
        signal = "WAIT" if rates_dropping else "DON'T REFINANCE"
    elif breakeven is None:
        # savings > 0 but PV of savings can never recover closing costs
        signal = "DON'T REFINANCE"
    elif breakeven < 24:
        signal = "REFINANCE NOW"
    elif breakeven <= 60:
        # borderline breakeven — WAIT either way in the 36–60 month band
        signal = "WAIT" if rates_dropping else "WAIT"
    else:
        # breakeven > 60: only worth reconsidering if rates are forecast lower
        signal = "WAIT" if rates_dropping else "DON'T REFINANCE"

    retcost_breakeven_rslt = {
        "estimated_new_rate": round(estimated_new_rate, 6),
        "closing_costs":      closing_costs,
        "new_payment":        round(new_payment, 2),
        "monthly_savings":    round(monthly_savings, 2),
        "breakeven_months":   round(breakeven, 1) if breakeven is not None else None,
        "heuristic_signal":   signal,
    }
    memory_update(state, retcost_breakeven_rslt)
    return retcost_breakeven_rslt


def analysis_agent(state: RefinanceState) -> dict:
    """
    Produce a recommendation and 2–3 sentence rationale via LLM.

    Reads:  full financial context from state (rates, costs, news summary)
    Writes: recommendation, analysis_reasoning
    """
    breakeven_display = _format_breakeven_display(state)

    prompt = f"""
You are a mortgage refinancing advisor. Based on the data below, produce a recommendation and reasoning.

BORROWER NUMBERS:
- Current rate:          {state["current_rate"]}%
- Estimated new rate:    {state["estimated_new_rate"]}%
- Current payment:       {state["monthly_payment"]}
- Estimated new payment: {state["new_payment"]}%
- Monthly savings:       ${state["monthly_savings"]}
- Closing costs:         ${state["closing_costs"]}
- Breakeven:             {breakeven_display}
- Remaining term:        {state["remaining_years"]} years

MARKET SIGNALS:
- Heuristic signal:  {state["heuristic_signal"]}
- Mortgage trend:    {state["mortgage_trend_recent"]}
- Fed trend:         {state["fed_trend_recent"]}
- Predicted rate 3m: {state["mortgage_predicted_3m_avg"]}%

NEWS SUMMARY: {state["news_summary"]}

RULES:
1. Your recommendation must agree with or explicitly override the heuristic signal with clear justification.
2. If recommending WAIT, state exactly what condition must be met before refinancing (e.g. rate drops below X%).
3. If breakeven is long or Never, acknowledge whether the borrower's remaining term makes it achievable.
4. Do not rely on rate predictions alone — they are uncertain. State the downside if the prediction is wrong.
5. WAIT and DON'T REFINANCE are not the same. WAIT means refinance when a specific condition is met. DON'T REFINANCE means the economics do not support it under any near-term scenario.
6. Refinancing is a hassle. Small Monthly savings, frequent refinancing, credit score reduction should be taken into consideration 

Respond in valid JSON only. No preamble, no markdown, no explanation outside the JSON.
{{
  "recommendation": "<REFINANCE NOW | WAIT | DON'T REFINANCE>",
  "analysis_reasoning": "<2-3 sentences that reference the actual numbers, state any condition for WAIT, and acknowledge uncertainty>"
}}
"""
    response = claude_chat([{"role": "user", "content": prompt}])
    parsed   = json.loads(strip_json_fences(response))

    analysis_agent_rslt = {
        "recommendation":     parsed["recommendation"],
        "analysis_reasoning": parsed["analysis_reasoning"],
    }
    memory_update(state, analysis_agent_rslt)
    return analysis_agent_rslt


def critic_agent(state: RefinanceState) -> dict:
    """
    Score the current recommendation 1–10 for logical consistency and
    factual accuracy against the hard numbers.

    Reads:  state.recommendation, analysis_reasoning, and all financial fields
    Writes: critic_score, critic_notes, critic_iterations
    """
    breakeven_display = _format_breakeven_display(state)

    prompt = f"""
You are a critical reviewer of mortgage refinancing recommendations.

RECOMMENDATION: {state["recommendation"]}
REASONING: {state["analysis_reasoning"]}

HARD NUMBERS TO VERIFY AGAINST:
- Heuristic signal:   {state["heuristic_signal"]}
- Breakeven:          {breakeven_display}
- Monthly savings:    ${state["monthly_savings"]}
- Estimated new rate: {state["estimated_new_rate"]}%
- Closing costs:      ${state["closing_costs"]}
- Predicted rate 3m:  {state["mortgage_predicted_3m_avg"]}%

Score the recommendation 1-10 based on logical consistency and factual accuracy. Be harsh.

Respond in valid JSON only. No preamble, no markdown, no explanation outside the JSON.
{{
  "critic_score": <integer 1-10>,
  "critic_notes": "<specific objections, or None if score is 8+>"
}}
"""
    response = claude_chat([{"role": "user", "content": prompt}])
    parsed   = json.loads(strip_json_fences(response))

    critic_agent_rslt = {
        "critic_score":      int(parsed["critic_score"]),
        "critic_notes":      parsed["critic_notes"],
        "critic_iterations": state.get("critic_iterations", 0) + 1,
    }
    return {**critic_agent_rslt, **memory_update(state, critic_agent_rslt)}


def refine_agent(state: RefinanceState) -> dict:
    """
    Revise the recommendation in response to critic feedback.

    Reads:  state.recommendation, analysis_reasoning, critic_score,
            critic_notes, and all financial context fields
    Writes: recommendation, analysis_reasoning
    """
    breakeven_display = _format_breakeven_display(state)

    prompt = f"""
You are a mortgage refinancing advisor revising your analysis based on critic feedback.

ORIGINAL RECOMMENDATION: {state["recommendation"]}
ORIGINAL REASONING: {state["analysis_reasoning"]}

CRITIC SCORE: {state["critic_score"]}/10
CRITIC NOTES: {state["critic_notes"]}

BORROWER NUMBERS:
- Current rate:          {state["current_rate"]}%
- Estimated new rate:    {state["estimated_new_rate"]}%
- Current payment:       {state["monthly_payment"]}
- Estimated new payment: {state["new_payment"]}%
- Monthly savings:       ${state["monthly_savings"]}
- Closing costs:         ${state["closing_costs"]}
- Breakeven:             {breakeven_display}
- Remaining term:        {state["remaining_years"]} years

MARKET SIGNALS:
- Heuristic signal:  {state["heuristic_signal"]}
- Mortgage trend:    {state["mortgage_trend_recent"]}
- Fed trend:         {state["fed_trend_recent"]}
- Predicted rate 3m: {state["mortgage_predicted_3m_avg"]}%

NEWS SUMMARY: {state["news_summary"]}

Address each criticism explicitly. Revise if the critic is right. Defend if they are wrong.

Respond in valid JSON only. No preamble, no markdown, no explanation outside the JSON.
{{
  "recommendation": "<REFINANCE NOW | WAIT | DON'T REFINANCE>",
  "analysis_reasoning": "<2-3 sentences>"
}}
"""
    response = claude_chat([{"role": "user", "content": prompt}])
    parsed   = json.loads(strip_json_fences(response))

    refine_agent_rslt = {
        "recommendation":     parsed["recommendation"],
        "analysis_reasoning": parsed["analysis_reasoning"],
    }
    memory_update(state, refine_agent_rslt)
    return refine_agent_rslt


def report_generator(state: RefinanceState) -> dict:
    """
    Generate a plain-language markdown refinancing report for the homeowner.

    Produces exactly 6 sections: Recommendation, Your Numbers, Market Context,
    Cost Analysis, News Highlights, Next Steps.

    Reads:  full state (rates, costs, analysis, news)
    Writes: report
    """
    breakeven_display = _format_breakeven_display(state)

    prompt = f"""
You are a mortgage refinancing advice report writer. Your audience is a homeowner with little financial knowledge. Use plain language, avoid jargon, and explain what numbers mean in practical terms.

BORROWER NUMBERS:
- Current rate:          {state["current_rate"]}%
- Estimated new rate:    {state["estimated_new_rate"]}%
- Current payment:       {state["monthly_payment"]}
- Estimated new payment: {state["new_payment"]}%
- Monthly savings:       ${state["monthly_savings"]}
- Closing costs:         ${state["closing_costs"]}
- Breakeven:             {breakeven_display}
- Remaining term:        {state["remaining_years"]} years

MARKET SIGNALS:
- Heuristic signal:  {state["heuristic_signal"]}
- Mortgage trend:    {state["mortgage_trend_recent"]}
- Fed trend:         {state["fed_trend_recent"]}
- Predicted rate 3m: {state["mortgage_predicted_3m_avg"]}%

NEWS SUMMARY: {state["news_summary"]}

ANALYSIS:
- Final recommendation:                 {state["recommendation"]}
- Reasoning:                            {state["analysis_reasoning"]}
- Critic score (indicates reliability): {state["critic_score"]}/10

Write a report titled '# Your Mortgage Refinancing Advice Report' with exactly these 6 sections in this order:

## Recommendation
One sentence headline. One sentence plain-language rationale.

## Your Numbers
| Item | Current | New | Difference |
|------|---------|-----|-----------|
| Interest rate |  |  |  |
| Monthly payment |  |  |  |
Explain what the interest difference and monthly payment difference means in everyday terms (e.g. "that's roughly the cost of a grocery run each month").

## Market Context
Explain the current rate environment and Fed outlook in plain terms. Include what the 3-month rate prediction means for the borrower.

## Cost Analysis
Explain closing costs and breakeven in plain terms. Tell the borrower how long they need to stay in the home for refinancing to make financial sense.

## News Highlights
Summarize the news in 2-3 sentences relevant to this borrower's decision.

## Next Steps
3-5 concrete actionable steps based on the recommendation. Be specific (e.g. "Call at least 3 lenders to compare rates" not "shop around").

Respond in valid markdown only. No preamble, no explanation outside the markdown.
"""
    response = claude_chat([{"role": "user", "content": prompt}])
    return {"report": response}


def validation_agent(state: RefinanceState) -> dict:
    """
    Validate the generated report for required sections and numerical sanity.

    If sections are missing, calls the LLM to fix formatting only (no new
    content added). Appends an AI disclaimer to any LLM-repaired report.

    Reads:  state.report, estimated_new_rate, closing_costs,
            breakeven_months, recommendation
    Writes: report, validation_passed, validation_notes
    """
    REQUIRED_SECTIONS = [
        "# Your Mortgage Refinancing Advice Report",
        "## Recommendation",
        "## Your Numbers",
        "## Market Context",
        "## Cost Analysis",
        "## News Highlights",
        "## Next Steps",
    ]

    report = state["report"]

    # ── Rule-based checks ─────────────────────────────────────────────────────
    missing_sections = [s for s in REQUIRED_SECTIONS if s not in report]
    numerical_issues = []

    # FIX — check against percentage scale
    if state["estimated_new_rate"] and state["estimated_new_rate"] > 100:
        numerical_issues.append("unreasonalble estimated rate")
    if state["closing_costs"] and not (1_000 <= state["closing_costs"] <= 30_000):
        numerical_issues.append(f"closing_costs ${state['closing_costs']} out of expected range")
    if state["breakeven_months"] and state["breakeven_months"] <= 0:
        numerical_issues.append("breakeven_months must be positive")
    if state["recommendation"] not in ["REFINANCE NOW", "WAIT", "DON'T REFINANCE"]:
        numerical_issues.append(f"invalid recommendation value: {state['recommendation']}")

    validation_notes = ", ".join(missing_sections + numerical_issues)

    # ── If format issues, fix markdown only — no new content ──────────────────
    if missing_sections:
        prompt = f"""
The following mortgage refinancing report has markdown formatting issues.
Fix the formatting only. Do not change, add, or remove any content or facts.
Ensure these section headers are present and correctly formatted: {", ".join(REQUIRED_SECTIONS)}

REPORT TO FIX:
{report}

Respond with the corrected markdown only. No preamble, no explanation outside the markdown.
"""
        report  = claude_chat([{"role": "user", "content": prompt}])

    report = (
        report
        + "\n## AI Disclaimer\nThis report was generated by an AI system with access to "
        "limited information. It is not prefessional financial advice. Please consult a licensed mortgage "
        "professional before finalizing any refinancing decision."
    )

    passed = len(validation_notes) == 0

    return {
        "report":            report,
        "validation_passed": passed,
        "validation_notes":  validation_notes,
        **memory_update(state, {"validation_history": {
            "passed": passed,
            "notes":  validation_notes,
        }}),
    }
