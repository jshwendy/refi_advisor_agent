"""
agent/graph.py

LangGraph graph wiring, critic routing, file export, and CLI entry point.
"""

import json
import os
from datetime import datetime

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    analysis_agent,
    cost_breakeven_heuristic,
    critic_agent,
    news_agent,
    news_summary_agent,
    rate_agent,
    rate_trend_predictor,
    refine_agent,
    report_generator,
    validation_agent,
)
from agent.state import RefinanceState


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════════════════════════════════════

def _route_critic(state: RefinanceState) -> str:
    """
    Route after critic_agent:
      - score ≥ 7 OR iterations ≥ 3    → report_generator
      - score decreased from last iter  → analysis_agent (restart reasoning)
      - score < 7 AND iterations < 3   → refine_agent
    """
    current_score  = state.get("critic_score", 0)
    iterations     = state.get("critic_iterations", 0)
    critic_history = state.get("memory", {}).get("critic_history", {})

    if len(critic_history) >= 2:
        prev_idx   = f"{len(critic_history) - 2}"
        prev_score = critic_history[prev_idx].get("score", current_score)
        if current_score < prev_score:
            return "analysis_agent"

    if current_score >= 7 or iterations >= 3:
        return "report_generator"

    return "refine_agent"


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_node(state: RefinanceState) -> dict:
    """
    Write final state, critic log, and report to timestamped files in cwd.
    Also computes and returns the run metrics dict.

    Reads:  full state
    Writes: metrics
    """
    import time

    end_time   = time.time()
    critic_log = state.get("memory", {})

    critic_scores = [
        v for v in critic_log.get("critic_score", {}).values()
        if isinstance(v, (int, float))
    ]
    metrics = {
        "latency_s":           round(end_time - state.get("pipeline_start_time", end_time), 1),
        "final_critic_score":  critic_scores[-1] if critic_scores else None,
        "avg_critic_score":    round(sum(critic_scores) / len(critic_scores), 1) if critic_scores else None,
        "critic_iterations":   state.get("critic_iterations", 0),
        "validation_passed":   state.get("validation_passed"),
        "heuristic_signal":    state.get("heuristic_signal"),
        "recommendation":      state.get("recommendation"),
        "heuristic_rec_match": state.get("heuristic_signal") == state.get("recommendation"),
    }

    log_dir   = os.path.join(os.path.dirname(__file__), "..", "log")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    final_state = {k: v for k, v in state.items() if k not in ("memory", "report")}

    with open(os.path.join(log_dir, f"{timestamp}_final_state.json"), "w") as f:
        json.dump(final_state, f, indent=2, default=str)

    with open(os.path.join(log_dir, f"{timestamp}_critic_log.json"), "w") as f:
        json.dump(critic_log, f, indent=2, default=str)

    with open(os.path.join(log_dir, f"{timestamp}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(log_dir, f"{timestamp}_final_report.md"), "w") as f:
        f.write(state.get("report", ""))

    print(
        f"\nExported:\n"
        f"  log/{timestamp}_final_state.json\n"
        f"  log/{timestamp}_critic_log.json\n"
        f"  log/{timestamp}_metrics.json\n"
        f"  log/{timestamp}_final_report.md"
    )

    return {"metrics": metrics}


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH WIRING
# ══════════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """Assemble and return the uncompiled LangGraph refinancing pipeline."""
    graph = StateGraph(RefinanceState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("rate_agent",               rate_agent)
    graph.add_node("news_agent",               news_agent)
    graph.add_node("rate_trend_predictor",     rate_trend_predictor)
    graph.add_node("news_summary_agent",       news_summary_agent)
    graph.add_node("cost_breakeven_heuristic", cost_breakeven_heuristic)
    graph.add_node("analysis_agent",           analysis_agent)
    graph.add_node("critic_agent",             critic_agent)
    graph.add_node("refine_agent",             refine_agent)
    graph.add_node("report_generator",         report_generator)
    graph.add_node("validation_agent",         validation_agent)
    graph.add_node("export_node",              export_node)

    # ── Fan-out: rate and news branches run in parallel ───────────────────────
    graph.add_edge(START, "rate_agent")
    graph.add_edge(START, "news_agent")

    # ── Rate branch ───────────────────────────────────────────────────────────
    graph.add_edge("rate_agent",           "rate_trend_predictor")
    graph.add_edge("rate_trend_predictor", "cost_breakeven_heuristic")

    # ── News branch ───────────────────────────────────────────────────────────
    graph.add_edge("news_agent", "news_summary_agent")

    # ── Fan-in: waits for both branches before analysis ───────────────────────
    graph.add_edge(["news_summary_agent", "cost_breakeven_heuristic"], "analysis_agent")

    # ── Critic / refiner loop ─────────────────────────────────────────────────
    graph.add_edge("analysis_agent", "critic_agent")
    graph.add_conditional_edges(
        "critic_agent",
        _route_critic,
        {"report_generator": "report_generator", "refine_agent": "refine_agent", "analysis_agent": "analysis_agent"},
    )
    graph.add_edge("refine_agent", "critic_agent")

    # ── Report → validation → export ─────────────────────────────────────────
    graph.add_edge("report_generator", "validation_agent")
    graph.add_edge("validation_agent", "export_node")
    graph.add_edge("export_node",      END)

    return graph


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from pprint import pprint

    fake_input = {
        "last_finance_date": "2021-03-15",
        "property_value":    520_000.0,
        "loan_balance":      387_000.0,
        "current_rate":      7.12,
        "monthly_payment":   2_608.00,
        "remaining_years":   27,
        "credit_score":      724,
        "loan_type":         "FIXED",
        "critic_iterations": 0,
    }

    app    = build_graph().compile()
    result = app.invoke(fake_input)
    pprint(result.get("report", "— no report —"))
