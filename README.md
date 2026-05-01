# Mortgage Refinancing Agent

A Streamlit app powered by a LangGraph multi-agent pipeline that analyzes whether you should refinance your mortgage right now.

## Setup

### 1. Install dependencies

```bash
pip install -r requirement.txt
```

### 2. Configure API keys

Stemming from `.env.example`, create a `.env` file in the project root with the following keys:

```env
FRED_API_KEY=...        # https://fred.stlouisfed.org/docs/api/api_key.html
TAVILY_API_KEY=...      # https://app.tavily.com
ANTHROPIC_API_KEY=...   # https://console.anthropic.com
```

All three keys are required. FRED and Tavily both offer free tiers. Swap to LLM of your choice.

### 3. Run the app

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Folder Structure

```
refi_agent/
├── app.py                  # Streamlit UI and input validation
├── requirement.txt
├── .env                    # Your API keys (not committed)
├── .env.example            # Template for .env
│
├── agent/
│   ├── graph.py            # LangGraph wiring and critic routing
│   ├── nodes.py            # All agent node implementations
│   ├── state.py            # Shared pipeline state definition
│   └── llm.py              # Anthropic API client
│
├── utils/
│   ├── rate_pull.py        # FRED rate fetching
│   ├── news_pull.py        # Tavily news fetching
│   └── breakeven_calc.py   # Refinance cost/breakeven math
│
└── log/                    # Auto-created at runtime
    └── <timestamp>_final_state.json
    └── <timestamp>_critic_log.json
    └── <timestamp>_final_report.md
```

## Usage

Fill in your mortgage details on the left panel:

| Field | Description |
|---|---|
| Property Value | Current market value of your home |
| Last Finance Date | Closing date of your current mortgage |
| Current Rate (%) | Annual interest rate |
| Loan Balance | Remaining principal owed |
| Remaining Years | Years left on the loan |
| Monthly Payment | Principal + interest only |
| Credit Score | Your FICO score |
| Loan Type | Fixed or ARM |

Click **Check Input and Analyze** to validate inputs before running. If validation flags issues you want to ignore, click **Just Analyze!** to proceed anyway. The analysis streams live on the right panel and a final report appears when complete.

## Agent Structure

The pipeline is a LangGraph graph with two parallel branches that fan in before the analysis and critique loop.

```
          ┌─────────────┐   ┌─────────────┐
          │  rate_agent │   │  news_agent │   ← parallel data fetch
          └──────┬──────┘   └──────┬──────┘
                 │                 │
     rate_trend_predictor   news_summary_agent
                 │                 │
     cost_breakeven_heuristic      │
                 └────────┬────────┘
                          │  fan-in
                   analysis_agent
                          │
                    critic_agent  ←──────────┐
                          │                  │
               ┌──────────┴──────────┐       │
           score ≥ 7            score < 7    │
           or 3 iters           & iters < 3  │
               │                    │        │
        report_generator       refine_agent ─┘
               │
        validation_agent
               │
          export_node  → writes JSON + Markdown files to disk
```

**Branches:**
- **Rate branch** — fetches live mortgage and Fed rate data (FRED), forecasts rate trends, and computes a cost/breakeven heuristic.
- **News branch** — pulls current mortgage market news (Tavily) and summarizes it.

**Critique loop** — after the initial analysis, a critic scores it (0–10). If the score is below 7 and fewer than 3 iterations have run, a refine agent rewrites the analysis and the critic re-evaluates. If the score drops between iterations, the pipeline restarts from the analysis agent.

**Output** — a validation agent reviews the final report before it is displayed and also exported to timestamped `.json` and `.md` files in the working directory.
