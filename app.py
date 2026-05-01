"""
app.py

Streamlit web interface for the mortgage refinancing agent.
Collects borrower inputs, validates them, and streams the analysis pipeline.

Run with:
    streamlit run app.py
"""

import os
from datetime import date

import streamlit as st

from agent import build_graph

# Inject Streamlit secrets into os.environ so all downstream code (llm.py,
# rate_pull.py, news_pull.py) finds them via os.getenv without modification.
for _key in ("ANTHROPIC_API_KEY", "FRED_API_KEY", "TAVILY_API_KEY"):
    if _key in st.secrets:
        os.environ.setdefault(_key, st.secrets[_key])


# ══════════════════════════════════════════════════════════════════════════════
# INPUT VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_inputs(
    property_value: float,
    loan_balance: float,
    current_rate: float,
    remaining_years: int,
    monthly_payment: float,
    credit_score: int,
    last_finance_date: date,
    loan_type: str,
) -> list[str]:
    """
    Validate mortgage inputs before running the analysis pipeline.

    Args:
        property_value:    Current market value of the home.
        loan_balance:      Remaining principal owed.
        current_rate:      Annual interest rate on current mortgage (%).
        remaining_years:   Years left on current loan.
        monthly_payment:   Current monthly P&I payment.
        credit_score:      Borrower FICO score.
        last_finance_date: Closing date of current mortgage.
        loan_type:         "FIXED" or "ARM".

    Returns:
        List of error strings; empty if all checks pass.
    """
    errors = []

    # ── Loan balance vs property value ────────────────────────────────────────
    if loan_balance >= property_value:
        errors.append("Loan balance cannot exceed property value. You may be underwater on your mortgage.")

    ltv = loan_balance / property_value
    if ltv > 0.97:
        errors.append(f"Loan-to-value ratio is {ltv:.0%} — most lenders require LTV below 97% to refinance.")

    # ── Payment reasonableness ────────────────────────────────────────────────
    r = (current_rate / 100) / 12
    n = remaining_years * 12
    expected_payment = loan_balance * r / (1 - (1 + r) ** -n)
    tolerance = 0.20   # allow 20% variance for taxes/insurance/rounding
    if monthly_payment < expected_payment * (1 - tolerance):
        errors.append(
            f"Monthly payment of {monthly_payment:,.0f} USD seems low. "
            f"Expected roughly {expected_payment:,.0f} USD based on your rate, balance, and term."
        )
    if monthly_payment > expected_payment * (1 + tolerance) * 2:
        errors.append(
            f"Monthly payment of {monthly_payment:,.0f} USD seems very high. "
            f"Expected roughly {expected_payment:,.0f} USD based on your rate, balance, and term."
        )

    # ── Rate sanity ───────────────────────────────────────────────────────────
    if current_rate < 2.0:
        errors.append("Current rate below 2% is unusually low — please double check.")
    if current_rate > 12.0:
        errors.append("Current rate above 12% is unusually high for a conventional mortgage — please double check.")

    # ── Remaining years vs last finance date ──────────────────────────────────
    years_since_finance   = (date.today() - last_finance_date).days / 365.25
    implied_original_term = remaining_years + years_since_finance
    if implied_original_term > 31:
        errors.append(
            f"Remaining term of {remaining_years} years plus {years_since_finance:.1f} years since "
            f"last financing implies a {implied_original_term:.0f}-year original term. "
            "Most mortgages are 10, 15, 20, or 30 years."
        )
    if years_since_finance < 0.5:
        errors.append(
            "Last finance date is less than 6 months ago. Refinancing this soon is unusual "
            "and may incur prepayment penalties."
        )

    # ── Credit score ──────────────────────────────────────────────────────────
    if credit_score < 620:
        errors.append("Credit score below 620 typically does not qualify for conventional refinancing.")
    if loan_type == "ARM" and credit_score < 640:
        errors.append("ARM refinancing generally requires a credit score of at least 640.")

    # ── Equity floor ──────────────────────────────────────────────────────────
    if property_value < loan_balance + 10_000:
        errors.append("Insufficient equity. Most lenders require at least 3-5% equity to refinance.")

    # ── Remaining term floor ──────────────────────────────────────────────────
    if remaining_years < 2:
        errors.append(
            "Less than 2 years remaining on your loan — refinancing closing costs "
            "would be very difficult to recover."
        )

    return errors


# ══════════════════════════════════════════════════════════════════════════════
# UI LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.title("Should I refinance my mortgage?")
st.subheader("We just need a few details")

# ── Top: two input columns ────────────────────────────────────────────────────
top_left, top_right = st.columns(2)

with top_left:
    property_value    = st.number_input("Your Property Value ($)", 100_000, 10_000_000, 480_000,
                        help="The current market value of your home.")
    last_finance_date = st.date_input("Your Last Finance Date", date(2023, 10, 15),
                        help="The closing date of your current mortgage.")
    current_rate      = st.number_input("Your Current Rate (%)", 0.0, 15.0, 7.75,
                        help="The annual interest rate on your current mortgage.")
    loan_balance      = st.number_input("Your Current Loan Balance ($)", 0, 999_999, 350_000,
                        help="The remaining amount you owe on your mortgage.")
    check_clicked     = st.button("Check Input and Analyze", use_container_width=True)

with top_right:

    remaining_years = st.slider("Your Remaining Years", 1, 50, 28,
                      help="The number of years left on your current loan.")
    monthly_payment = st.number_input("Your Current Monthly Payment ($)", 0, 999_999, 2_554,
                      help="Your current monthly mortgage payment (principal + interest only).")
    credit_score    = st.slider("Your Credit Score", 580, 850, 780, step=10,
                      help="Your FICO credit score.")
    loan_type       = st.selectbox("Your Loan Type", ["FIXED", "ARM"],
                      help="FIXED = rate never changes. ARM = rate can adjust after initial period.")
    analyze_clicked = st.button("Just Analyze!", use_container_width=True)

# Handle button clicks after all inputs are defined
if check_clicked:
    st.session_state.force_analyze = False
    errors = validate_inputs(
        property_value, loan_balance, current_rate,
        remaining_years, monthly_payment, credit_score,
        last_finance_date, loan_type,
    )
    if errors:
        for e in errors:
            st.error(e)
        st.warning(
            "Your inputs may be unreasonable. You can still proceed by clicking "
            "'Just Analyze!'. Results may be unreliable though."
        )
    else:
        st.session_state.force_analyze = True

if analyze_clicked:
    st.session_state.force_analyze = True

# ── Bottom: status + report ───────────────────────────────────────────────────
st.divider()

if st.session_state.get("force_analyze"):
    user_input = {
        "last_finance_date": str(last_finance_date),
        "property_value":    float(property_value),
        "loan_balance":      float(loan_balance),
        "current_rate":      float(current_rate),
        "monthly_payment":   float(monthly_payment),
        "remaining_years":   int(remaining_years),
        "credit_score":      int(credit_score),
        "loan_type":         loan_type,
        "critic_iterations": 0,
    }

    app        = build_graph().compile()
    status_box = st.status("Running analysis...", expanded=True)

    label_map = {
        "rate_agent":               "📈 Fetching mortgage & Fed rate data...",
        "news_agent":               "📰 Pulling market news...",
        "rate_trend_predictor":     "🔮 Forecasting rate trends...",
        "news_summary_agent":       "✍️ Summarizing news...",
        "cost_breakeven_heuristic": "🧮 Calculating breakeven...",
        "analysis_agent":           "🧠 Analyzing your refinance opportunity...",
        "critic_agent":             "🔍 Reviewing analysis...",
        "refine_agent":             "✏️ Refining recommendation...",
        "report_generator":         "📄 Generating report...",
        "validation_agent":         "✅ Validating output...",
    }

    with status_box:
        for chunk in app.stream(user_input, stream_mode="updates"):
            node_name = list(chunk.keys())[0]
            st.write(label_map.get(node_name, f"Running {node_name}..."))
            if node_name == "validation_agent":
                st.session_state.result = chunk["validation_agent"]["report"]

    status_box.update(label="Analysis complete!", state="complete")
    st.session_state.force_analyze = False

if "result" in st.session_state:
    st.markdown(st.session_state.result.replace("$", "\\$"))
