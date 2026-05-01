"""
utils/rate_pull.py

Fetch 30-year mortgage rate and Fed funds rate history from FRED,
then compute trend direction and recent averages.
"""

import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()


class RatePull:
    def __init__(self, last_finance_date: str):
        """
        Args:
            last_finance_date: ISO date string of the borrower's last refi or
                               loan origination (e.g. "2022-06-15").
        """
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            raise EnvironmentError("FRED_API_KEY not found in .env")
        self.fred              = Fred(api_key=api_key)
        self.last_finance_date = last_finance_date

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fetch_series(
        self, series_id: str, len_hist: int, len_recent: int
    ) -> tuple[list[float], str, float, float, float]:
        """
        Fetch a single FRED series from the refi date window to today.

        Args:
            series_id:  FRED series identifier (e.g. "MORTGAGE30US").
            len_hist:   Number of trailing observations to keep as history.
            len_recent: Number of trailing observations that define "current".

        Returns:
            Tuple of (history, latest_date, refi_avg, current_avg, delta):
              history     — len_hist observations, oldest → newest
              latest_date — ISO date of the last observation
              refi_avg    — mean of the ±6-week window around last_finance_date
              current_avg — mean of the trailing len_recent observations
              delta       — current_avg - refi_avg (negative = cheaper now)
        """
        refi_dt    = datetime.strptime(self.last_finance_date, "%Y-%m-%d")
        refi_start = (refi_dt - timedelta(weeks=6)).strftime("%Y-%m-%d")
        refi_end   = (refi_dt + timedelta(weeks=6)).strftime("%Y-%m-%d")

        series      = self.fred.get_series(series_id, observation_start=refi_start).dropna()
        tail        = series.iloc[-len_recent:]
        history     = [round(float(v), 4) for v in series.iloc[-len_hist:]]
        latest_date = str(tail.index[-1].date())
        refi_avg    = round(float(series[refi_start:refi_end].mean()), 4)
        current_avg = round(float(tail.mean()), 4)
        delta       = round(current_avg - refi_avg, 4)
        return history, latest_date, refi_avg, current_avg, delta

    def _compute_trend(self, values: list[float], len_mo: int) -> str:
        """
        Classify a rate series as RISING, FALLING, VOLATILE, or STABLE.

        Splits the trailing 3×len_mo observations into three equal buckets and
        compares consecutive bucket averages. Threshold is 0.10 percentage points.

        Args:
            values: Rate history list, oldest → newest.
            len_mo: Number of observations per bucket (e.g. 4 for weekly data).

        Returns:
            One of: "RISING" | "FALLING" | "VOLATILE" | "STABLE"
        """
        mo1_avg = sum(values[-3 * len_mo:-2 * len_mo]) / len_mo
        mo2_avg = sum(values[-2 * len_mo:-len_mo])      / len_mo
        mo3_avg = sum(values[-len_mo:])                 / len_mo
        delta1  = mo2_avg - mo1_avg
        delta2  = mo3_avg - mo2_avg
        if delta1 > 0 and delta2 > 0:
            return "RISING"
        elif delta1 < 0 and delta2 < 0:
            return "FALLING"
        elif (delta1 < 0 and delta2 > 0) or (delta1 > 0 and delta2 < 0):
            return "VOLATILE"
        else:
            return "STABLE"

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_all(
        self,
        mortgage_series_id: str = "MORTGAGE30US",
        fed_series_id: str = "DFF",
    ) -> dict:
        """
        Fetch mortgage and Fed rate data and return a flat dict with
        prefixed keys to avoid collisions.

        Args:
            mortgage_series_id: FRED series for 30-year fixed mortgage rate.
            fed_series_id:      FRED series for the Fed funds effective rate.

        Returns:
            Dict with 8 mortgage_* keys and 8 fed_* keys.
        """
        m_hist, m_date, m_refi, m_cur, m_delta = self._fetch_series(mortgage_series_id, 52, 12)
        f_hist, f_date, f_refi, f_cur, f_delta = self._fetch_series(fed_series_id, 360, 90)
        return {
            "mortgage_source":       f"FRED {mortgage_series_id}",
            "mortgage_as_of":        m_date,
            "mortgage_rate_today":   m_hist[-1],
            "mortgage_trend_recent": self._compute_trend(m_hist, 4),
            "mortgage_hist_1yr":     m_hist,
            "mortgage_refi_avg":     m_refi,
            "mortgage_current_avg":  m_cur,
            "mortgage_refi_delta":   m_delta,

            "fed_source":            f"FRED {fed_series_id}",
            "fed_as_of":             f_date,
            "fed_rate_today":        f_hist[-1],
            "fed_trend_recent":      self._compute_trend(f_hist, 30),
            "fed_hist_1yr":          f_hist,
            "fed_refi_avg":          f_refi,
            "fed_current_avg":       f_cur,
            "fed_refi_delta":        f_delta,
        }


if __name__ == "__main__":
    client = RatePull(last_finance_date="2022-06-15")
    data   = client.fetch_all(mortgage_series_id="MORTGAGE30US", fed_series_id="DFF")

    print("\n📊 MORTGAGE RATE")
    print(f"  Latest rate  : {data['mortgage_rate_today']}%")
    print(f"  Recent trend : {data['mortgage_trend_recent']}")
    print(f"  Refi avg     : {data['mortgage_refi_avg']}%  →  Now: {data['mortgage_current_avg']}%  (Δ {data['mortgage_refi_delta']:+.4f})")
    print(f"  Source       : {data['mortgage_source']}  (as of {data['mortgage_as_of']})")

    print("\n🏦 FED FUNDS RATE")
    print(f"  Fed rate     : {data['fed_rate_today']}%")
    print(f"  Recent trend : {data['fed_trend_recent']}")
    print(f"  Refi avg     : {data['fed_refi_avg']}%  →  Now: {data['fed_current_avg']}%  (Δ {data['fed_refi_delta']:+.4f})")
    print(f"  Source       : {data['fed_source']}  (as of {data['fed_as_of']})")
