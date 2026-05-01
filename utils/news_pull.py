"""
utils/news_pull.py

Search for current mortgage market news via the Tavily API,
deduplicate results, and assemble a raw text blob for LLM summarization.
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from tavily import TavilyClient as _TavilySDK

load_dotenv()


# Queries are kept narrow so results stay on-topic
SEARCH_QUERIES = [
    "mortgage rates",
    "Federal Reserve interest rate",
    "refinance my mortgage",
]

MAX_RESULTS_PER_QUERY = 3   # 3 queries × 3 = 9 articles total, well within free tier


class NewsPull:
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise EnvironmentError("TAVILY_API_KEY not found in .env")
        self.client = _TavilySDK(api_key=api_key)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _search(self, query: str) -> list[dict]:
        """
        Run a single Tavily search.

        Args:
            query: Search string to send to the Tavily API.

        Returns:
            List of {"title", "url", "content", "score"} dicts.
        """
        response = self.client.search(
            query=query,
            search_depth="basic",    # "basic" uses 1 credit; "advanced" uses 2
            max_results=MAX_RESULTS_PER_QUERY,
            include_answer=False,    # the LLM will summarize, not Tavily
        )
        results = []
        for r in response.get("results", []):
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "content": r.get("content", ""),   # truncated snippet (~200 chars)
                "score":   round(r.get("score", 0.0), 3),
            })
        return results

    def _deduplicate(self, articles: list[dict]) -> list[dict]:
        """
        Remove duplicate URLs, keeping the highest-scoring copy.

        Args:
            articles: Flat list of article dicts from one or more queries.

        Returns:
            Deduplicated list sorted by score descending.
        """
        seen_urls = {}
        for a in articles:
            url = a["url"]
            if url not in seen_urls or a["score"] > seen_urls[url]["score"]:
                seen_urls[url] = a
        return sorted(seen_urls.values(), key=lambda x: x["score"], reverse=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_mortgage_news(self) -> dict:
        """
        Run all SEARCH_QUERIES and return deduplicated results.

        Returns:
            Dict with:
              articles — list of {"title", "url", "content", "score"} dicts
              raw_text — concatenated content string for LLM summarization
        """
        all_articles = []
        month_yyyy   = datetime.now().strftime("%B %Y")
        for query in SEARCH_QUERIES:
            query = f"{query} {month_yyyy}"
            all_articles.extend(self._search(query))

        deduped = self._deduplicate(all_articles)

        raw_lines = [
            f"[{i}] {a['title']}\n{a['content']}\nSource: {a['url']}"
            for i, a in enumerate(deduped, 1)
        ]

        return {
            "articles": deduped,
            "raw_text": "\n\n".join(raw_lines),
        }


if __name__ == "__main__":
    client = NewsPull()
    data   = client.fetch_mortgage_news()

    print(f"  Articles    : {len(data['articles'])} (after dedup)\n")

    for i, a in enumerate(data["articles"], 1):
        print(f"  [{i}] {a['title']}")
        print(f"       Score: {a['score']}  |  {a['url']}")
        print(f"       {a['content'][:120]}...")
        print()

    print("─" * 60)
    print("raw_text preview:")
    print(data["raw_text"][:500])
