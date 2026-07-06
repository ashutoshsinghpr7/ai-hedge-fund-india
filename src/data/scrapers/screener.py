"""Screener.in web scraper for Indian stock fundamentals."""

import logging
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from src.data.scrapers import Scraper

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache/webscraper/screener")
RATE_LIMIT_SECONDS = 1.0


def _strip_suffix(ticker: str) -> str:
    return ticker.replace(".NS", "").replace(".BO", "").upper()


class ScreenerScraper(Scraper):
    """Scrapes fundamental financial data from Screener.in."""

    def __init__(self):
        self._last_request = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < RATE_LIMIT_SECONDS:
            time.sleep(RATE_LIMIT_SECONDS - elapsed)
        self._last_request = time.time()

    def _cached_url(self, url: str, cache_key: str) -> str | None:
        cache_path = CACHE_DIR / f"{cache_key}.html"
        if cache_path.exists():
            age = time.time() - cache_path.stat().st_mtime
            if age < 86400:
                return cache_path.read_text(encoding="utf-8", errors="replace")

        try:
            self._rate_limit()
            resp = httpx.get(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; ai-hedge-fund-india/1.0; +https://github.com/ashutoshsinghpr7/ai-hedge-fund-india)",
                "Accept": "text/html",
            }, timeout=15, follow_redirects=True)

            if resp.status_code != 200:
                logger.warning("Screener HTTP %s for %s", resp.status_code, url)
                return None

            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(resp.text, encoding="utf-8")

            return resp.text
        except Exception as e:
            logger.warning("Screener request failed for %s: %s", url, e)
            return None

    def fetch_financials(self, ticker: str) -> dict | None:
        """Fetch key financial ratios from Screener.in."""
        symbol = _strip_suffix(ticker)
        url = f"https://www.screener.in/company/{symbol}/consolidated/"
        html = self._cached_url(url, symbol.lower())

        if not html:
            return None

        soup = BeautifulSoup(html, "html.parser")

        data: dict = {"source": "screener.in", "ratios": {}}

        name_el = soup.select_one("h1")
        if name_el:
            data["company_name"] = name_el.text.strip()

        price_el = soup.select_one("#top .number .value")
        if price_el:
            data["current_price"] = self._parse_num(price_el.text)

        ratios = data["ratios"]

        for li in soup.select("#top-ratios li"):
            text = li.get_text(" ", strip=True)
            name, val = self._split_ratio(text)
            if name and val is not None:
                ratios[name] = val

        tables = soup.select("table.data-table")
        for table in tables:
            for row in table.select("tr"):
                cells = [td.get_text(" ", strip=True) for td in row.select("td, th")]
                if len(cells) >= 2:
                    header = cells[0].lower()
                    if "sales" in header:
                        ratios["sales"] = self._parse_num(cells[1])
                    elif any(k in header for k in ("roe", "return on equity")):
                        ratios["roe"] = self._parse_percent(cells[1])
                    elif any(k in header for k in ("roce", "return on capital")):
                        ratios["roce"] = self._parse_percent(cells[1])
                    elif "operating profit margin" in header:
                        ratios["opm"] = self._parse_percent(cells[1])
                    elif "net profit margin" in header:
                        ratios["npm"] = self._parse_percent(cells[1])
                    elif "debt to equity" in header:
                        ratios["debt_to_equity"] = self._parse_num(cells[1])
                    elif "eps in rs" in header:
                        ratios["eps"] = self._parse_num(cells[1])
                    elif "book value" in header:
                        ratios["book_value"] = self._parse_num(cells[1])
                    elif "dividend yield" in header:
                        ratios["dividend_yield"] = self._parse_percent(cells[1])
                    elif "cash" in header and "equivalent" in header:
                        ratios["cash"] = self._parse_num(cells[1])

        if not data.get("ratios"):
            return None
        return data

    def fetch_prices(self, ticker: str) -> list[dict] | None:
        """Screener.in doesn't provide historical price data."""
        return None

    def fetch_news(self, ticker: str) -> list[dict] | None:
        """Screener.in doesn't provide news."""
        return None

    def _parse_num(self, text: str) -> float | None:
        if not text:
            return None
        text = text.strip().replace(",", "").replace("₹", "").replace("%", "")
        # Handle abbreviated values like "5,678 Cr"
        multiplier = 1.0
        text_lower = text.lower()
        for suffix, mult in [("cr", 1e7), ("lac", 1e5), ("k", 1e3), ("bn", 1e9)]:
            if suffix in text_lower:
                multiplier = mult
                text_lower = text_lower.replace(suffix, "")
                text = "".join(c for c in text if c.isdigit() or c in ".-")
                break
        try:
            return float(text_lower.strip() or text.strip()) * multiplier
        except (ValueError, TypeError):
            return None

    def _parse_percent(self, text: str) -> float | None:
        val = self._parse_num(text)
        if val is not None:
            return val / 100.0 if val > 1 else val
        return None

    def _split_ratio(self, text: str) -> tuple[str | None, float | None]:
        """Parse a ratio string like 'Market Cap  ₹ 58,514 Cr.'"""
        val = self._parse_num(text)
        if val is None:
            return None, None

        name_parts = []
        for part in text.split():
            cleaned = part.strip().replace(",", "").replace("₹", "").replace("%", "")
            try:
                float(cleaned)
                break
            except ValueError:
                name_parts.append(part)
        name = " ".join(name_parts).strip().lower()
        name = name.rstrip(":").strip()
        return name or None, val
