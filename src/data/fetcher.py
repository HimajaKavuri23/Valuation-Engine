"""
Data Fetcher Module
-------------------
Pulls live financial statements from Yahoo Finance for any ticker.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime


class FinancialDataFetcher:

    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self.info = self._get_info()

    def _get_info(self) -> dict:
        try:
            return self.stock.info
        except Exception as e:
            print(f"Warning: Could not fetch info for {self.ticker}: {e}")
            return {}

    def get_company_overview(self) -> dict:
        return {
            "ticker":             self.ticker,
            "name":               self.info.get("longName", "N/A"),
            "sector":             self.info.get("sector", "N/A"),
            "industry":           self.info.get("industry", "N/A"),
            "market_cap":         self.info.get("marketCap", 0),
            "current_price":      self.info.get("currentPrice", 0),
            "beta":               self.info.get("beta", 1.0),
            "shares_outstanding": self.info.get("sharesOutstanding", 0),
            "fetched_at":         datetime.now().isoformat(),
        }

    def get_income_statement(self) -> pd.DataFrame:
        try:
            df = self.stock.financials
            if df is None or df.empty:
                return pd.DataFrame()
            return df
        except Exception as e:
            print(f"Error fetching income statement: {e}")
            return pd.DataFrame()

    def get_balance_sheet(self) -> pd.DataFrame:
        try:
            df = self.stock.balance_sheet
            if df is None or df.empty:
                return pd.DataFrame()
            return df
        except Exception as e:
            print(f"Error fetching balance sheet: {e}")
            return pd.DataFrame()

    def get_cash_flow(self) -> pd.DataFrame:
        try:
            df = self.stock.cashflow
            if df is None or df.empty:
                return pd.DataFrame()
            return df
        except Exception as e:
            print(f"Error fetching cash flow: {e}")
            return pd.DataFrame()

    def get_free_cash_flow(self) -> pd.Series:
        cf = self.get_cash_flow()
        if cf.empty:
            return pd.Series()
        try:
            operating_cf = pd.Series()
            for label in ["Operating Cash Flow", "Cash Flow From Operations",
                          "Net Cash Provided By Operating Activities"]:
                if label in cf.index:
                    operating_cf = cf.loc[label]
                    break

            capex = pd.Series()
            for label in ["Capital Expenditure", "Capital Expenditures",
                          "Purchase Of Property Plant And Equipment",
                          "Purchases Of Property And Equipment"]:
                if label in cf.index:
                    capex = cf.loc[label]
                    break

            if operating_cf.empty or capex.empty:
                print(f"Warning: Could not find Operating CF or CapEx rows")
                print(f"Available rows: {list(cf.index)}")
                return pd.Series()

            fcf = operating_cf + capex
            fcf.name = "Free Cash Flow"
            return fcf

        except Exception as e:
            print(f"Error calculating FCF: {e}")
            return pd.Series()

    def get_all_financials(self) -> dict:
        print(f"\nFetching data for {self.ticker}...")
        overview      = self.get_company_overview()
        income_stmt   = self.get_income_statement()
        balance_sheet = self.get_balance_sheet()
        cash_flow     = self.get_cash_flow()
        fcf           = self.get_free_cash_flow()

        print(f"✓ {overview['name']} ({self.ticker})")
        print(f"  Sector:       {overview['sector']}")
        print(f"  Market Cap:   ${overview['market_cap']:,.0f}")
        print(f"  Price:        ${overview['current_price']:.2f}")
        print(f"  Beta:         {overview['beta']}")
        if not fcf.empty:
            print(f"  Latest FCF:   ${fcf.iloc[0]:,.0f}")

        return {
            "overview":      overview,
            "income_stmt":   income_stmt,
            "balance_sheet": balance_sheet,
            "cash_flow":     cash_flow,
            "fcf":           fcf,
        }


if __name__ == "__main__":
    fetcher = FinancialDataFetcher("AAPL")
    data = fetcher.get_all_financials()

    print("\n── Free Cash Flow ──")
    if not data["fcf"].empty:
        for date, value in data["fcf"].items():
            print(f"  {date.year}: ${value:,.0f}")