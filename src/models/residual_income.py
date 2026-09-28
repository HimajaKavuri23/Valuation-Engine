
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class RIMAssumptions:
    cost_of_equity: float = 0.10      # Required return on equity
    fade_years: int = 5                # Years of excess return fade
    terminal_growth: float = 0.03     # Long-run growth rate
    mean_reversion_rate: float = 0.5  # How fast ROE reverts to COE


class ResidualIncomeModel:
    def __init__(self, financials: dict, assumptions: RIMAssumptions = None):
        self.financials = financials
        self.assumptions = assumptions or RIMAssumptions()
        self.income_stmt = financials.get("income_stmt", pd.DataFrame())
        self.balance_sheet = financials.get("balance_sheet", pd.DataFrame())

    def _get_latest(self, df: pd.DataFrame, *keys) -> float:
        """Get most recent value for any of the given keys."""
        for key in keys:
            if key in df.index:
                vals = df.loc[key].dropna()
                if not vals.empty:
                    return float(vals.iloc[0])
        return 0.0

    def calculate(self) -> dict:
        try:
            # Extract key inputs
            net_income = self._get_latest(
                self.income_stmt,
                "Net Income", "NetIncome", "netIncome"
            )
            book_equity = self._get_latest(
                self.balance_sheet,
                "Stockholders Equity", "StockholdersEquity",
                "Total Stockholder Equity", "commonStockEquity"
            )
            shares = self.financials.get("shares_outstanding", 1)
            current_price = self.financials.get("current_price", 0)

            if book_equity <= 0 or shares <= 0:
                return {"error": "Insufficient balance sheet data for RIM"}

            # Current ROE
            roe = net_income / book_equity
            coe = self.assumptions.cost_of_equity

            # Book value per share
            bvps = book_equity / shares

            # Project residual income over fade period
            pv_ri = 0.0
            projected_roe = roe
            projected_bvps = bvps

            for year in range(1, self.assumptions.fade_years + 1):
                # ROE fades toward COE over time
                projected_roe = (
                    projected_roe * (1 - self.assumptions.mean_reversion_rate) +
                    coe * self.assumptions.mean_reversion_rate
                )

                # Residual income per share
                ri = (projected_roe - coe) * projected_bvps

                # Discount back
                pv_ri += ri / (1 + coe) ** year

                # Grow book value
                projected_bvps *= (1 + projected_roe * 0.5)

            # Terminal value — perpetuity of final year RI
            final_ri = (projected_roe - coe) * projected_bvps
            terminal_value = final_ri / (coe - self.assumptions.terminal_growth) \
                if coe > self.assumptions.terminal_growth else 0
            pv_terminal = terminal_value / (1 + coe) ** self.assumptions.fade_years

            # Intrinsic value per share
            intrinsic_value = bvps + pv_ri + pv_terminal

            # Upside/downside
            upside_pct = ((intrinsic_value - current_price) / current_price * 100) \
                if current_price > 0 else 0

            return {
                "methodology": "Residual Income Model",
                "intrinsic_value": round(intrinsic_value, 2),
                "current_price": current_price,
                "upside_downside_pct": round(upside_pct, 2),
                "book_value_per_share": round(bvps, 2),
                "current_roe": round(roe * 100, 2),
                "cost_of_equity": round(coe * 100, 2),
                "pv_residual_income": round(pv_ri, 2),
                "pv_terminal": round(pv_terminal, 2),
                "net_income": net_income,
                "book_equity": book_equity,
                "shares": shares,
                "excess_return": round((roe - coe) * 100, 2),
            }

        except Exception as e:
            return {"error": f"RIM calculation failed: {str(e)}"}