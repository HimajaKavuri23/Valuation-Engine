
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class REITAssumptions:
    cost_of_equity: float = 0.08      # Required return for REIT
    growth_rate: float = 0.03         # FFO growth rate
    projection_years: int = 5         # Years to project FFO
    payout_ratio: float = 0.90        # REITs must pay 90%+ of taxable income


class REITValuationModel:
    def __init__(self, financials: dict, assumptions: REITAssumptions = None):
        self.financials = financials
        self.assumptions = assumptions or REITAssumptions()
        self.income_stmt = financials.get("income_stmt", pd.DataFrame())
        self.cash_flow = financials.get("cash_flow", pd.DataFrame())
        self.balance_sheet = financials.get("balance_sheet", pd.DataFrame())

    def _get_latest(self, df: pd.DataFrame, *keys) -> float:
        for key in keys:
            if key in df.index:
                vals = df.loc[key].dropna()
                if not vals.empty:
                    return float(vals.iloc[0])
        return 0.0

    def calculate_ffo(self) -> float:
        """FFO = Net Income + Depreciation & Amortization - Gains on Property Sales"""
        net_income = self._get_latest(
            self.income_stmt,
            "Net Income", "NetIncome", "netIncome"
        )
        depreciation = self._get_latest(
            self.cash_flow,
            "Depreciation And Amortization",
            "Depreciation Amortization Depletion",
            "DepreciationAndAmortization"
        )
        # Gains on sales — reduce FFO (not recurring)
        gains = self._get_latest(
            self.income_stmt,
            "Gain Loss On Sale Of Business",
            "GainLossOnSaleOfBusiness",
        )
        ffo = net_income + depreciation - gains
        return ffo

    def calculate_affo(self, ffo: float) -> float:
        """AFFO = FFO - Maintenance CapEx - Straight Line Rent Adjustments"""
        total_capex = abs(self._get_latest(
            self.cash_flow,
            "Purchase Of Investment Properties",
            "Capital Expenditure",
            "CapitalExpenditure",
            "Purchase Of Property Plant And Equipment"
        ))
        maintenance_capex = total_capex * 0.10
        affo = ffo - maintenance_capex
        return affo

    def calculate(self) -> dict:
        try:
            shares = self.financials.get("shares_outstanding", 1)
            current_price = self.financials.get("current_price", 0)

            if shares <= 0:
                return {"error": "Insufficient share data for REIT valuation"}

            # Calculate FFO and AFFO
            ffo = self.calculate_ffo()
            affo = self.calculate_affo(ffo)

            ffo_per_share = ffo / shares
            affo_per_share = affo / shares

            # Price/FFO multiple (standard REIT metric)
            price_to_ffo = current_price / ffo_per_share if ffo_per_share > 0 else 0

            # Intrinsic value via dividend discount on AFFO
            # V = AFFO_per_share * payout / (cost_of_equity - growth)
            coe = self.assumptions.cost_of_equity
            g = self.assumptions.growth_rate

            if coe <= g:
                g = coe - 0.01  # Prevent division by zero

            annual_dividend = affo_per_share * self.assumptions.payout_ratio
            intrinsic_value = annual_dividend / (coe - g) if (coe - g) > 0 else 0

            upside_pct = ((intrinsic_value - current_price) / current_price * 100) \
                if current_price > 0 else 0

            # Project FFO growth
            projected_ffo = []
            proj_ffo = ffo_per_share
            for _ in range(self.assumptions.projection_years):
                proj_ffo *= (1 + g)
                projected_ffo.append(round(proj_ffo, 2))

            return {
                "methodology": "FFO/AFFO Valuation",
                "intrinsic_value": round(intrinsic_value, 2),
                "current_price": current_price,
                "upside_downside_pct": round(upside_pct, 2),
                "ffo": round(ffo, 0),
                "affo": round(affo, 0),
                "ffo_per_share": round(ffo_per_share, 2),
                "affo_per_share": round(affo_per_share, 2),
                "price_to_ffo": round(price_to_ffo, 2),
                "annual_dividend_from_affo": round(annual_dividend, 2),
                "cost_of_equity": round(coe * 100, 2),
                "growth_rate": round(g * 100, 2),
                "projected_ffo_per_share": projected_ffo,
                "shares": shares,
            }

        except Exception as e:
            return {"error": f"REIT valuation failed: {str(e)}"}