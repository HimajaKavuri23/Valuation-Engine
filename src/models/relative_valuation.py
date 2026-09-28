
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class RelativeValuationAssumptions:
    ev_revenue_multiple: float = None    # If None, use industry median
    ev_ebitda_multiple: float = None     # If None, use industry median
    target_multiple_discount: float = 0.20  # Discount to industry multiple for risk


class RelativeValuationModel:
    # Industry median multiples (approximate)
    INDUSTRY_EV_REVENUE = {
        "Technology": 6.0,
        "Communication Services": 4.0,
        "Consumer Cyclical": 1.5,
        "Consumer Defensive": 1.2,
        "Healthcare": 4.0,
        "Industrials": 2.0,
        "Energy": 1.5,
        "Financial Services": 3.0,
        "Real Estate": 8.0,
        "Basic Materials": 1.5,
        "Utilities": 2.5,
    }

    INDUSTRY_EV_EBITDA = {
        "Technology": 20.0,
        "Communication Services": 12.0,
        "Consumer Cyclical": 10.0,
        "Consumer Defensive": 12.0,
        "Healthcare": 15.0,
        "Industrials": 12.0,
        "Energy": 8.0,
        "Financial Services": 12.0,
        "Real Estate": 18.0,
        "Basic Materials": 8.0,
        "Utilities": 12.0,
    }

    def __init__(self, financials: dict, assumptions: RelativeValuationAssumptions = None):
        self.financials = financials
        self.assumptions = assumptions or RelativeValuationAssumptions()
        self.income_stmt = financials.get("income_stmt", pd.DataFrame())
        self.balance_sheet = financials.get("balance_sheet", pd.DataFrame())

    def _get_latest(self, df: pd.DataFrame, *keys) -> float:
        for key in keys:
            if key in df.index:
                vals = df.loc[key].dropna()
                if not vals.empty:
                    return float(vals.iloc[0])
        return 0.0

    def calculate(self) -> dict:
        try:
            sector = self.financials.get("sector", "Technology")
            shares = self.financials.get("shares_outstanding", 1)
            current_price = self.financials.get("current_price", 0)
            market_cap = current_price * shares

            # Get financial metrics
            revenue = self._get_latest(
                self.income_stmt,
                "Total Revenue", "TotalRevenue", "Revenue"
            )
            ebitda = self._get_latest(
                self.income_stmt,
                "EBITDA", "Ebitda", "EBITDAActual"
            )
            total_debt = self._get_latest(
                self.balance_sheet,
                "Total Debt", "TotalDebt", "LongTermDebt"
            )
            cash = self._get_latest(
                self.balance_sheet,
                "Cash And Cash Equivalents", "CashAndCashEquivalents",
                "Cash", "CashCashEquivalentsAndShortTermInvestments"
            )

            # Enterprise value
            ev = market_cap + total_debt - cash

            # Get industry multiples
            ev_rev_multiple = self.INDUSTRY_EV_REVENUE.get(sector, 3.0)
            ev_ebitda_multiple = self.INDUSTRY_EV_EBITDA.get(sector, 12.0)

            # Apply discount for negative FCF risk
            discount = self.assumptions.target_multiple_discount
            adj_ev_rev = ev_rev_multiple * (1 - discount)
            adj_ev_ebitda = ev_ebitda_multiple * (1 - discount) if ebitda > 0 else None

            # Calculate intrinsic values
            intrinsic_ev_revenue = adj_ev_rev * revenue
            intrinsic_value_revenue = (intrinsic_ev_revenue - total_debt + cash) / shares \
                if shares > 0 else 0

            intrinsic_value_ebitda = None
            if ebitda > 0 and adj_ev_ebitda:
                intrinsic_ev_ebitda = adj_ev_ebitda * ebitda
                intrinsic_value_ebitda = (intrinsic_ev_ebitda - total_debt + cash) / shares

            # Blended intrinsic value
            if intrinsic_value_ebitda:
                intrinsic_value = (intrinsic_value_revenue + intrinsic_value_ebitda) / 2
                methodology_note = "EV/Revenue and EV/EBITDA blended"
            else:
                intrinsic_value = intrinsic_value_revenue
                methodology_note = "EV/Revenue only (EBITDA negative or unavailable)"

            upside_pct = ((intrinsic_value - current_price) / current_price * 100) \
                if current_price > 0 else 0

            # Current multiples
            current_ev_revenue = ev / revenue if revenue > 0 else 0
            current_ev_ebitda = ev / ebitda if ebitda > 0 else None

            return {
                "methodology": "Relative Valuation",
                "methodology_note": methodology_note,
                "intrinsic_value": round(intrinsic_value, 2),
                "current_price": current_price,
                "upside_downside_pct": round(upside_pct, 2),
                "revenue": revenue,
                "ebitda": ebitda if ebitda else 0,
                "enterprise_value": ev,
                "current_ev_revenue": round(current_ev_revenue, 2),
                "current_ev_ebitda": round(current_ev_ebitda, 2) if current_ev_ebitda else None,
                "industry_ev_revenue": ev_rev_multiple,
                "industry_ev_ebitda": ev_ebitda_multiple,
                "adj_ev_revenue": round(adj_ev_rev, 2),
                "total_debt": total_debt,
                "cash": cash,
                "sector": sector,
                "shares": shares,
            }

        except Exception as e:
            return {"error": f"Relative valuation failed: {str(e)}"}