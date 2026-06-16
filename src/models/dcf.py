"""
DCF Model Module
----------------
Discounted Cash Flow valuation model.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class DCFAssumptions:
    projection_years: int   = 5
    revenue_growth:   float = 0.08
    fcf_margin:       float = 0.25
    terminal_growth:  float = 0.03
    risk_free_rate:   float = 0.045
    equity_risk_prem: float = 0.055
    debt_spread:      float = 0.02
    tax_rate:         float = 0.21


class WACCCalculator:

    def __init__(self, assumptions: DCFAssumptions):
        self.a = assumptions

    def cost_of_equity(self, beta: float) -> float:
        return self.a.risk_free_rate + beta * self.a.equity_risk_prem

    def cost_of_debt(self) -> float:
        pre_tax = self.a.risk_free_rate + self.a.debt_spread
        return pre_tax * (1 - self.a.tax_rate)

    def calculate(self, market_cap: float, total_debt: float, beta: float) -> float:
        total_value = market_cap + total_debt
        if total_value == 0:
            return 0.10
        weight_equity = market_cap / total_value
        weight_debt   = total_debt / total_value
        re = self.cost_of_equity(beta)
        rd = self.cost_of_debt()
        return (weight_equity * re) + (weight_debt * rd)


class DCFModel:

    def __init__(self, financial_data: dict, assumptions: DCFAssumptions):
        self.data        = financial_data
        self.assumptions = assumptions
        self.overview    = financial_data["overview"]
        self.wacc_calc   = WACCCalculator(assumptions)
        self.results     = {}

    def _get_total_debt(self) -> float:
        bs = self.data.get("balance_sheet", pd.DataFrame())
        if bs.empty:
            return 0
        for label in [
            "Total Debt",
            "Long Term Debt",
            "Current Debt",
            "Total Liabilities Net Minority Interest",
            "Long Term Debt And Capital Lease Obligation",
        ]:
            if label in bs.index:
                val = bs.loc[label].iloc[0]
                return float(val) if not pd.isna(val) else 0
        return 0

    def _get_cash(self) -> float:
        bs = self.data.get("balance_sheet", pd.DataFrame())
        if bs.empty:
            return 0
        for label in [
            "Cash And Cash Equivalents",
            "Cash Cash Equivalents And Short Term Investments",
        ]:
            if label in bs.index:
                val = bs.loc[label].iloc[0]
                return float(val) if not pd.isna(val) else 0
        return 0

    def _get_base_fcf(self) -> float:
        fcf = self.data.get("fcf", pd.Series())
        if fcf.empty:
            return 0
        clean = fcf.dropna()
        return float(clean.iloc[0]) if not clean.empty else 0

    def project_fcf(self) -> list:
        base_fcf    = self._get_base_fcf()
        growth_rate = self.assumptions.revenue_growth
        n_years     = self.assumptions.projection_years
        projected   = []
        current_fcf = base_fcf
        for year in range(1, n_years + 1):
            current_fcf = current_fcf * (1 + growth_rate)
            projected.append(current_fcf)
        return projected

    def calculate_terminal_value(self, final_year_fcf: float, wacc: float) -> float:
        g = self.assumptions.terminal_growth
        if wacc <= g:
            raise ValueError(f"WACC ({wacc:.1%}) must be greater than terminal growth ({g:.1%})")
        return final_year_fcf * (1 + g) / (wacc - g)

    def discount_cash_flows(self, cash_flows: list, wacc: float) -> list:
        return [cf / (1 + wacc) ** (i + 1) for i, cf in enumerate(cash_flows)]

    def calculate_intrinsic_value(self) -> dict:
        market_cap     = self.overview.get("market_cap", 0)
        total_debt     = self._get_total_debt()
        beta           = self.overview.get("beta", 1.0)
        wacc           = self.wacc_calc.calculate(market_cap, total_debt, beta)
        projected_fcfs = self.project_fcf()
        terminal_value = self.calculate_terminal_value(projected_fcfs[-1], wacc)
        pv_fcfs        = self.discount_cash_flows(projected_fcfs, wacc)
        pv_terminal    = terminal_value / (1 + wacc) ** self.assumptions.projection_years
        enterprise_value = sum(pv_fcfs) + pv_terminal
        cash           = self._get_cash()
        equity_value   = enterprise_value - total_debt + cash
        shares         = self.overview.get("shares_outstanding", 1)
        intrinsic_value_per_share = equity_value / shares if shares > 0 else 0
        current_price  = self.overview.get("current_price", 0)
        upside_downside = ((intrinsic_value_per_share - current_price) / current_price * 100
                           if current_price > 0 else 0)

        self.results = {
            "ticker":              self.overview["ticker"],
            "company_name":        self.overview["name"],
            "current_price":       current_price,
            "intrinsic_value":     intrinsic_value_per_share,
            "upside_downside_pct": upside_downside,
            "wacc":                wacc,
            "terminal_growth":     self.assumptions.terminal_growth,
            "enterprise_value":    enterprise_value,
            "equity_value":        equity_value,
            "pv_fcfs":             pv_fcfs,
            "projected_fcfs":      projected_fcfs,
            "pv_terminal":         pv_terminal,
            "terminal_value":      terminal_value,
            "total_debt":          total_debt,
            "cash":                cash,
            "shares_outstanding":  shares,
            "base_fcf":            self._get_base_fcf(),
        }
        return self.results

    def sensitivity_analysis(self) -> pd.DataFrame:
        wacc_range   = np.arange(0.06, 0.14, 0.01)
        growth_range = np.arange(0.01, 0.06, 0.01)
        table = pd.DataFrame(
            index=[f"{w:.0%}" for w in wacc_range],
            columns=[f"{g:.0%}" for g in growth_range]
        )
        base_fcf   = self._get_base_fcf()
        total_debt = self._get_total_debt()
        cash       = self._get_cash()
        shares     = self.overview.get("shares_outstanding", 1)

        for w in wacc_range:
            for g in growth_range:
                if w <= g:
                    table.loc[f"{w:.0%}", f"{g:.0%}"] = "N/A"
                    continue
                try:
                    fcfs  = [base_fcf * (1 + self.assumptions.revenue_growth) ** yr
                             for yr in range(1, self.assumptions.projection_years + 1)]
                    tv    = fcfs[-1] * (1 + g) / (w - g)
                    pv    = sum([cf / (1 + w) ** (i + 1) for i, cf in enumerate(fcfs)])
                    pv_tv = tv / (1 + w) ** self.assumptions.projection_years
                    ev    = pv + pv_tv
                    eq    = ev - total_debt + cash
                    val   = eq / shares
                    table.loc[f"{w:.0%}", f"{g:.0%}"] = f"${val:.0f}"
                except Exception:
                    table.loc[f"{w:.0%}", f"{g:.0%}"] = "N/A"
        return table

    def print_summary(self):
        if not self.results:
            self.calculate_intrinsic_value()
        r = self.results
        verdict = "UNDERVALUED" if r["upside_downside_pct"] > 0 else "OVERVALUED"
        print(f"\n{'='*55}")
        print(f"  DCF VALUATION — {r['company_name']} ({r['ticker']})")
        print(f"{'='*55}")
        print(f"  Current Price:     ${r['current_price']:.2f}")
        print(f"  Intrinsic Value:   ${r['intrinsic_value']:.2f}")
        print(f"  Upside/Downside:   {r['upside_downside_pct']:+.1f}%  {verdict}")
        print(f"  WACC:              {r['wacc']:.2%}")
        print(f"  Enterprise Value:  ${r['enterprise_value']:,.0f}")
        print(f"{'='*55}\n")


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data.fetcher import FinancialDataFetcher

    fetcher     = FinancialDataFetcher("AAPL")
    data        = fetcher.get_all_financials()
    assumptions = DCFAssumptions()
    model       = DCFModel(data, assumptions)
    results     = model.calculate_intrinsic_value()
    model.print_summary()

    print("\nSensitivity Analysis:")
    print(model.sensitivity_analysis().to_string())