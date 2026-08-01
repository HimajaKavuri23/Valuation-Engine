# Monte Carlo Simulation Module

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple
from src.models.dcf import DCFModel, DCFAssumptions


@dataclass
class SimulationConfig:
    
    n_simulations:      int   = 10000  # Number of scenarios to run

    growth_mean:        float = 0.08
    growth_std:         float = 0.03

    wacc_mean:          float = 0.10
    wacc_std:           float = 0.015

    terminal_mean:      float = 0.03
    terminal_std:       float = 0.005

    fcf_margin_std:     float = 0.02


class MonteCarloSimulator:
    

    def __init__(self, financial_data: dict, config: SimulationConfig = None):
        self.data    = financial_data
        self.config  = config or SimulationConfig()
        self.results = []

    def _run_single_simulation(
        self,
        growth_rate:    float,
        wacc_override:  float,
        terminal_growth: float
    ) -> float:
        
        
        try:
            assumptions = DCFAssumptions(
                projection_years = 5,
                revenue_growth   = max(0.001, growth_rate),    # Floor at 0.1%
                terminal_growth  = max(0.001, terminal_growth), # Floor at 0.1%
                risk_free_rate   = 0.045,
                equity_risk_prem = 0.055,
            )

            model   = DCFModel(self.data, assumptions)
            results = model.calculate_intrinsic_value()

            
            wacc = max(0.05, wacc_override)  # Floor at 5%
            if wacc <= assumptions.terminal_growth:
                return np.nan


            base_fcf   = results["base_fcf"]
            total_debt = results["total_debt"]
            cash       = results["cash"]
            shares     = results["shares_outstanding"]

            fcfs = [base_fcf * (1 + growth_rate) ** yr
                    for yr in range(1, 6)]
            tv   = fcfs[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
            pv   = sum([cf / (1 + wacc) ** (i + 1) for i, cf in enumerate(fcfs)])
            pv_tv = tv / (1 + wacc) ** 5
            ev   = pv + pv_tv
            eq   = ev - total_debt + cash

            return eq / shares if shares > 0 else np.nan

        except Exception:
            return np.nan

    def run(self) -> pd.DataFrame:
        
        cfg = self.config
        print(f"\nRunning {cfg.n_simulations:,} Monte Carlo simulations...")

        np.random.seed(42)  # For reproducibility
        growth_samples   = np.random.normal(cfg.growth_mean,   cfg.growth_std,   cfg.n_simulations)
        wacc_samples     = np.random.normal(cfg.wacc_mean,     cfg.wacc_std,     cfg.n_simulations)
        terminal_samples = np.random.normal(cfg.terminal_mean, cfg.terminal_std, cfg.n_simulations)

        intrinsic_values = []
        for i in range(cfg.n_simulations):
            val = self._run_single_simulation(
                growth_rate     = growth_samples[i],
                wacc_override   = wacc_samples[i],
                terminal_growth = terminal_samples[i]
            )
            intrinsic_values.append(val)

            # Progress indicator every 2000 runs
            if (i + 1) % 2000 == 0:
                print(f"  {i+1:,} / {cfg.n_simulations:,} simulations complete...")

        self.results = pd.DataFrame({
            "growth_rate":     growth_samples,
            "wacc":            wacc_samples,
            "terminal_growth": terminal_samples,
            "intrinsic_value": intrinsic_values,
        }).dropna()

        print(f"✓ {len(self.results):,} valid simulations completed\n")
        return self.results

    def get_statistics(self) -> dict:
        
        if self.results is None or len(self.results) == 0:
            raise ValueError("Run simulations first with .run()")

        vals           = self.results["intrinsic_value"]
        current_price  = self.data["overview"].get("current_price", 0)

        stats = {
            "mean":          vals.mean(),
            "median":        vals.median(),
            "std":           vals.std(),
            "p10":           vals.quantile(0.10),  # Bear case
            "p25":           vals.quantile(0.25),
            "p50":           vals.quantile(0.50),  # Base case
            "p75":           vals.quantile(0.75),
            "p90":           vals.quantile(0.90),  # Bull case
            "current_price": current_price,
            "prob_undervalued": (vals > current_price).mean(),  # % of scenarios where stock is cheap
            "n_simulations": len(vals),
        }

        return stats

    def print_summary(self):
        
        stats = self.get_statistics()
        ticker = self.data["overview"]["ticker"]
        name   = self.data["overview"]["name"]

        print(f"\n{'='*55}")
        print(f"  MONTE CARLO SIMULATION — {name} ({ticker})")
        print(f"  {stats['n_simulations']:,} simulations")
        print(f"{'='*55}")
        print(f"  Current Market Price:  ${stats['current_price']:.2f}")
        print(f"{'─'*55}")
        print(f"  Bear Case  (10th pct): ${stats['p10']:.2f}")
        print(f"  Low Case   (25th pct): ${stats['p25']:.2f}")
        print(f"  Base Case  (50th pct): ${stats['p50']:.2f}")
        print(f"  High Case  (75th pct): ${stats['p75']:.2f}")
        print(f"  Bull Case  (90th pct): ${stats['p90']:.2f}")
        print(f"{'─'*55}")
        print(f"  Mean Intrinsic Value:  ${stats['mean']:.2f}")
        print(f"  Std Deviation:         ${stats['std']:.2f}")
        print(f"{'─'*55}")
        prob = stats['prob_undervalued'] * 100
        print(f"  Probability stock is UNDERVALUED: {prob:.1f}%")
        if prob > 50:
            print(f"  Verdict: MORE LIKELY UNDERVALUED 🟢")
        else:
            print(f"  Verdict: MORE LIKELY OVERVALUED 🔴")
        print(f"{'='*55}\n")


# Test directly 
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data.fetcher import FinancialDataFetcher

    # Fetch data
    fetcher = FinancialDataFetcher("AAPL")
    data    = fetcher.get_all_financials()

    # Run simulation
    config    = SimulationConfig(n_simulations=10000)
    simulator = MonteCarloSimulator(data, config)
    results   = simulator.run()

    # Print summary
    simulator.print_summary()

    # Show distribution stats
    print("Distribution of intrinsic values:")
    print(results["intrinsic_value"].describe().apply(lambda x: f"${x:.2f}"))