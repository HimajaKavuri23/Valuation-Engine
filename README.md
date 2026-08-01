# Intelligent Financial Valuation Engine

A DCF-based valuation tool for any US-listed company, built with Python and deployed on Streamlit.

## What it does

- Pulls live financial statements via the yfinance API
- Calculates intrinsic value per share using a Discounted Cash Flow model
- Runs 10,000 Monte Carlo scenarios to produce a probability distribution of intrinsic value
- Generates a 40-scenario sensitivity matrix across discount rates and growth rate assumptions
- Deployed live at [valuation-engine.streamlit.app](https://valuation-engine.streamlit.app)

## Methodology

The DCF model follows standard financial analysis practice:
- WACC is calculated using CAPM with the stock's beta, risk-free rate, and equity risk premium
- Free Cash Flow is projected over 5 years using historical growth rates
- Terminal Value is computed using the Gordon Growth Model
- Monte Carlo simulation randomizes key assumptions to quantify valuation uncertainty

## Tech stack

Python, Pandas, NumPy, Plotly, Streamlit, yfinance

## Project structure

src/data/fetcher.py        - Live financial data ingestion
src/models/dcf.py          - DCF model and WACC calculator
src/models/monte_carlo.py  - Monte Carlo simulation engine
src/dashboard/app.py       - Streamlit dashboard

## Author

Himaja Kavuri - MS Analytics, USC
Portfolio: https://himajakavuri23.github.io
LinkedIn: https://linkedin.com/in/himaja-kavuri
