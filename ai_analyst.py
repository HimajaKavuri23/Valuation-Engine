# AI Analyst Module

import anthropic
from rag_retriever import get_rag_context


def get_ai_analysis(results: dict, ticker: str = None, user_question: str = None) -> str:
    client = anthropic.Anthropic()

    # RAG retrieval
    rag_query = user_question if user_question else "key risk factors material risks business operations"
    rag_result = get_rag_context(ticker, query=rag_query, top_k=5) if ticker else {'context': '', 'success': False}

    sec_context = rag_result['context'] if rag_result['success'] else "SEC filing retrieval unavailable."
    source_url = rag_result.get('source_url', '')

    # Pre-calculate everything in Python
    tv_pct = round(
        results.get('pv_terminal', 0) / results.get('equity_value', 1) * 100, 1
    ) if results.get('equity_value', 0) != 0 else 0

    net_debt = results.get('total_debt', 0) - results.get('cash', 0)

    upside = results.get('upside_downside_pct', 0)
    if upside > 0:
        valuation_signal = f"a {upside:.1f}% upside relative to the current market price under the model's assumptions"
    else:
        valuation_signal = f"a {abs(upside):.1f}% downside relative to the current market price under the model's assumptions"

    context = f"""
You are a financial analyst assistant. All numbers below were calculated by Python.
Do not recalculate or infer any values. Use conditional, precise language throughout.
Never say a stock "is" overvalued or undervalued. Always say "the DCF model indicates" or "under current assumptions."
Do not assign subjective confidence labels.

PRE-CALCULATED VALUATION DATA (do not recalculate):
Intrinsic Value Per Share: {results.get('intrinsic_value', 'N/A')}
Current Market Price: {results.get('current_price', 'N/A')}
Upside / Downside: {results.get('upside_downside_pct', 'N/A')}%
Valuation Signal: The DCF model indicates {valuation_signal}
WACC: {results.get('wacc', 'N/A')}
Total Equity Value: {results.get('equity_value', 'N/A')}
PV of Terminal Value: {results.get('pv_terminal', 'N/A')}
Terminal Value as % of Equity (pre-calculated): {tv_pct}%
Total Debt: {results.get('total_debt', 'N/A')}
Cash: {results.get('cash', 'N/A')}
Net Debt (pre-calculated): {net_debt:,.0f}

RELEVANT PASSAGES FROM SEC 10-K FILING (retrieved via semantic search):
Source: {source_url}

{sec_context}

RESPONSE RULES:
1. Separate your response into two clearly labeled sections:
   - MODEL-IMPLIED RISKS: derived only from the DCF data above
   - COMPANY-SPECIFIC RISKS: derived only from the SEC passages above
2. For every company-specific risk, cite it as: (Source: SEC EDGAR 10-K, Item 1A, Passage [N])
   where N is the passage number from the retrieved context
3. Do not mix the two categories
4. Do not calculate any numbers — only interpret what Python provided
5. State terminal value as: "{tv_pct}% of estimated equity value is attributable to terminal value"
6. Use conditional language: "the model indicates", "under current assumptions", "based on disclosed risk factors"
7. Use "Net Debt Position" not "Net Debt Burden"
8. Do not use ** bold ** markdown around dollar amounts or numbers
9. End with this exact disclaimer: "This analysis is based solely on pre-calculated model outputs and disclosed SEC filing language. Market prices may reflect expectations and information not captured by the model's current assumptions."
"""

    if user_question:
        prompt = f"{context}\n\nUser question: {user_question}\n\nAnswer using only the data provided above. Cite specific passages when referencing SEC filing content."
    else:
        prompt = f"{context}\n\nProvide an executive summary separating model-implied risks from company-specific risks. Cite passage numbers for all SEC filing references."

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text