# SEC Filing Retriever Module

import requests
import re


def get_company_cik(ticker: str) -> str:
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "Himaja Kavuri hkavuri@usc.edu"}
    response = requests.get(url, headers=headers)
    data = response.json()
    for key, company in data.items():
        if company["ticker"].upper() == ticker.upper():
            return str(company["cik_str"]).zfill(10)
    return None


def get_latest_10k_accession(cik: str) -> tuple:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {"User-Agent": "Himaja Kavuri hkavuri@usc.edu"}
    response = requests.get(url, headers=headers)
    data = response.json()
    filings = data["filings"]["recent"]
    forms = filings["form"]
    accession_numbers = filings["accessionNumber"]
    primary_docs = filings["primaryDocument"]
    for i, form in enumerate(forms):
        if form == "10-K":
            return accession_numbers[i], primary_docs[i]
    return None, None


def get_risk_factors(ticker: str) -> str:
    try:
        cik = get_company_cik(ticker)
        if not cik:
            return f"Could not find SEC filing for {ticker}."

        accession, primary_doc = get_latest_10k_accession(cik)
        if not accession or not primary_doc:
            return f"Could not retrieve 10-K filing for {ticker}."

        # Use the primaryDocument field directly — no guessing needed
        cik_int = str(int(cik))
        accession_clean = accession.replace("-", "")
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_clean}/{primary_doc}"

        headers = {"User-Agent": "Himaja Kavuri hkavuri@usc.edu"}
        response = requests.get(doc_url, headers=headers)
        text = response.text

        # Find Item 1A — skip table of contents
        start_idx = -1
        search_from = 0
        attempts = 0

        while attempts < 20:
            idx = text.find('Item 1A', search_from)
            if idx == -1:
                break

            sample = re.sub(r'<[^>]+>', '', text[idx:idx+2000])
            sample = re.sub(r'\s+', ' ', sample).strip()

            # Real section has substantial content with multiple risk mentions
            if (len(sample) > 500 and
                    sample.lower().count('risk') > 2):
                start_idx = idx
                break

            search_from = idx + 100
            attempts += 1

        # Fallback — search for RISK FACTORS directly
        if start_idx == -1:
            for marker in ["RISK FACTORS", "Risk Factors"]:
                idx = 0
                while True:
                    idx = text.find(marker, idx)
                    if idx == -1:
                        break
                    sample = re.sub(r'<[^>]+>', '', text[idx:idx+2000])
                    sample = re.sub(r'\s+', ' ', sample).strip()
                    if len(sample) > 500:
                        start_idx = idx
                        break
                    idx += 100
                if start_idx != -1:
                    break

        if start_idx == -1:
            return f"Could not extract risk factors from {ticker} 10-K filing."

        # Find end of section
        end_idx = len(text)
        for marker in ["Item 1B", "ITEM 1B", "Item 2", "ITEM 2"]:
            idx = text.find(marker, start_idx + 1000)
            if idx != -1:
                end_idx = min(end_idx, idx)

        # Clean text
        risk_text = text[start_idx:end_idx]
        clean_text = re.sub(r'<[^>]+>', ' ', risk_text)
        clean_text = re.sub(r'&#\d+;', ' ', clean_text)
        clean_text = re.sub(r'&[a-z]+;', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        if len(clean_text) > 4000:
            clean_text = clean_text[:4000] + "... [truncated]"

        return f"Source: SEC EDGAR 10-K Filing\n\n{clean_text}"

    except Exception as e:
        return f"Could not retrieve SEC filing: {str(e)}"