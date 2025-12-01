#utility.py

import os 
import requests 
import requests 
import re
import time


API_KEY  = os.getenv("OPENAI_API_KEY", "cse476")
API_BASE = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")  
MODEL    = os.getenv("MODEL_NAME", "bens_model")

"""
tutorial code 
"""
def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
                                model: str = MODEL,
                                temperature: float = 0.0,
                                timeout: int = 60) -> dict:
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int, 'error': str or None, 'headers': dict }
    
    (From tutorial - DO NOT MODIFY)
    """
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 128,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs   = dict(resp.headers)
        if status == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
        else:
            err_text = None
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}

"""
TODO: create regex for easy conversions of prompt 
- normalize_text
- extract_number
- answer_final

"""

def normalize_text(s: str) -> str:
    """Normalize text for comparison."""
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    
    synonyms = {
        "unchanged": "stay the same",
        "no change": "stay the same",
        "same": "stay the same",
        "second place": "second",
        "2nd": "second",
        "first place": "first",
        "third place": "third",
    }
    return synonyms.get(s, s)


def extract_number(s: str):
    """Extract first number from string."""
    if not s:
        return None
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    return m.group(0) if m else None


def extract_final_answer(text: str) -> str:
    """
    Extract the final answer from model output.
    Looks for patterns like 'Answer: X' or just returns cleaned text.
    """
    if not text:
        return ""
    
    # Look for "Answer:" pattern
    answer_match = re.search(r'(?:answer|final answer|result):\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
    
    # Look for the last line if multi-line
    lines = text.strip().split('\n')
    if len(lines) > 1:
        return lines[-1].strip()
    
    return text.strip()