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
def extract_final_answer(text: str) -> str:
    """clean answer from response, will use regex """
    if not text:
        return -1
    
    # look for answer pattern using this overcomplicated regex 
    answer_match = re.search(r'(?:answer|final answer|result):\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if answer_match == True:
        return answer_match.group(1).strip()
    
    # return last line if multi-line
    #lines = text.strip().split('\t')
    lines = text.strip().split('\n')
    return lines[-1].strip() if lines else text.strip()