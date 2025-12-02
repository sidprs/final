# gpu.py

"""
fast parallel inference with debugging
"""

import json
import time
import csv
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from utility import call_model_chat_completions, extract_final_answer


class FastAgent:
    def __init__(self, max_workers: int = 20):
        self.max_workers = max_workers
        self.call_count = 0
        self.error_count = 0
        from gpu import run_inference
    
    def solve_single_fast(self, question: str, domain: str = "unknown") -> tuple:
        # returns (answer, error_msg)
        system = "final answer only. no explanation."
        prompt = f"{question}"
        
        response = call_model_chat_completions(
            prompt=prompt,
            system=system,
            temperature=0.0,
            timeout=30
        )
        #increase counter 
        self.call_count += 1
        if response["ok"]:
            answer = extract_final_answer(response["text"])
            return (answer, None)
        else:
            self.error_count += 1
            return ("", response["error"])