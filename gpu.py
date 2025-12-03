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

    def solve_batch(self, questions: List[Dict[str, Any]], verify: bool = False) -> List[Dict[str, Any]]:
        total = len(questions)
        results = [None] * total
        errors = []
        
        print(f"\nprocessing {total} questions with {self.max_workers} workers")
        
        # test first question synchronously
        if total > 0:
            print("\ntesting first question...")
            test_q = questions[0]
            test_answer, test_error = self.solve_single_fast(
                test_q.get("input", ""),
                test_q.get("domain", "unknown")
            )
            
            if test_error:
                print(f"ERROR on test question: {test_error}")
                print(f"question was: {test_q.get('input', '')[:100]}")
                print("\ncheck your API_BASE and API_KEY settings")
                return []
            else:
                print(f"test successful. answer: {test_answer}")
        
        start_time = time.time()
        #start_time = t.time()
        last_print = start_time
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {}
            for idx, q in enumerate(questions):
                future = executor.submit(
                    self.solve_single_fast,
                    q.get("input", ""),
                    q.get("domain", "unknown")
                )
                future_to_idx[future] = idx
            
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    answer, error = future.result()
                    
                    if error:
                        errors.append((idx, error))
                    
                    results[idx] = {
                        "id": questions[idx].get("id", f"q_{idx}"),
                        "output": answer
                    }
                    completed += 1
                    
                    now = time.time()
                    if completed % 100 == 0 or (now - last_print) >= 5:
                        elapsed = now - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        remaining = (total - completed) / rate if rate > 0 else 0
                        pct = (completed / total) * 100
                        print(f"  [{pct:5.1f}%] {completed:5d}/{total} | {rate:4.1f} q/s | eta {remaining:5.1f}s | errors: {self.error_count}")
                        last_print = now
                        
                except Exception as e:
                    errors.append((idx, str(e)))
                    results[idx] = {
                        "id": questions[idx].get("id", f"q_{idx}"),
                        "output": ""
                    }