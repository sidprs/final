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
                    
                #TODO write debug statements to verify if this works   
                    
                return results
def load_test_data(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, 'r') as f:
        return json.load(f)


def write_json_output(results: List[Dict[str, Any]], filepath: str):
    output_only = [{"output": r["output"]} for r in results]
    with open(filepath, 'w') as f:
        json.dump(output_only, f, indent=2)


def write_csv_output(results: List[Dict[str, Any]], filepath: str):
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'output'])
        writer.writeheader()
        writer.writerows(results)


def run_inference(test_file: str, 
                  output_json: str, 
                  output_csv: str = None, 
                  workers: int = 20,
                  verify: bool = False,
                  limit: int = None):
    
    print(f"loading test data from {test_file}")
    questions = load_test_data(test_file)
    
    if limit:
        questions = questions[:limit]
        print(f"limiting to first {limit} questions")
    
    print(f"loaded {len(questions)} questions")
    
    # run agent
    agent = FastAgent(max_workers=workers)
    results = agent.solve_batch(questions, verify=verify)
    
    if not results:
        print("\nno results generated. check errors above.")
        return []
    
    # write json
    write_json_output(results, output_json)
    print(f"\nwrote json to {output_json}")
    
    # write csv
    if output_csv:
        write_csv_output(results, output_csv)
        print(f"wrote csv to {output_csv}")
    
    return results


if __name__ == "__main__":
    import argparse as arg
    
    parser = arg.ArgumentParser()
    parser.add_argument('test_file', nargs='?', default='cse_476_final_project_test_data.json')
    parser.add_argument('--workers', type=int, default=20)
    #parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--limit', type=int)
    
    args = parser.parse_args()
    
    run_inference(
        args.test_file,
        "cse_476_final_project_answers.json",
        "answers.csv",
        workers=args.workers,
        verify=args.verify,
        limit=args.limit
    )