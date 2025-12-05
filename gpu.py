# gpu.py

"""
fast parallel inference with debugging
"""

import json
import time
import csv
import re
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from utility import call_model_chat_completions, extract_final_answer


class FutureAgent:
    def __init__(self, max_workers: int = 20):
        self.max_workers = max_workers
        self.call_count = 0
        self.error_count = 0
        self.use_cot = False # new chain of thought implementation, turned off for testing
        self.cot_method = "zero shot"  
        self.algorithm_stats = {"direct": 0, "cot": 0, "self_consistency": 0}
 
    def reset(self)->None:
        # resets the counter per run 
        self.call_count = 0
        self.error_count = 0
    
    def solve_direct(self, question: str, domain: str = "unknown") -> tuple[str,str]:
        # returns (answer, error_msg)
        # i think that some output is not complete so update the prompt
        self.algorithm_stats["direct"] += 1
        system = "final answer only. dont give a explanation but make the output be complete."
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
    
    def solve_cot(self, question: str, domain: str) -> tuple:
        """
        Zero Shot COT: Chain of Thought
        Simple implementation using chain 
        """
        system = "You are the greatest assistant of all time . Think step by step and provide the final answer."
        prompt = f"""{question} Let's think step by step."""
        response = call_model_chat_completions(
            prompt=prompt,
            system=system,
            temperature=0.0,
            timeout=45
        )
        self.call_count += 1
        if response["ok"]:
            answer = self._extract_from_reasoning(response["text"])
            return (answer, None)
        else:
            self.error_count += 1
            return ("", response["error"])
        
    def solve_self_const(self, question:str,domain: str = "unknown", n_samples: int = 3) -> tuple[str, str]:
        self.algorithm_stats["self_consistency"] += 1
        
        system = "You are the greatest assistant of all time. Provide a clear, concise answer."
        prompt = f"{question}"
        
        answers = []
        for i in range(n_samples):
            response = call_model_chat_completions(
                prompt=prompt,
                system=system,
                temperature=0.7,  
                timeout=30
            )
            self.call_count += 1
            
            if response["ok"]:
                answer = extract_final_answer(response["text"])
                answers.append(answer)
        
        if not answers:
            self.error_count += 1
            return ("", "All attempts failed")
        
        final_answer = self._majority_vote(answers)
        return (final_answer, None)
    
    def _extract_from_reasoning(self, text: str) -> str:
        """Extract final answer from chain of thought reasoning"""
        # Look for common conclusion patterns
        patterns = [
            r'(?:therefore|thus|so|final answer|answer)[:,]?\s*(.+?)(?:\n|$)',
            r'(?:the answer is)[:,]?\s*(.+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        return lines[-1] if lines else text.strip()
    
    def _majority_vote(self, answers: List[str]) -> str:
        if not answers:
            return ""
        
        normalized = {}
        for ans in answers:
            norm = ans.strip().lower()
            if norm not in normalized:
                normalized[norm] = ans  
        
        from collections import Counter
        counts = Counter(ans.strip().lower() for ans in answers)
        most_common = counts.most_common(1)[0][0]
        
        return normalized.get(most_common, answers[0])
    
    def select_algorithm(self, question: str, domain: str) -> str:
        """Choose which algorithm to use based on question """
        q_lower = question.lower()
        if any(keyword in q_lower for keyword in ['why', 'explain', 'how', 'calculate', 'prove']):
            return "cot"
        if any(keyword in q_lower for keyword in ['estimate', 'approximately', 'which', 'best', 'optimal']):
            return "self_consistency"
        return "direct"
    
    def solve_single_adaptive(self, question: str, domain: str = "unknown") -> tuple[str, str]:
        """ choose algorithm based on question type"""
        algorithm = self.select_algorithm(question, domain)
        
        if algorithm == "cot":
            return self.solve_cot(question, domain)
        elif algorithm == "self_consistency":
            return self.solve_self_const(question, domain, n_samples=3)
        else:
            return self.solve_direct(question, domain)
    
        
    def solve_batch(self, questions: List[Dict[str, Any]], verify: bool = False, use_adaptive: bool = True) -> List[Dict[str, Any]]:
        total = len(questions)
        results = [None] * total
        errors = []
        byproduct = [] 
        print(f"\nprocessing {total} questions with {self.max_workers} workers")
        
        # test first question synchronously
        if total > 0:
            print("\ntesting first question...")
            test_q = questions[0]
            if use_adaptive:
                test_answer, test_error = self.solve_single_adaptive(
                    test_q.get("input", ""),
                    test_q.get("domain", "unknown")
                )
            else:
                test_answer, test_error = self.solve_direct(
                    test_q.get("input", ""),
                    test_q.get("domain", "unknown")
                )
                
            if test_error:
                print(f"ERROR on test question: {test_error}")
                print(f"question was: {test_q.get('input', '')[:100]}")
                print("\ncheck your API_KEY settings")
                return -1
            else:
                print(f"test successful. answer: {test_answer}")
        
        start_time = time.time()
        #start_time = t.time()
        last_print = start_time
        
        #using threadpool instead of other metal core inference
        with ThreadPoolExecutor(max_workers=self.max_workers) as exec:
            """
            using threads to launch parallel threads (futures)
            these futures will return async 
            TODO: determine if deadlocking is causing increase in error rates
            """
            future_to_idx = {}
            for idx, q in enumerate(questions):
                if use_adaptive:
                    future = exec.submit(
                        self.solve_single_adaptive,
                        q.get("input", ""),
                        q.get("domain", "unknown")
                    )
                else:
                    future = exec.submit(
                        self.solve_direct,
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
                    
                    instant = time.time()
                    if completed % 100 == 0 or (instant - last_print) >= 5:
                        elapsed = instant - start_time
                        print(f" [{elapsed / 100 }]")
                        rate = (completed / elapsed) if elapsed > 0 else 0
                        remaining = (total - completed) / rate if rate > 0 else 0
                        pct = (completed / total) * 100
                        print(f"  [{pct:5.1f}%] {completed:5d}/{total} | {rate:4.2f} q/s | errors: {self.error_count}")
                        last_print = instant
                        
                except Exception as e:
                    errors.append((idx, str(e)))
                    results[idx] = {
                        "id": questions[idx].get("id", f"q_{idx}"),
                        "output": ""
                    }
                    
            elapsed = time.time() - start_time
            #print out some information  
            print(f"\ncompleted {total} questions in {elapsed:.1f}s ({elapsed/60:.1f}m)")
            print(f"total api calls: {self.call_count}")
            print(f"errors: {self.error_count}/{total}")
            print(f"total API calls: {self.call_count}")
            
            if errors and len(errors) <= 5:
                print("\nfirst few errors:")
                for idx, err in errors[:5]:
                    print(f"  q_{idx}: {err} was found")
            
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


def run_inference(test_file: str, output_json: str, 
output_csv: str = None, workers: int = 20,
                  verify: bool = False, limit: int = None, use_adaptive: bool = True):
    # possible idle infinite loop so will update limit bounds
     
    print(f"loading test data from {test_file}")
    questions = load_test_data(test_file)
    
    if limit:
        questions = questions[:limit]
        print(f"limiting to first {limit} questions")
    
    print(f"loaded {len(questions)} questions")
    
    # run agent
    agent = FutureAgent(max_workers=workers)
    results = agent.solve_batch(questions, verify=verify)
    
    if not results:
        print("\nno results generated. check errors above.")
        return []
    
    # write JSON output
    write_json_output(results, output_json)
    print(f"\nwrote json to {output_json}")

    # write CSV output if requested
    if output_csv:
        write_csv_output(results, output_csv)
        print(f"wrote csv to {output_csv}")

    
    return results


if __name__ == "__main__":
    import argparse as arg
    
    parser = arg.ArgumentParser()
    parser.add_argument('test_file', nargs='?', default='json/cse_476_final_project_test_data.json')
    parser.add_argument('--workers', type=int, default=20)
    # uncomment this line if your GPU is beefier 
    #parser.add_argument('--workers', type=int, default=50)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--adaptive', action='store_true', default=True )
    parser.add_argument('--algorithm', choices=['direct', 'cot', 'self_consistency'])

    args = parser.parse_args()
    
    run_inference(
        args.test_file,
        "cse_476_final_project_answers.json",
        "test_runs/answers.csv",
        workers=args.workers,
        verify=args.verify,
        limit=args.limit,
        use_adaptive=args.adaptive and not args.algorithm
    )
