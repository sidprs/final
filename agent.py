"""
agent.py
"""

from pathlib import Path
import json
import argparse
from gpu import run_inference

TEST_PATH = Path("json/cse476_final_project_dev_data.json")
INPUT_PATH = Path("json/cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("json/cse_476_final_project_answers.json")


def validate_results(answers):
    # just validate format, not length
    for idx, answer in enumerate(answers):
        if len(answer["output"]) >= 5000:
            raise ValueError(f"output too long at {idx}")
        
        
def compare_with_expected(results, dev_data):
    """using the dev data and test correctness"""
    correct = 0
    total = len(results)
    
    
    for i, (result, dev_item) in enumerate(zip(results, dev_data)):
        generated = result.get("output", "").strip()
        expected = dev_item.get("output", "").strip()
        
        is_correct = generated == expected
        if is_correct:
            correct += 1
        
        if not is_correct or i < 1:  # show first wrong and all incorrect
            print(f"\n Question {i+1}:")
            print(f"   Input: {dev_item.get('input', '')[:100]}...")   #include only 100 
            print(f"   Expected: {expected[:200]}")
            print(f"   Generated: {generated[:200]}")
            print(f"   Match: {' YES ' if is_correct else ' NO '}")
    
    print("\n" + "="*60)
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print("="*60)
    
    return correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=20, help='workers (default: 50)')
    parser.add_argument('--verify', action='store_true', help='enable verification')
    parser.add_argument('--dev', action='store_true', help='use dev data')
    parser.add_argument('--test_file', type=str, default=str(INPUT_PATH), help='path to input JSON file')
    #parser.add_argument('--test', type=int, help='test on first N questions')
    parser.add_argument('--test', type=int, help='test on first N questions')
    
    args = parser.parse_args()
    
    print("cse 476 final project agent")
    
    # load
    print(args)
    
    
    with open(INPUT_PATH, 'r') as f:
        questions = json.load(f)
    
    print(f"loaded {len(questions)} questions")
    
    # run
    run_inference(
        args.test_file,    
        str(OUTPUT_PATH),
        output_csv="answers.csv",
        workers=args.workers,
        verify=args.verify,
        limit=args.test
    )

    # validate format only
    with open(OUTPUT_PATH, 'r') as f:
        saved = json.load(f)
    
    validate_results(saved)
    
    num_processed = args.test if args.test else len(questions)
    print(f"\nvalidated {len(saved)} answers")
    print(f"saved to {OUTPUT_PATH}")
    
    if args.test:
        print(f"\nthis was a test run on {num_processed} questions")
        print(f"to run full dataset: python agent.py --workers 30")


if __name__ == "__main__":
    main()