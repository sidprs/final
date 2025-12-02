# agent.py

from pathlib import Path
import json
import argparse
from gpu import run_inference


INPUT_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse_476_final_project_answers.json")


def validate_results(answers):
    # just validate format, not length
    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"missing output at {idx}")
        if not isinstance(answer["output"], str):
            raise TypeError(f"non-string output at {idx}")
        if len(answer["output"]) >= 5000:
            raise ValueError(f"output too long at {idx}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=20, help='workers (default: 20)')
    parser.add_argument('--verify', action='store_true', help='enable verification')
    #parser.add_argument('--test', type=int, help='test on first N questions')
    parser.add_argument('--test', type=int, help='test on first N questions')
    
    args = parser.parse_args()
    
    print("cse 476 final project agent")
    
    # load
    with open(INPUT_PATH, 'r') as f:
        questions = json.load(f)
    
    print(f"loaded {len(questions)} questions")
    
    # run
    results = run_inference(
        str(INPUT_PATH),
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