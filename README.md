# CSE 476 Project

## Usage
# Quick Test (5 questions)


python agent.py --test 5
# Medium Test (50 questions)
python agent.py --test 50
# Full Run (all questions)
python agent.py --workers 30
# Debug CoT (Chain of Thought)
python gpu.py json/cse_476_final_project_test_data.json --algorithm cot --limit 10
# Check Results
head json/cse_476_final_project_answers.json


# 1. generate answers for test data
python agent.py

# 2. check the output file
less cse_476_final_project_answers.json


# Conclusion 

This project using GPU threading to run "futures" which calls the API in parallel threads.
This resulted in a faster runtime 

For better accuracy / results from the LLM API provided, I created a comprehensive system prompt and user prompt
I have also created a few shot prompt (which is disabled due to bugs that reduces runtime)
which increases the accuracy of the program

My error rate has a bug and does not accurately detect incorrect output. I was trying to debug but couldnt find it out


