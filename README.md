# CSE 476 Project


# Project Title

A brief description of what this project does and who it's for


## Demo

Insert gif or link to demo


## Optimizations


This project using GPU threading to run "futures" which calls the API in parallel threads.
This resulted in a faster runtime 

For better accuracy / results from the LLM API provided, I created a comprehensive system prompt and user prompt
I have also created a few shot prompt (which is disabled due to bugs that reduces runtime)
which increases the accuracy of the program

My error rate has a bug and does not accurately detect incorrect output. I was trying to debug but couldnt find it out


## Running Tests

To run tests, run the following command

```bash
  python agent.py --test 5
  python agent.py --test 50
  python agent.py --workers 30
  python gpu.py json/cse_476_final_project_test_data.json --algorithm cot --limit 10
  head json/cse_476_final_project_answers.json
```




