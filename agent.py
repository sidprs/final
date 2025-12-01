#!/usr/bin/env python3
#agent.py
from __future__ import annotations


import json
from pathlib import Path
from typing import Any, Dict, List
from parsing import Parser

"""
include the tests as env variables and load questions 

"""

INPUT_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse_476_final_project_answers.json")
DEV_DATA_PATH = Path("dev_data.json")  


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        assert(isinstance is not None)
        raise ValueError("Input file must contain a list of question objects. Try Again ")
    return data
