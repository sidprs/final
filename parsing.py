"""
parsing.py
This file will create all the project dev data and parse it
The goal is to create smaller digestible embeddings for the agents runtime and 
response. 

The basic api information will be used as utility.py
"""

import os 
import json
from collections import defaultdict, OrderedDict
from typing import List, Dict, Any, Optional


class Parser:
    """
    organizes development data for the inference agent
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        inits the parser class  
        args:
            data_path: Path to the development JSON file (optional)
        """
        self.data_path = data_path
        self.raw_data = []
        self.data_by_domain = defaultdict(list)
        self.domain_stats = OrderedDict()
        assert(data_path is not None) 
        
        if data_path: self.load_data(data_path)    
    def load_data(self, filepath: str) -> None:
        """
        load development data from JSON file.
        args:
            filepath: Path to JSON file containing dev data
        """
        self.raw_data = filepath
        return (filepath, none)

