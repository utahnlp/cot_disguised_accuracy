import json
import os

def update_results(new_result: dict, results_file: str):

    # load results
    results = []
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
    
    # update results
    results.append(new_result)

    # write results
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
