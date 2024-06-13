import argparse
from collections import Counter
import os

import torch
from tqdm import tqdm

from mcq import MCQ, MCQDataset
from model import LLM
from utils import update_results


def predict(
        model_id: str, 
        task_id: str, 
        out_dir: str, 
        shuffle_strategy: str = 'none', 
        max_examples: int = 500, 
        calibrate: bool = False):

    shuffle_dir = f'shuffle_{shuffle_strategy}'
    calibrate_str = '_calibrated' if calibrate else ''
    out_path = os.path.join(out_dir, shuffle_dir, f'{task_id}_{model_id}{calibrate_str}.json')
    if os.path.exists(out_path):
        raise Exception(f'file already exists, please delete it and run script again\n  file: {out_path}')
    print(f'out_path: {out_path}')

    # load dataset
    print('loading dataset', flush=True)
    dataset = MCQDataset(
        dataset_id=task_id, 
        model_id=model_id, 
        sort_by_cot_len=True, 
        shuffle_strategy=shuffle_strategy)

    # load model
    print('loading model', flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    llm = LLM(model_id=model_id, task_type='mcq', device=device, calibrate=calibrate)

    if calibrate: 
        option_lens = [len(x['valid_options']) for x in dataset.data]
        assert len(Counter(option_lens)) == 1, 'need to have the same number of options for every question to calibrate'
        print('calibrating model', flush=True)

        valid_options = valid_options = dataset.data[0]['valid_options']
        valid_option_ids = [llm.tokenizer.convert_tokens_to_ids(o) for o in valid_options]
        valid_option_vals = ['' for _ in valid_options]

        empty_mcq = MCQ(idx=None, question='N/A', choices=valid_options, label=None, valid_options=valid_option_vals, model_id=model_id)

        llm.init_calibration(
            no_cot_prompt=empty_mcq.prompt_no_cot,
            cot_prompt=empty_mcq.prompt_cot,
            valid_option_ids=valid_option_ids)
        
    if os.path.exists(out_path):
        raise Exception(f'file already exists, please delete it and run script again\n  file: {out_path}')

    # predict on all questions in dataset    
    print('getting predictions', flush=True)
    for i, mcq in tqdm(enumerate(dataset.data), total=len(dataset.data)):
        # preemptively quit after max_examples examples
        if max_examples and i == max_examples:
            exit()

        # get tokens for valid options
        valid_option_ids = [llm.tokenizer.convert_tokens_to_ids(o) for o in mcq['valid_options']]
       
        pred_no_cot = llm.predict_no_cot(prompt=mcq['no_cot']['prompt'], valid_option_ids=valid_option_ids)
        pred_with_cot = llm.predict_cot(prompt=mcq['cot']['prompt'], valid_option_ids=valid_option_ids)

        mcq['no_cot']['pred'] = pred_no_cot['pred']
        mcq['cot']['pred'] = pred_with_cot['pred']
        mcq['cot']['output'] = pred_with_cot['output']
        mcq['cot']['output_truncated'] = pred_with_cot['truncated']
        update_results(mcq, out_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_id', '-m', 
        type=str, 
        choices=[
            'llama-2-7b', 'llama-2-13b', 'llama-2-70b', 
            'flan-t5-small', 'flan-t5-base', 'flan-t5-large', 'flan-t5-xl', 'flan-t5-xxl', 'flan-ul2',
            'pythia-70m-dpo', 'pythia-160m-dpo', 'pythia-410m-dpo', 'pythia-1b-dpo', 'pythia-1.4b-dpo','pythia-2.8b-dpo',
        ],
        help='Which model to use (choice: %(choices)s)')
    parser.add_argument(
        '--task_id', '-t', 
        type=str, 
        choices=['aqua', 'arc_easy', 'arc_challenge', 'openbook_qa', 'logiqa', 'truthful_qa', 'hella_swag', 'mmlu'], 
        help='Which task to use (choice: %(choices)s)')
    parser.add_argument('--out_dir', '-o', type=str, help='Directory where output will be written')
    parser.add_argument(
        '--shuffle_strategy', 
        type=str, 
        default='none', 
        choices=['none', 'once', 'twice'],
        help='How to shuffle the dataset. \
            "none" does nothing. \
            "once" randomizes the order w.r.t. the dataset, but the CoT/no-CoT conditions see the same ordering. \
            "twice" randomizes the order s.t. the CoT/no-CoT conditions see a different ordering')
    parser.add_argument('--max_examples', type=int, default=500)
    parser.add_argument('--calibrate', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    predict(
        model_id=args.model_id, 
        task_id=args.task_id, 
        out_dir=args.out_dir, 
        shuffle_strategy=args.shuffle_strategy,
        max_examples=args.max_examples,
        calibrate=args.calibrate)
