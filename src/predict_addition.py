import argparse
import os

import torch
from tqdm import tqdm

from addition import AdditionDataset
from model import LLM
from utils import update_results


def predict(model_id: str, n: int, digits: int, operands: int, out_dir: str, use_precomputed: bool = False, data_dir: str = None):

    precomputed_path = 'precomputed' if use_precomputed else 'random'
    out_path = os.path.join(out_dir, precomputed_path, f'n{n}_d{digits}_o{operands}_{model_id}.json')
    if os.path.exists(out_path):
        raise Exception(f'file already exists, please delete it and run script again\n  file: {out_path}')
    print(f'out_path: {out_path}')

    # load dataset
    print('loading dataset', flush=True)
    if use_precomputed:
        data_path = os.path.join(data_dir, f'n{n}_d{digits}_o{operands}.json')
        print(f'  from precomputed: {data_path}')
        dataset = AdditionDataset.from_file(data_path, model_id=model_id)
    else:
        print(f'  from random')
        dataset = AdditionDataset.from_random(n=n, digits=digits, operands=operands, model_id=model_id)

    # load model
    print('loading model', flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    llm = LLM(model_id, 'addition', device)

    # predict on all questions in dataset    
    print('getting predictions', flush=True)
    for q in tqdm(dataset.data, total=len(dataset.data)):
       
        pred_no_cot = llm.predict_no_cot(prompt=q['no_cot']['prompt'])
        pred_with_cot = llm.predict_cot(prompt=q['cot']['prompt'])

        q['no_cot']['pred'] = pred_no_cot['pred']
        q['no_cot']['output'] = pred_no_cot['output']
        q['cot']['pred'] = pred_with_cot['pred']
        q['cot']['output'] = pred_with_cot['output']
        update_results(q, out_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_id', '-m', 
        type=str, 
        choices=[
            'llama-2-7b', 'llama-2-13b', 'llama-2-70b', 
            'flan-t5-small', 'flan-t5-base', 'flan-t5-large', 'flan-t5-xl', 'flan-t5-xxl',
            'pythia-70m-dpo', 'pythia-160m-dpo', 'pythia-410m-dpo', 'pythia-1b-dpo', 'pythia-1.4b-dpo','pythia-2.8b-dpo',
        ],
        help='Which model to use (choice: %(choices)s)')
    parser.add_argument(
        '--num_examples',
        type=int, 
        default=1000, 
        help='How many addition problems to evaluate on.')
    parser.add_argument(
        '--digits',
        type=int,
        choices=[2, 3],
        help='How many digits per number in each addition problem.')
    parser.add_argument('--operands',
        type=int,
        choices=[2, 4, 8, 16],
        help='How many operands per addition problem.')
    parser.add_argument('--out_dir', '-o', type=str, help='Directory where output will be written')
    parser.add_argument('--use_precomputed', action='store_true', help='Use a precomputed dataset')
    parser.add_argument('--use_random', action='store_false', dest='use_precomputed', help='Use a randomly generated dataset')
    parser.add_argument('--data_dir', '-d', type=str, help='Directory where precomputed data is stored')
    args = parser.parse_args()

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    predict(
        model_id=args.model_id, 
        n=args.num_examples, 
        digits=args.digits, 
        operands=args.operands, 
        out_dir=args.out_dir,
        use_precomputed=args.use_precomputed,
        data_dir=args.data_dir)
