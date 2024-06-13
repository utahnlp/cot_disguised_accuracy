import json
import os
import random

from prompt import construct_addition_prompt


class AdditionQuestion:
    def __init__(
            self, 
            idx: int, 
            numbers: list, 
            answer: int, 
            model_id: str = None, 
            prompt_no_cot: str = None, 
            prompt_cot: str = None):
        
        self.idx = idx
        self.numbers = numbers
        self.answer = answer
        self.model_id = model_id

        if prompt_no_cot and prompt_cot:
            self.prompt_no_cot = prompt_no_cot
            self.prompt_cot = prompt_cot
        elif model_id:
            self.prompt_no_cot = construct_addition_prompt(model_id, self.numbers, include_cot=False)
            self.prompt_cot = construct_addition_prompt(model_id, self.numbers, include_cot=True)

    @classmethod
    def from_random(cls, idx: int, digits: int, operands: int, model_id: str = None):
        assert digits in [2, 3]
        assert operands in [2, 4, 8, 16]

        number_range = (10, 99) if digits == 2 else (100, 999)
        numbers = [random.randint(*number_range) for _ in range(operands)]
        answer = sum(numbers)

        return cls(idx, numbers, answer, model_id)
    
    @classmethod
    def from_dict(cls, d: dict, model_id: str):
        return cls(
            idx=d['idx'],
            numbers=d['numbers'],
            answer=d['answer'],
            model_id=d.get('model_id', model_id),
            prompt_no_cot=d.get('no_cot', {}).get('prompt'),
            prompt_cot=d.get('cot', {}).get('prompt'))


    def to_dict(self) -> dict:
        d = {
            'idx': self.idx,
            'numbers': self.numbers,
            'answer': self.answer}

        if self.model_id:
            d['no_cot'] = {'prompt': self.prompt_no_cot}
            d['cot'] = {'prompt': self.prompt_cot}

        return d


class AdditionDataset:
    def __init__(self, data: list):
        self.data = data
        
    @classmethod
    def from_random(cls, n: int, digits: int, operands: int, model_id: str = None):
        data = [
            AdditionQuestion.from_random(i, digits, operands, model_id).to_dict()
            for i in range(n)]
        return cls(data)

    @classmethod
    def from_file(cls, file_path: str, model_id: str):
        with open(file_path) as f:
            raw_data = json.load(f)

        data = []
        for row in raw_data:
            # an overly complicated way of getting AdditionQuestion to fill in the prompts for model_id
            data.append(AdditionQuestion.from_dict(row, model_id=model_id).to_dict())

        return cls(data)


def create_dataset(out_dir: str, n: int, digits: int, operands: int):
    out_file = os.path.join(out_dir, f'n{n}_d{digits}_o{operands}.json')
    if os.path.exists(out_file):
        raise Exception(f'file already exists, please delete it and run script again\n  file: {out_file}')
    
    dataset = AdditionDataset.from_random(n, digits, operands)

    with open(out_file, 'w') as f:
        json.dump(dataset.data, f, indent=2)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        '--out_dir', '-o', 
        type=str, 
        help='Directory where datasets will be stored')
    parser.add_argument(
        '--num_examples',
        type=int, 
        default=100, 
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
    args = parser.parse_args()

    create_dataset(out_dir=args.out_dir, n=args.num_examples, digits=args.digits, operands=args.operands)
