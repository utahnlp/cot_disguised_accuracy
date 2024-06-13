import os
from typing import Tuple

from datasets import load_dataset
import string
import random

from prompt import construct_mcq_prompt


HF_CACHE_DIR = os.getenv('HF_HOME')
ALL_VALID_OPTIONS = list(string.ascii_uppercase)


def shuffle_choice_order(choices: list, label: str, valid_options: list) -> Tuple[str, list]:
        # make tuples of (bool, choice)
        correct_idx = valid_options.index(label)
        pairs = [(i == correct_idx, c) for i, c in enumerate(choices)]
        
        # shuffle choices
        random.shuffle(pairs)

        # reconstruct new_choices and new_label
        new_choices = []
        for i, (bool_label, choice) in enumerate(pairs):
            new_choices.append(choice)
            if bool_label:
                new_label = valid_options[i]

        return new_choices, new_label


class MCQ:
    def __init__(
            self, 
            idx: int, 
            question: str, 
            choices: list, 
            label: str, 
            valid_options: list, 
            context: str = '',
            model_id: str = None,
            shuffle_strategy: str = 'none'):
        
        assert shuffle_strategy in ['none', 'once', 'twice']
        
        self.idx = idx
        self.question = question
        self.choices = choices
        self.label = label
        self.valid_options = valid_options
        self.context = context
        self.model_id = model_id

        if shuffle_strategy == 'none':
            # cot-choice-order = non-cot-choice-order = original-dataset-order
            self.choices_no_cot, self.label_no_cot = self.choices, self.label
            self.choices_cot, self.label_cot = self.choices, self.label
        if shuffle_strategy == 'once':
            # cot-choice-order = non-cot-choice-order != original-dataset-order
            self.choices_no_cot, self.label_no_cot = shuffle_choice_order(choices, label, self.valid_options)
            self.choices_cot, self.label_cot = self.choices_no_cot, self.label_no_cot
        elif shuffle_strategy == 'twice':
            # cot-choice-order != non-cot-choice-order != original-dataset-order
            self.choices_no_cot, self.label_no_cot = shuffle_choice_order(choices, label, self.valid_options)
            self.choices_cot, self.label_cot = shuffle_choice_order(choices, label, self.valid_options)

        if model_id:
            self._generate_prompts(model_id)


    def _generate_prompts(self, model_id: str):
        self.prompt_no_cot = construct_mcq_prompt(
            model_id=model_id,
            question=self.question, 
            choices='\n'.join([f'({l}) {c}' for l, c in zip(self.valid_options, self.choices_no_cot)]), 
            context=self.context, 
            include_cot=False)
        self.prompt_cot = construct_mcq_prompt(
            model_id=model_id,
            question=self.question, 
            choices='\n'.join([f'({l}) {c}' for l, c in zip(self.valid_options, self.choices_cot)]), 
            context=self.context, 
            include_cot=True)

    
    def to_dict(self) -> dict:
        d = {
            'idx': self.idx,
            'question': self.question,
            'choices': self.choices,
            'label': self.label,
            'valid_options': self.valid_options,
            'no_cot': {
                'choices': self.choices_no_cot,
                'label': self.label_no_cot},
            'cot': {
                'choices': self.choices_cot,
                'label': self.label_cot}    
        }

        if self.model_id:
            d['no_cot']['prompt'] = self.prompt_no_cot
            d['cot']['prompt'] = self.prompt_cot
        if self.context:
            d['context'] = self.context

        return d

class MCQDataset:
    def __init__(
            self, 
            dataset_id: str, 
            model_id: str, 
            sort_by_cot_len: bool = False, 
            shuffle_strategy: str = 'none'):
        
        self.dataset_id = dataset_id
        self.model_id = model_id
        self.shuffle_strategy = shuffle_strategy

        # call corresponding preprocess function for dataset id
        # each function needs to set the following variables
        # self.dataset - using datasets.load_dataset() with appropriate subset and test split
        # self.data = list of MCQ.to_dict() items
  
        if self.dataset_id == 'arc_challenge':
            self._preprocess_arc(subset='ARC-Challenge')
        elif self.dataset_id == 'arc_easy':
            self._preprocess_arc(subset='ARC-Easy')
        elif self.dataset_id == 'aqua':
            self._preprocess_aqua()
        elif self.dataset_id == 'openbook_qa':
            self._preprocess_openbook()
        elif self.dataset_id == 'logiqa':
            self._preprocess_logiqa()        
        elif self.dataset_id == 'truthful_qa':
            self._preprocess_truthful()
        elif self.dataset_id == 'hella_swag':
            self._preprocess_hella_swag()
        elif self.dataset_id == 'mmlu':
            self._preprocess_mmlu()
        else:
            raise ValueError
        
        if sort_by_cot_len:
            self.data.sort(key=lambda mcq: len(mcq['cot']['prompt']))


    def _preprocess_arc(self, subset: str):
        self.dataset = load_dataset('ai2_arc', subset, split='test')
        self.data = []

        for idx, ex in enumerate(self.dataset):
            question = ex['question']
            choices = ex['choices']['text']
            valid_options = ALL_VALID_OPTIONS[:len(choices)]

            label = ex['answerKey']
            if label.isdigit():
                label = valid_options[int(label) - 1] # some examples have label in ['1', '2', '3', '4'] 
            
            mcq = MCQ(
                idx=idx, 
                question=question, 
                choices=choices, 
                label=label, 
                valid_options=valid_options,
                model_id=self.model_id,
                shuffle_strategy=self.shuffle_strategy)
            self.data.append(mcq.to_dict())


    def _preprocess_aqua(self):
        self.dataset = load_dataset('aqua_rat', 'raw', split='test')
        self.data = []

        for idx, ex in enumerate(self.dataset):
            label = ex['correct']
            question = ex['question']
            choices = [x[2:] for x in ex['options']]
            valid_options = ALL_VALID_OPTIONS[:len(choices)]

            mcq = MCQ(
                idx=idx, 
                question=question, 
                choices=choices, 
                label=label, 
                valid_options=valid_options,
                model_id=self.model_id,
                shuffle_strategy=self.shuffle_strategy)
            self.data.append(mcq.to_dict())


    def _preprocess_openbook(self):
        # could combine this with _preprocess_arc - only diff is the column names in the dataset
        self.dataset = load_dataset('openbookqa', 'main', split='test')
        self.data = []

        for idx, ex in enumerate(self.dataset):
            label = ex['answerKey']
            question = ex['question_stem']
            choices = ex['choices']['text']
            valid_options = ALL_VALID_OPTIONS[:len(choices)]

            mcq = MCQ(
                idx=idx, 
                question=question, 
                choices=choices, 
                label=label, 
                valid_options=valid_options,
                model_id=self.model_id,
                shuffle_strategy=self.shuffle_strategy)
            self.data.append(mcq.to_dict())


    def _preprocess_logiqa(self):
        self.dataset = load_dataset('lucasmccabe/logiqa', split='test')
        self.data = []

        for idx, ex in enumerate(self.dataset):
            choices = ex['options']
            valid_options = ALL_VALID_OPTIONS[:len(choices)]
            label = valid_options[ex['correct_option']]
            question = ex['query']
            context = ex['context']

            mcq = MCQ(
                idx=idx, 
                question=question, 
                choices=choices, 
                label=label, 
                valid_options=valid_options,
                context=context,
                model_id=self.model_id,
                shuffle_strategy=self.shuffle_strategy)
            self.data.append(mcq.to_dict())


    def _preprocess_truthful(self):
        self.dataset = load_dataset('truthful_qa', 'multiple_choice', split='validation')
        self.data = []

        for idx, ex in enumerate(self.dataset):
            question = ex['question']
            choices = ex['mc1_targets']['choices']
            valid_options = ALL_VALID_OPTIONS[:len(choices)]
            correct_idx = ex['mc1_targets']['labels'].index(1)
            label = valid_options[correct_idx]

            mcq = MCQ(
                idx=idx, 
                question=question, 
                choices=choices, 
                label=label, 
                valid_options=valid_options,
                model_id=self.model_id,
                shuffle_strategy=self.shuffle_strategy)
            self.data.append(mcq.to_dict())


    def _preprocess_hella_swag(self):
        self.dataset = load_dataset('Rowan/hellaswag', split='validation')
        self.data = []

        for idx, ex in enumerate(self.dataset):
            choices = ex['endings']
            valid_options = ALL_VALID_OPTIONS[:len(choices)]
            label = valid_options[int(ex['label'])]
            question = ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
            context = ex['activity_label']

            mcq = MCQ(
                idx=idx, 
                question=question, 
                choices=choices, 
                label=label, 
                valid_options=valid_options,
                context=context,
                model_id=self.model_id,
                shuffle_strategy=self.shuffle_strategy)
            self.data.append(mcq.to_dict())
            

    def _preprocess_mmlu(self):
        mmlu_subsets = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 
           'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 
           'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 
           'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 
           'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 
           'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 
           'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 
           'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 
           'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 
           'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 
           'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 
           'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 
           'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

        self.data = []
        self.datasets = {}
        idx = 0

        for mmlu_subset in mmlu_subsets:
            dataset = load_dataset('lukaemon/mmlu', mmlu_subset, split='test')
            self.datasets[mmlu_subset] = dataset
            for i, ex in enumerate(dataset):
                if i > 200:
                    break
                
                question = ex['input']
                choices = [ex['A'], ex['B'], ex['C'], ex['D'], ]
                valid_options = ALL_VALID_OPTIONS[:len(choices)]
                label = ex['target']

                mcq = MCQ(
                    idx=idx, 
                    question=question, 
                    choices=choices, 
                    label=label, 
                    valid_options=valid_options,
                    model_id=self.model_id,
                    shuffle_strategy=self.shuffle_strategy)
                self.data.append(mcq.to_dict())
                idx += 1
