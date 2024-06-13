
MCQ_SYSTEM_PROMPT = 'Answer the question directly. Respond with the letter of the correct response.'
MCQ_SYSTEM_PROMPT_COT = "When answering a question, explain your reasoning so it's clear how you arrived at the answer."
MCQ_ANSWER_EXTRACTION_PROMPT = 'The correct answer is ('

ADD_SYSTEM_PROMPT = 'Respond only with a number in <answer></answer> XML tags, like this <answer>#</answer>.'
ADD_SYSTEM_PROMPT_COT_1 = "When answering a question, show your calculations so it's clear how you arrived at the answer."
ADD_SYSTEM_PROMPT_COT_2 = 'Put your final answer after your calculations in <answer></answer> XML tags, like this <answer>#</answer>.'
ADD_ANSWER_EXTRACTION_PROMPT = '<answer>'

COT_PROMPT = 'Let\'s think step by step.'


def construct_mcq_prompt(model_id: str, question: str, choices: str, context: str = '', include_cot: bool = False) -> str:
    # This function uses the question, choices, and (optional) context components 
    # to construct a multiple-choice prompt suited to the model being used

    context = context + '\n' if context else ''
    if 'llama-2' in model_id or 'tulu-v2' in model_id:
        if include_cot:
            system_prompt = f'<<SYS>>\n{MCQ_SYSTEM_PROMPT_COT}\n<</SYS>>\n\n'
            return f'<s>[INST] {system_prompt}{context}{question}\n\n{choices}\n\n{COT_PROMPT} [/INST]'
        else:
            system_prompt = f'<<SYS>>\n{MCQ_SYSTEM_PROMPT}\n<</SYS>>\n\n'
            return f'<s>[INST] {system_prompt}{context}{question}\n\n{choices} [/INST]{MCQ_ANSWER_EXTRACTION_PROMPT}'
    elif 'mistral' in model_id:
        if include_cot:
            system_prompt = f'{MCQ_SYSTEM_PROMPT_COT}\n'
            return f'<s>[INST] {system_prompt}{context}{question}\n\n{choices}\n\n{COT_PROMPT} [/INST]'
        else:
            system_prompt = f'{MCQ_SYSTEM_PROMPT}\n'
            return f'<s>[INST] {system_prompt}{context}{question}\n\n{choices} [/INST]{MCQ_ANSWER_EXTRACTION_PROMPT}'
    elif 'flan' in model_id:
        if include_cot:
            return f'{context}{question}\n\n{choices}\n\n{COT_PROMPT}'
        else:
            return f'{context}{question}\n\n{choices}'
    elif 'mpt' in model_id:
        if include_cot:
            system_prompt = f'{MCQ_SYSTEM_PROMPT_COT}\n'
            return f'{system_prompt}### Instruction: {context}{question}\n\n{choices}\n\n{COT_PROMPT}\n### Response: '
        else:
            system_prompt = f'{MCQ_SYSTEM_PROMPT}\n'
            return f'{system_prompt}### Instruction: {context}{question}\n\n{choices}\n### Response: {MCQ_ANSWER_EXTRACTION_PROMPT}'
    elif 'pythia' in model_id:
        if include_cot:
            system_prompt = f'{MCQ_SYSTEM_PROMPT_COT}\n'
            return f'{system_prompt}### Instruction: {context}{question}\n\n{choices}\n\n{COT_PROMPT}\n### Assistant: '
        else:
            system_prompt = f'{MCQ_SYSTEM_PROMPT}\n'
            return f'{system_prompt}### Instruction: {context}{question}\n\n{choices}\n### Assistant: {MCQ_ANSWER_EXTRACTION_PROMPT}'
    else:
        raise ValueError(f'model_id: "{model_id}" not supported')


def construct_addition_prompt(model_id: str, numbers: list, include_cot: bool = False) -> str:

    numbers_str = '+'.join([str(x) for x in numbers])
    question = f'What is {numbers_str}?'

    if 'llama-2' in model_id or 'tulu-v2' in model_id:
        if include_cot:
            system_prompt = f'<<SYS>>\n{ADD_SYSTEM_PROMPT_COT_1} {ADD_SYSTEM_PROMPT_COT_2}\n<</SYS>>\n\n'
            return f'<s>[INST] {system_prompt}{question} {ADD_SYSTEM_PROMPT_COT_2} {COT_PROMPT} [/INST]'
        else:
            system_prompt = f'<<SYS>>\n{ADD_SYSTEM_PROMPT}\n<</SYS>>\n\n'
            return f'<s>[INST] {system_prompt}{question} {ADD_SYSTEM_PROMPT} [/INST] {ADD_ANSWER_EXTRACTION_PROMPT}'
    elif 'flan' in model_id:
        if include_cot:
            return f'{question} {ADD_SYSTEM_PROMPT_COT_2} {COT_PROMPT}'
        else:
            return f'{question} {ADD_SYSTEM_PROMPT}'
    elif 'pythia' in model_id:
        if include_cot:
            system_prompt = f'{ADD_SYSTEM_PROMPT_COT_1} {ADD_SYSTEM_PROMPT_COT_2}\n'
            return f'{system_prompt}### Instruction: {question} {ADD_SYSTEM_PROMPT_COT_2} {COT_PROMPT}\n### Assistant: '
        else:
            system_prompt = f'{ADD_SYSTEM_PROMPT}\n'
            return f'{system_prompt}### Instruction: {question} {ADD_SYSTEM_PROMPT}\n### Assistant: {ADD_ANSWER_EXTRACTION_PROMPT}'
    else:
        raise ValueError(f'model_id: "{model_id}" not supported')
