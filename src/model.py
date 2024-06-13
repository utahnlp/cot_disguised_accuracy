import os

import torch
import transformers
from transformers import LlamaForCausalLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from prompt import MCQ_ANSWER_EXTRACTION_PROMPT, ADD_ANSWER_EXTRACTION_PROMPT


MODEL_ARGS = {
    'temperature': 0.8,
    'top_p': 0.95
}

MODEL_ID_MAPPING = {
    'llama-2-7b': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama-2-13b': 'meta-llama/Llama-2-13b-chat-hf', 
    'llama-2-70b': 'meta-llama/Llama-2-70b-chat-hf',
    'flan-t5-small': 'google/flan-t5-small',
    'flan-t5-base': 'google/flan-t5-base',
    'flan-t5-large': 'google/flan-t5-large',
    'flan-t5-xl': 'google/flan-t5-xl',
    'flan-t5-xxl': 'google/flan-t5-xxl',
    'flan-ul2': 'google/flan-ul2',
    'pythia-70m-dpo': 'lomahony/pythia-70m-helpful-dpo',
    'pythia-160m-dpo': 'lomahony/pythia-160m-helpful-dpo',
    'pythia-410m-dpo': 'lomahony/pythia-410m-helpful-dpo',
    'pythia-1b-dpo': 'lomahony/pythia-1b-helpful-dpo',
    'pythia-1.4b-dpo': 'lomahony/pythia-1.4b-helpful-dpo',
    'pythia-2.8b-dpo': 'lomahony/pythia-2.8b-helpful-dpo',
}

TOTAL_ALLOWABLE_TOKENS = 700

HF_CACHE_DIR = os.getenv('HF_HOME')


class LLM:
    def __init__(self, model_id: str, task_type: str, device: str, calibrate: bool = False):
        assert task_type in ['mcq', 'addition']

        self.model_id = model_id
        self.device = device
        hf_model_path = MODEL_ID_MAPPING[model_id]
        self.task_type = task_type

        assert not (task_type == 'addition' and calibrate), 'calibration is only implemented for mcq'
        self.calibrate = calibrate
        self.calibration_params = {}

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_path,add_bos_token=False)

        # load model
        if 'llama-2' in model_id:
            self.model_type = 'decoder'
            if '70b' in model_id:
                nf4_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16)
                self.model = LlamaForCausalLM.from_pretrained(
                    hf_model_path,
                    torch_dtype=torch.bfloat16,
                    device_map='balanced',
                    quantization_config=nf4_config,
                    use_flash_attention_2=True,
                    cache_dir=HF_CACHE_DIR)
            else:
                self.model = LlamaForCausalLM.from_pretrained(
                    hf_model_path,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    use_flash_attention_2=True,
                    cache_dir=HF_CACHE_DIR)
            self.max_length = self.model.config.max_position_embeddings
        elif 'flan-t5' in model_id:
            self.model_type = 'encoder-decoder'
            self.model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_path).to(device)
            self.max_length = 512
        elif 'flan-ul2' in model_id:
            self.model_type = 'encoder-decoder'
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                hf_model_path, 
                device_map='auto', 
                load_in_8bit=True)
            self.max_length = 512
        elif 'pythia' in model_id:
            self.model_type = 'decoder'
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                cache_dir=HF_CACHE_DIR)
            self.max_length = self.model.config.max_position_embeddings
        else:
            raise ValueError(f'model_id: "{model_id}" not supported')
        
        # prepare answer-extraction prompt
        if task_type == 'mcq':
            self.answer_extraction_prompt = self.tokenizer(
                MCQ_ANSWER_EXTRACTION_PROMPT, 
                truncation=True, 
                return_tensors='pt')['input_ids'].to(device)
        else:
            self.answer_extraction_prompt = self.tokenizer(
                ADD_ANSWER_EXTRACTION_PROMPT, 
                truncation=True, 
                return_tensors='pt')['input_ids'].to(device)

    
    def init_calibration(self, no_cot_prompt: str, cot_prompt: str, valid_option_ids: list):
        assert 'pythia' in self.model_id

        self.calibration_params = {}
        
        # no cot
        prompt_token_ids = self.tokenizer(
            [no_cot_prompt], 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt')['input_ids']
        out = self.model(prompt_token_ids.to(self.device))
        probs = out.logits[0, -1, valid_option_ids].softmax(0)
        W = torch.diag(1/probs)
        self.calibration_params['no_cot'] = W

        # cot
        prompt_token_ids = self.tokenizer(
            [cot_prompt], 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt')['input_ids']
        out_tok_ids_w_answer_extraction = torch.concat([prompt_token_ids[0].to(self.device), self.answer_extraction_prompt[0]])
        out = self.model(out_tok_ids_w_answer_extraction.unsqueeze(0))
        probs = out.logits[0, -1, valid_option_ids].softmax(0)
        W = torch.diag(1/probs)
        self.calibration_params['cot'] = W
            
    
    def predict_no_cot(self, **xargs):
        if self.task_type == 'mcq':
            return self._predict_mcq_no_cot(**xargs)
        elif self.task_type == 'addition':
            return self._predict_addition(**xargs)
        

    def predict_cot(self, **xargs):
        if self.task_type == 'mcq':
            return self._predict_mcq_cot(**xargs)
        elif self.task_type == 'addition':
            return self._predict_addition(**xargs)


    def _predict_mcq_no_cot(self, prompt: str, valid_option_ids: list) -> dict:        
        # generate response without CoT

        # tokenize prompt
        prompt_token_ids = self.tokenizer(
            [prompt], 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt')['input_ids']
        
        if self.model_type == 'encoder-decoder':
            # encoder-decoder model
            decoder_input_ids = self.model._shift_right(self.answer_extraction_prompt)
            out = self.model(
                prompt_token_ids.to(self.device), 
                decoder_input_ids=decoder_input_ids.to(self.device))
            pred_idx = torch.argmax(out.logits[0, -1, valid_option_ids]).cpu()
            pred = self.tokenizer.convert_ids_to_tokens(valid_option_ids[pred_idx])

        else:
            # decoder-only model
            out = self.model(prompt_token_ids.to(self.device))
            if self.calibrate:
                assert self.calibration_params, 'need to run init_calibration() before predicting with self.calibrate=True'
                W = self.calibration_params['no_cot']
                probs = out.logits[0, -1, valid_option_ids].softmax(0)
                pred_idx = torch.argmax(W @ probs).cpu()
            else:
                pred_idx = torch.argmax(out.logits[0, -1, valid_option_ids]).cpu()
            pred = self.tokenizer.convert_ids_to_tokens(valid_option_ids[pred_idx])

        return {'pred': pred}


    def _predict_mcq_cot(self, prompt: str, valid_option_ids: list) -> dict:
        # generate response with CoT

        # tokenize prompt
        prompt_token_ids = self.tokenizer(
            [prompt], 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt')['input_ids']
        
        # truncate generation so that we fit on gpu (if needed) for 70b Llama 2 model
        if self.model_id in ['llama-2-70b', 'flan-ul2']:
            max_tokens_to_generate = TOTAL_ALLOWABLE_TOKENS - prompt_token_ids.shape[1]

            out_tok_ids = self.model.generate(
                prompt_token_ids.to(self.device), 
                max_new_tokens=max_tokens_to_generate, 
                do_sample=True,
                **MODEL_ARGS)[0]
        else:
            out_tok_ids = self.model.generate(
                prompt_token_ids.to(self.device), 
                max_new_tokens=1000,
                do_sample=True,
                **MODEL_ARGS)[0]
            
        output = self.tokenizer.decode(out_tok_ids.cpu())
        truncated = False
        if self.model_type == 'encoder-decoder':
            # encoder-decoder model
            output = output.strip()

            out_tok_ids_w_answer_extraction= torch.concat(
                [out_tok_ids[:-1], self.answer_extraction_prompt[0]])[:-1]
            out = self.model(
                prompt_token_ids.to(self.device), 
                decoder_input_ids=out_tok_ids_w_answer_extraction.unsqueeze(0).to(self.device))

        else:
            # decoder-only model
            if '[/INST]' in output:
                output = output.split('[/INST]')[-1].strip()
            elif '### Response' in output:
                output = output.split('### Response')[-1].strip()
            # Pythia format
            elif '### Assistant' in output:
                output = output.split('### Assistant')[-1].strip()

            # remove </s> token and add answer_extraction_prompt, 'So the answer is (', so we can extract the answer
            out_tok_ids_w_answer_extraction = torch.concat([out_tok_ids[:-1], self.answer_extraction_prompt[0]])
            out = self.model(out_tok_ids_w_answer_extraction.unsqueeze(0))
            
            # flag those that are truncated in the json output
            if '70b' in self.model_id:
                truncated = out.logits.shape[1] > max_tokens_to_generate

        if self.calibrate:
            assert self.calibration_params, 'need to run init_calibration() before predicting with self.calibrate=True'
            W = self.calibration_params['cot']
            probs = out.logits[0, -1, valid_option_ids].softmax(0)
            pred_idx = torch.argmax(W @ probs).cpu()
        else:
            pred_idx = torch.argmax(out.logits[0, -1, valid_option_ids]).cpu()
        pred = self.tokenizer.convert_ids_to_tokens(valid_option_ids[pred_idx])

        return {'pred': pred, 'output': output, 'truncated': truncated}


    def _predict_addition(self, prompt: str) -> dict:
        
        # tokenize prompt
        prompt_token_ids = self.tokenizer(
            [prompt], 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt')['input_ids']
        
        # truncate generation so that we fit on gpu (if needed) for 70b Llama 2 model
        if '70b' in self.model_id:
            max_tokens_to_generate = TOTAL_ALLOWABLE_TOKENS - prompt_token_ids.shape[1]

            out_tok_ids = self.model.generate(
                prompt_token_ids.to(self.device), 
                max_new_tokens=max_tokens_to_generate, 
                **MODEL_ARGS)[0]
        else:
            out_tok_ids = self.model.generate(
                prompt_token_ids.to(self.device), 
                max_new_tokens=1000,
                do_sample=True,
                **MODEL_ARGS)[0]
            
        output = self.tokenizer.decode(out_tok_ids.cpu()).strip()

        if '[/INST]' in output:
            output = output.split('[/INST]')[-1].strip()
        elif '### Response' in output:
            output = output.split('### Response')[-1].strip()
        # Pythia format
        elif '### Assistant' in output:
            output = output.split('### Assistant')[-1].strip()

        pred = self._extract_addition_answer(output)

        return {'pred': pred, 'output': output}
    
    
    def _extract_addition_answer(self, output: str) -> int:
        try:
            # try to extract contents of last <answer></answer> tags and cast to int
            part_in_tags = output.split('<answer>')[-1].split('</answer>')[0]
            return int(part_in_tags.strip())
        except:
            return None

