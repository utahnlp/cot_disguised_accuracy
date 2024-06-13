# Chain-of-Thought Unfaithfulness as Disguised Accuracy

This repository contains the code to accompany our 2023 MLRC paper: [**Chain-of-Thought Unfaithfulness as Disguised Accuracy**](https://arxiv.org/abs/2402.14897)

## Getting Started

Install the required python dependencies with

```bash
pip install -r requirements.txt
```

Ensure you have access to the Llama 2 family of models (see the top of [this page](https://huggingface.co/meta-llama/Llama-2-7b)), and that you've [authenticated your account using a Hugging Face access token](https://huggingface.co/docs/hub/en/security-tokens). To avoid installing the models everytime your run an experiment, make sure you set the environment variable `HF_HOME` where the models will be downloaded:

```bash
export HF_HOME=./.huggingface_cache # or wherever you want to store the models
```


## Running Experiments

### Multiple Choice Question (MCQ)

The code below shows how to recreate the results for the MCQ experiments. 

```bash
# make the results directory
mkdir -p ./results/mcq 

# run experiment for a particular model, task, and shuffle strategy
python src/predict_mcq.py \
    --model_id <MODEL_ID> \
    --task_id <TASK_ID> \
    --shuffle_strategy <SHUFFLE_STRATEGY> \
    --out_dir ./results/mcq
```

The options for `<MODEL_ID>` are:
* Llama 2 models: `['llama-2-7b', 'llama-2-13b', 'llama-2-70b']`
* FLAN-T5/UL2 models: `['flan-t5-small', 'flan-t5-base', 'flan-t5-large', 'flan-t5-xl', 'flan-t5-xxl', 'flan-ul2']`
* Pythia DPO models: `['pythia-70m-dpo', 'pythia-160m-dpo', 'pythia-410m-dpo', 'pythia-1b-dpo', 'pythia-1.4b-dpo', 'pythia-2.8b-dpo']`.

The options for `<TASK_ID>` are: `['aqua', 'arc_easy', 'arc_challenge', 'openbook_qa', 'logiqa', 'truthful_qa', 'hella_swag', 'mmlu']`

The options for `<SHUFFLE_STRATEGY>` are:
* `'none'` - all question choices are presented in the same order as the original test dataset as downloaded from Hugging Face. This is referred to as "Original Ordering" in the paper.
* `'once'` - The MCQ choice ordering is shuffled as to be different from the original test dataset.
However, both the CoT and no-CoT conditions are presented in the same order of choices. This is referred to as "Same Ordering" in the paper.
* `'twice'` - The MCQ choice ordering is shuffled such that the CoT and no-CoT conditions
get different orderings from each other, and they are both different from the original dataset. This is referred to as "Different Ordering" in the paper.

### Addition

The code below shows how to recreate the results for the addition experiments. 

```bash
# make the results directory
mkdir -p ./results/addition 

# run experiment for a particular model and digit/operand configuration
python src/predict_addition.py \
    --model_id <MODEL_ID> \
    --num_examples <NUM_EXAMPLES> \
    --digits <DIGITS> \
    --operands <OPERANDS> \
    --out_dir ./results/addition  \
    --data_dir ./data \
    --use_precomputed
```

The choices for `<MODEL_ID>` are the same as for the MCQ experiments.

`<DIGITS>` should be one of `[2, 3]`, and `<OPERANDS>` should be one of `[2, 4, 8, 16]`.

To use our precomputed addition data, set `<NUM_EXAMPLES>` to `1000`, and include the `--use_precomputed` flag. To create your own precomputed data, see [create_addition_data.sh](./create_addition_data.sh). To randomly create examples on the fly, use the `--use_random` flag instead of the `--use_precomputed` flag.

## Citation

```
@misc{bentham2024chainofthought,
      title={Chain-of-Thought Unfaithfulness as Disguised Accuracy}, 
      author={Oliver Bentham and Nathan Stringham and Ana Marasović},
      year={2024},
      eprint={2402.14897},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



