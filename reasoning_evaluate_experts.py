import argparse
from copy import deepcopy
import os
import sys

import torch
from transformers import AutoTokenizer, LlamaForCausalLM, set_seed

from models.llama_experts import get_llama_expert
from models.llama import LLamaWrapperEval
from utils.datasets_utils import get_reasoning_task, set_total_tests, CodeEvalImportHook, MODEL_MAP, TASK_MAP

sys.path.insert(0, '[/path/to/your/repo]/IR3DE/utils')
sys.meta_path.insert(0, CodeEvalImportHook())

from lm_eval.evaluator import evaluate

os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate LLaMA experts on reasoning tasks')
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument('--train-domain', type=str, default='avg', choices=['math', 'multilingual', 'coding', 'instruction', 'avg'], help='Domain of the expert model to evaluate')
    parser.add_argument('--test-domain', type=str, default='coding', choices=['math', 'multilingual', 'coding', 'instruction'], help='Domain of the evaluation dataset')
    args = parser.parse_args()
    return args

def main():

    args = get_args()
    set_seed(args.seed)

    print("Evaluating LLaMA expert trained on domain:", args.train_domain)
    print("Using evaluation dataset from domain:", args.test_domain)

    if args.train_domain != 'avg':
        model_name = MODEL_MAP[args.train_domain]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(model_name)
        model.cuda()  # type: ignore
        model_state_dict = model.state_dict()
        model_state_dict = {f'_model.{k}': v for k, v in model_state_dict.items()}
    else:
        print("Averaging experts from all domains for evaluation...")
        avg_model = LlamaForCausalLM.from_pretrained(MODEL_MAP['coding'])  # Load any model just to get the config and tokenizer, we'll overwrite the weights with the average state dict
        avg_model.cuda()  # type: ignore
        tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP['coding'])
        avg_state_dict = {}
        for domain in TASK_MAP.keys():
            model_name = MODEL_MAP[domain]
            model = LlamaForCausalLM.from_pretrained(model_name)
            model.cuda()  # type: ignore
            model_state_dict = model.state_dict()
            assert isinstance(avg_state_dict, dict)
            for k in model_state_dict.keys():
                if k not in avg_state_dict:
                    if k in ('model.embed_tokens.weight', 'lm_head.weight'):
                        avg_state_dict[k] = model_state_dict[k][:128256, :]
                    else:
                        avg_state_dict[k] = deepcopy(model_state_dict[k])
                else:
                    if k in ('model.embed_tokens.weight', 'lm_head.weight'):
                        avg_state_dict[k] += model_state_dict[k][:128256, :]
                    else:
                        avg_state_dict[k] += deepcopy(model_state_dict[k])
        assert isinstance(avg_state_dict, dict)
        for k in avg_state_dict.keys():
            avg_state_dict[k] /= len(TASK_MAP)
        model_state_dict = avg_state_dict
        model = avg_model
        model.load_state_dict(model_state_dict, strict=True)
        print("Average model loaded successfully.")

    model_custom = get_llama_expert(3e9, config=model.config)[0]
    model_custom.cuda()
    model_state_dict = {f"_model.{k}": v for k, v in model_state_dict.items()}
    model_custom.load_state_dict(model_state_dict, strict=True)
    model_custom = LLamaWrapperEval(model_custom, tokenizer, device="cuda")

    task_name = TASK_MAP[args.test_domain]
    task_dict, limit = get_reasoning_task(task_name)

    model_custom.model.eval()
    with torch.no_grad():

        if task_name == "humaneval":
            set_total_tests(limit if limit else 164)  # 164 is the total humaneval problems
            
        results = evaluate(
            lm=model_custom,
            task_dict=task_dict,
            limit=limit,
            confirm_run_unsafe_code=True,
        )

    assert results is not None
    # MATH
    if task_name == "gsm8k":
        print(f"Final result: {round(results['results'][task_name]['exact_match,flexible-extract'] * 100, 2)}")
    # MULTILINGUAL UNDERSTANDING
    elif task_name == "m_arc":
        avg = 0.0
        for k, v in results['results'].items():
            print(f"{k}: {round(v['acc_norm,none'] * 100, 2)}")
            avg += v['acc_norm,none']
        avg /= len(results['results'])
        print(f"Final result (average): {round(avg * 100, 2)}")
    # CODING
    elif task_name == "humaneval":
        print(f"Final result: {round(results['results'][task_name]['pass@1,create_test'] * 100, 2)}")
    # INSTRUCTION FOLLOWING
    elif task_name == "ifeval":
        print(f"Final result: {round(results['results'][task_name]['inst_level_strict_acc,none'] * 100, 2)}")
    else:
        raise NotImplementedError("Unknown task for final result printing")


if __name__ == '__main__':
    main()
