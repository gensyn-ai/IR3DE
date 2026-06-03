import os
import argparse

import torch
from transformers import set_seed

from utils.datasets_utils import get_dataloaders
from models.llama_experts import get_llama_expert
from utils.datasets_utils import eval_clm


def get_args():
    parser = argparse.ArgumentParser(description='Cross-evaluate LLaMA experts across domains')
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-parameters', type=float, default=1.15e8, help='Number of model parameters')
    parser.add_argument("--num-test-steps", type=int, default=16, help="Number of test evaluation steps")
    parser.add_argument('--eval-avg-experts', action='store_true', help='Whether to evaluate the average of all experts or just the individual experts')
    args = parser.parse_args()
    return args


def load_domain_statedict(domain, seed=42, num_parameters=1.15e8):
    path = os.path.join('checkpoints', f"llama_{int(num_parameters)}_{domain}_seed={seed}_freeze_attn.pth")
    if os.path.isfile(path):
        print(f"Loading checkpoint from {path}")
        state = torch.load(path, map_location='cpu')
    else:
        raise FileNotFoundError(f"Checkpoint path not found: {path}")
    return state


def eval_loop(all_domains, domain_1, all_dataloaders, model, num_test_steps, device, perplexities):
    for domain_2 in all_domains:
        print(f"Evaluating domain {domain_2} using model trained on {domain_1}...")
        dataloaders = all_dataloaders[domain_2]
        perplexity = eval_clm(dataloaders['test'], model, tot_steps=num_test_steps, device=device, ev_type='Test', dtype=torch.bfloat16)
        print(f"Perplexity on domain {domain_2} using model from {domain_1}: {perplexity:.4f}")
        if perplexities.get(domain_1) is None:
            perplexities[domain_1] = {}
        perplexities[domain_1][domain_2] = perplexity

def main():
    
    args = get_args()
    all_domains = ['cs_l1', 'math_l1', 'physics_l1', 'History_and_events', 'Philosophy_and_thinking']

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_dataloaders = {}
    for domain in all_domains:
        all_dataloaders[domain] = get_dataloaders(dataset_name=domain, batch_size=args.batch_size, num_workers=args.num_workers)
    
    if args.eval_avg_experts:

        print("Evaluating average of all experts...")
        avg_state_dict = None
        for domain in all_domains:
            state_dict = load_domain_statedict(domain, seed=args.seed, num_parameters=args.num_parameters)
            if avg_state_dict is None:
                avg_state_dict = state_dict
            else:
                for k in avg_state_dict.keys():
                    avg_state_dict[k] += state_dict[k]
        for k in avg_state_dict.keys():  # type: ignore
            avg_state_dict[k] /= len(all_domains)  # type: ignore
        model, _, _ = get_llama_expert(args.num_parameters)
        model.load_state_dict(avg_state_dict, strict=False)  # type: ignore
        model.to(device)
        perplexities = {}
        eval_loop(all_domains, 'avg_model', all_dataloaders, model, args.num_test_steps, device, perplexities)
        print("," + ",".join(f"{domain}" for domain in all_domains))
        print(f"avg_model," + ','.join(str(round(perplexities['avg_model'][domain], 2)) for domain in all_domains))
    
    else:

        perplexities = {}

        for domain_1 in all_domains:

            model, _, _ = get_llama_expert(args.num_parameters)

            state_dict = load_domain_statedict(domain_1, seed=args.seed, num_parameters=args.num_parameters)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)

            eval_loop(all_domains, domain_1, all_dataloaders, model, args.num_test_steps, device, perplexities)

        print("," + ",".join(f"{domain}" for domain in all_domains))
        for domain_1 in all_domains:
            print(f"{domain_1}", end=",")
            print(','.join(str(round(perplexities[domain_1][domain_2], 2)) for domain_2 in all_domains))


if __name__ == '__main__':
    main()
