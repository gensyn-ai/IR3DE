import os
import sys
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel, LlamaForCausalLM, set_seed

from models.llama_experts import get_llama_expert
from models.llama import LLamaWrapperEval
from utils.import_hooks import CodeEvalImportHook
from utils.router_utils import generate_router, ReasoningRouteWrapper
from utils.reasoning_utils import TASK_MAP, MODEL_MAP, get_dataloaders, get_task, set_total_tests

sys.path.insert(0, '[/path/to/your/repo]/IR3DE/utils')
sys.meta_path.insert(0, CodeEvalImportHook())

from lm_eval.evaluator import evaluate

os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def get_args():
    parser = argparse.ArgumentParser(description='Cross-evaluate LLaMA experts across domains')
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument("--use-flash-attn", action='store_true', help="Whether to use flash attention implementation")
    parser.add_argument('--max_num_steps', type=int, default=1000, help='Maximum number of steps for training the router')
    parser.add_argument('--max_num_test_router_steps', type=int, default=16, help='Maximum number of steps for testing the router')
    parser.add_argument('--lambda_', type=float, default=1e-2, help='Regularization strength for ridge regression')
    parser.add_argument('--strategy', type=str, default='kNN', choices=['majority_voting', 'average', 'MoDEM', 'random_router', 'kNN'], help='Strategy for routing decisions')
    parser.add_argument('--router-size', type=str, default='large', choices=['small', 'large'], help='Size of the MoDEM router')
    parser.add_argument('--print_interval', type=int, default=16, help='Interval for printing progress during training and evaluation')
    parser.add_argument('--max_num_samples', type=int, default=1000, help='Maximum number of samples to use from each domain')
    parser.add_argument('--entropy-top-k', type=int, default=10, help='Top-k entropy for routing decisions. If None, no entropy-based filtering is applied.')
    parser.add_argument('--k-knn', type=int, default=5, help='Number of neighbors for kNN strategy')
    args = parser.parse_args()
    return args


def main():

    all_domains = [
        'math', 
        'multilingual', 
        'coding', 
        'instruction'
    ]

    args = get_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B')
    tokenizer.pad_token = tokenizer.eos_token

    all_dataloaders = {}
    all_models = {}

    for domain in all_domains:
        print(f"Processing domain: {domain}...")
        all_dataloaders[domain] = get_dataloaders(domain, max_num_samples=args.max_num_samples, tokenizer=tokenizer, batch_size=args.batch_size, num_workers=args.num_workers)
        model_name = MODEL_MAP[domain]
        model = LlamaForCausalLM.from_pretrained(model_name)
        model.cuda()  # type: ignore
        model_state_dict = model.state_dict()
        model_state_dict = {f'_model.{k}': v for k, v in model_state_dict.items()}

        model_custom = get_llama_expert(3e9, config=model.config, attn_implementation=args.use_flash_attn)[0]
        model_custom.cuda()
        model_custom.load_state_dict(model_state_dict, strict=True)
        all_models[domain] = model_custom

    router, embedder = generate_router(
        sample_model=all_models['coding'],
        all_dataloaders=all_dataloaders,
        all_domains=all_domains,
        device=device,
        lambda_=args.lambda_,
        max_num_steps=args.max_num_steps,
        print_interval=args.print_interval,
        strategy=args.strategy,
        router_size=args.router_size,
        ask_embedder=True,
        setting='reasoning'
    )

    if args.strategy == 'kNN':
        bert = BertModel.from_pretrained("google-bert/bert-base-uncased")
        encoding_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        bert.eval()
        bert.to(device)  # type: ignore
        all_embeddings = []
        domain_ids = []
        decoding_tokenizer = tokenizer
        encoding_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        for domain in all_domains:
            print(f"Extracting bert embeddings for domain: {domain}...")
            for i, batch in tqdm(enumerate(all_dataloaders[domain]['train'])):
                input_ids = batch["input_ids"].to(device)
                decoded = [decoding_tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                encoded = encoding_tokenizer(decoded, return_tensors='pt', padding=True, truncation=True).to(input_ids.device)
                with torch.no_grad():
                    embeddings = bert(encoded['input_ids']).last_hidden_state[:, 0, :].cpu()  # CLS token embedding
                all_embeddings.append(embeddings)
                domain_ids.extend([all_domains.index(domain)] * embeddings.size(0))
        all_embeddings = torch.cat(all_embeddings, dim=0).to(device)
        all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
        domain_ids = torch.tensor(domain_ids).to(device)
    else:
        all_embeddings, domain_ids = None, None
        bert = None
        decoding_tokenizer = None
        encoding_tokenizer = None

    model = ReasoningRouteWrapper(list(all_models.values()), embedder, router, strategy=args.strategy,  # type: ignore
                                  entropy_top_k=args.entropy_top_k, model_size=args.router_size,
                                  all_embeddings=all_embeddings, domain_ids=domain_ids, bert=bert,
                                  encoding_tokenizer=encoding_tokenizer, decoding_tokenizer=decoding_tokenizer, k_knn=args.k_knn)
    model.set_count_mode(True)
    model = LLamaWrapperEval(model, tokenizer, device="cuda")

    model.model.eval()

    with torch.no_grad():
        for i, domain in enumerate(all_domains):
            model.expert_id = i
            with torch.no_grad():
                task_name = TASK_MAP[domain]
                task_dict, limit = get_task(task_name)
                if task_name == "humaneval":
                    set_total_tests(limit if limit else 164)  # 164 is the total humaneval problems
                results = evaluate(
                    lm=model,
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

            print(f"Confusion matrix:\n{model.model.confusion_matrix}")
            accuracy = model.model.confusion_matrix.diag().sum().item() / model.model.confusion_matrix.sum().item()
            normalized_accuracy = (model.model.confusion_matrix.diag() / model.model.confusion_matrix.sum(dim=1)).mean().item()
            print(f"\nRouting accuracy until domain {domain} ({i + 1}/{len(all_domains)}): {accuracy * 100:.2f}%")
            print(f"Normalized routing accuracy until domain {domain} ({i + 1}/{len(all_domains)}): {normalized_accuracy * 100:.2f}%")


if __name__ == '__main__':
    main()
