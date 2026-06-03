import os
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaForCausalLM, set_seed, AutoTokenizer, BertModel

from utils.datasets_utils import eval_clm_predicted_domains, get_dataloaders, PredictedDomainDataset, compute_metrics_clm_predicted_domains, get_clm2_dataloaders
from models.llama_experts import get_llama_expert
from utils.router_utils import generate_router


def get_args():

    parser = argparse.ArgumentParser(description="CLM Inference Routing")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--setting', type=str, default='clm2', choices=['clm', 'clm2'], help='Task setting for the router')
    parser.add_argument('--strategy', type=str, default='majority_voting', choices=['majority_voting', 'average', 'random_router', 'MoDEM', 'kNN'], help='Strategy for routing decisions')
    parser.add_argument('--num_parameters', type=float, default=1e9, help='Number of parameters for the model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--max_num_steps', type=int, default=1000, help='Maximum number of steps for generating the rr router')
    parser.add_argument('--max_num_test_router_steps', type=int, default=16, help='Maximum number of steps for testing the router')
    parser.add_argument('--lambda_', type=float, default=1e-2, help='Regularization strength for ridge regression')
    parser.add_argument('--print_interval', type=int, default=16, help='Interval for printing progress during training and evaluation')
    parser.add_argument('--entropy-top-k', type=int, default=None, help='Top-k entropy for routing decisions. If None, no entropy-based filtering is applied.')
    parser.add_argument('--router-size', type=str, default='large', choices=['small', 'large'], help='Size of the MoDEM router')
    parser.add_argument('--k-knn', type=int, default=5, help='Number of neighbors for kNN strategy')
    parser.add_argument('--max-num-batches-knn', type=int, default=100, help='Maximum number of samples to use from each domain for the knn strategy')
    parser.add_argument('--max-num-tokens', type=int, default=1024, help='Maximum number of tokens per sample')
    parser.add_argument('--max-num-samples', type=int, default=10000, help='Maximum number of samples to use in the dataset')
    args = parser.parse_args()
    return args


def main():

    print("\n****************************************************************************************************\n")

    args = get_args()
    
    if args.setting == 'clm':
        all_domains = ['cs_l1', 'math_l1', 'physics_l1', 'History_and_events', 'Philosophy_and_thinking']
    elif args.setting == 'clm2':
        all_domains = ['math', 'bio', 'legal', 'dialogue']
    else:
        raise ValueError(f"Invalid setting: {args.setting}")

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_dataloaders = {}
    all_models = {}
    
    for domain in all_domains:
        if args.setting == 'clm':
            all_dataloaders[domain] = get_dataloaders(dataset_name=domain, batch_size=args.batch_size, num_workers=args.num_workers, drop_last_test=True)
        elif args.setting == 'clm2':
            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
            tokenizer.pad_token = tokenizer.eos_token
            all_dataloaders[domain] = get_clm2_dataloaders(domain=domain, batch_size=args.batch_size, num_workers=args.num_workers, tokenizer=tokenizer, max_num_tokens=args.max_num_tokens, max_num_samples=args.max_num_samples)
        else:
            raise ValueError(f"Invalid setting: {args.setting}")
        path = os.path.join('checkpoints', f"llama_{int(args.num_parameters)}_{domain}_seed={args.seed}{'_freeze_attn' if args.setting == 'clm' else ''}.pth")
        if os.path.isfile(path):
            print(f"Loading checkpoint from {path}")
            state = torch.load(path, map_location='cpu')
        else:
            raise FileNotFoundError(f"Checkpoint path not found: {path}")
        
        if args.num_parameters == 1.15e8:
            model, _, _ = get_llama_expert(args.num_parameters)
        elif args.num_parameters == 1e9:
            model_base = LlamaForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')
            model, _, _ = get_llama_expert(args.num_parameters, config=model_base.config)
        else:
            raise ValueError(f"Unsupported number of parameters: {args.num_parameters}")
        
        model.load_state_dict(state, strict=False)
        model.to(device)
        all_models[domain] = model

    router, embedder = generate_router(
        sample_model=all_models[all_domains[0]],
        all_dataloaders=all_dataloaders,
        all_domains=all_domains,
        device=device,
        lambda_=args.lambda_,
        max_num_steps=args.max_num_steps,
        print_interval=args.print_interval,
        strategy=args.strategy,
        setting=args.setting,
        router_size=args.router_size
    )
    if args.strategy == 'MoDEM':
        decoding_tokenizer =AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        encoding_tokenizer = AutoTokenizer.from_pretrained(f"microsoft/deberta-v3-{args.router_size}")
    elif args.strategy == 'kNN':
        decoding_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        encoding_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    else:
        decoding_tokenizer, encoding_tokenizer = None, None

    if args.strategy == 'kNN':
        assert decoding_tokenizer is not None and encoding_tokenizer is not None
        bert = BertModel.from_pretrained("google-bert/bert-base-uncased")
        bert.eval()
        bert.to(device)  # type: ignore
        all_embeddings = []
        domain_ids = []
        for domain in all_domains:
            print(f"Extracting bert embeddings for domain: {domain}...")
            for i, batch in tqdm(enumerate(all_dataloaders[domain]['train']), total=args.max_num_batches_knn):
                if i >= args.max_num_batches_knn:
                    break
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

    confusion_matrix = torch.zeros((len(all_domains), len(all_domains)), dtype=torch.int32)
    correct = 0
    total = 0

    predicted_inputs = [[] for _ in all_domains]
    predicted_labels = [[] for _ in all_domains]
    domain_labels = [[] for _ in all_domains]

    for expert_id, (domain, dataloaders) in enumerate(all_dataloaders.items()):
        print(f"Evaluating routing for domain {domain} with expert_id {expert_id}...")
        test_dataloader = dataloaders['test']
        for step, batch in enumerate(test_dataloader):
            
            if batch["input_ids"].size(0) != args.batch_size and args.setting == 'clm2' and args.strategy == 'majority_voting':
                continue

            input_ids = batch["input_ids"].to(device, non_blocking=True)  
            labels = batch["label"].to(device, non_blocking=True)
            
            if args.strategy == 'MoDEM':
                assert router is not None, "Router must be generated for MoDEM strategy."
                assert decoding_tokenizer is not None and encoding_tokenizer is not None
                router.to(device)
                decoded = [decoding_tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                encoded = encoding_tokenizer(decoded, return_tensors='pt', padding=True, truncation=True).to(input_ids.device)
                outputs = router(encoded.input_ids)
                assigned_experts = torch.argmax(outputs.logits, dim=-1)
            
            elif args.strategy == 'kNN':
                assert all_embeddings is not None and domain_ids is not None, "Embeddings and domain IDs must be generated for kNN strategy."
                assert bert is not None, "BERT model must be loaded for kNN strategy."
                assert encoding_tokenizer is not None and decoding_tokenizer is not None, "Tokenizers must be loaded for kNN strategy."
                decoded = [decoding_tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                encoded = encoding_tokenizer(decoded, return_tensors='pt', padding=True, truncation=True).to(input_ids.device)
                with torch.no_grad():
                    query_embeddings = bert(encoded['input_ids']).last_hidden_state[:, 0, :]
                query_norm = F.normalize(query_embeddings, p=2, dim=1)
                cos_sim = torch.matmul(query_norm, all_embeddings.T)
                cos_dist = 1 - cos_sim
                knn_indices = torch.topk(cos_dist, k=args.k_knn, largest=False).indices
                knn_domain_ids = domain_ids[knn_indices]
                assigned_experts = torch.mode(knn_domain_ids, dim=1).values.to(device)

            elif args.strategy == 'random_router':
                assigned_experts = torch.randint(0, len(all_domains), (input_ids.size(0),), device=device)
            
            else:

                assert embedder is not None
                assert router is not None

                X = embedder(input_ids)
                X = X.reshape(-1, X.size(-1))
                outputs = router(X)

                if args.strategy == 'majority_voting':

                    if args.entropy_top_k is not None:
                        outputs = outputs.view(args.batch_size, -1, len(all_domains))
                        probs = F.softmax(outputs, dim=2)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=2)
                        k = min(args.entropy_top_k, entropy.size(1))
                        _, idx = torch.topk(entropy, k=k, largest=False, dim=1)
                        mask = torch.zeros_like(entropy, dtype=torch.bool)
                        mask.scatter_(1, idx, True)
                        # expand indices to match last dim
                        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, len(all_domains))     # (16, 10, 5)
                        # gather along dim=1
                        outputs = outputs.gather(1, idx_expanded)                             # (16, 10, 5)

                    predicted_expert = torch.argmax(outputs, dim=-1)
                    assigned_experts = predicted_expert.view(args.batch_size, -1).mode(dim=1)[0]

                elif args.strategy == 'average':
                    batch_output = outputs.view(args.batch_size, -1, len(all_domains)).mean(dim=1)
                    assigned_experts = torch.argmax(batch_output, dim=1)
                else:
                    raise ValueError(f"Invalid strategy: {args.strategy}")
            
            for pred in assigned_experts:
                confusion_matrix[expert_id, int(pred.item())] += 1
            
            correct += (assigned_experts == expert_id).sum().item()
            total += assigned_experts.size(0)

            for i, pred_expert_id in enumerate(assigned_experts):
                predicted_inputs[int(pred_expert_id.item())].append(input_ids[i].cpu())
                predicted_labels[int(pred_expert_id.item())].append(labels[i].cpu())
                domain_labels[int(pred_expert_id.item())].append(expert_id)
            if (step + 1) % args.print_interval == 0:
                print(f"Evaluated {step + 1} batches for domain {domain}. Current accuracy: {correct / total * 100:.2f}%")
            if step + 1 >= args.max_num_test_router_steps:
                break

    accuracy = correct / total if total > 0 else 0
    print(f"Evaluation accuracy: {accuracy*100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix)

    predicted_test_dataloaders = []
    for domain_id in range(len(all_domains)):
        dataset = PredictedDomainDataset(predicted_inputs[domain_id], predicted_labels[domain_id], domain_labels[domain_id])
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
        predicted_test_dataloaders.append(dataloader)

    perplexities = {}
    tot_ppl_loss = np.array([0.0 for _ in all_domains])
    tot_num_tokens = np.array([0 for _ in all_domains])

    for domain_id, dataloader in enumerate(predicted_test_dataloaders):
        print(f"Evaluating predicted routing for domain {all_domains[domain_id]} with expert_id {domain_id}...")
        model = all_models[all_domains[domain_id]]
        ppl_loss, num_tokens = eval_clm_predicted_domains(
            dataloader=dataloader,
            model=model,
            tot_steps=args.max_num_test_router_steps,
            device=device,
            ev_type='Test',
            num_domains=len(all_domains),
        )
        tot_ppl_loss += np.array(ppl_loss)
        tot_num_tokens += np.array(num_tokens)

    perplexities = compute_metrics_clm_predicted_domains(len(all_domains), tot_ppl_loss.tolist(), tot_num_tokens.tolist())

    print("\nPerplexities for each domain using the RR inference router:")

    for domain in all_domains:
        print(f"{domain},", end='')
    print()

    for domain, ppl in zip(all_domains, perplexities):
        print(f"{ppl:.2f},", end='')
    print()
    print("\n****************************************************************************************************\n")


if __name__ == '__main__':
    main()
