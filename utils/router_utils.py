from typing import List
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from models.llama import LlamaWrapper, LlamaWrapperWithMLPSize


def generate_router(sample_model, all_dataloaders, all_domains, device, lambda_, max_num_steps, print_interval, strategy='majority_voting', setting='reasoning', router_size='small', ask_embedder=False):

    if strategy == 'MoDEM':

        router = AutoModelForSequenceClassification.from_pretrained(
            f"microsoft/deberta-v3-{router_size}",
            num_labels=len(all_domains)
        )

        if setting == 'reasoning':
            path = f'checkpoints/MoDEM_router{f"_size={router_size}" if router_size == "large" else ""}_setting=reasoning_lr=0.1_numepochs=100.pth'
        elif setting == 'clm':
            path = f'checkpoints/MoDEM_router{f"_size={router_size}" if router_size == "large" else ""}_setting=clm_lr=0.1_numepochs={"100" if router_size == "small" else "10"}.pth'
        elif setting == 'clm2':
            path = f'checkpoints/MoDEM_router{f"_size={router_size}" if router_size == "large" else ""}_setting=clm2_lr=0.1_numepochs=10.pth'
        else:
            raise ValueError(f"Unknown setting: {setting}")
        
        state = torch.load(path, map_location='cpu')
        router.load_state_dict(state)
        return router, None

    if strategy in ('random_router', 'kNN'):
        if ask_embedder:
            embedder = deepcopy(sample_model._model.model.embed_tokens).to(torch.float32)
            embedder.eval()
            return None, embedder
        return None, None

    if strategy in ('majority_voting', 'average'):

        embedder = deepcopy(sample_model._model.model.embed_tokens).to(torch.float32)
        embedder.eval()

        A = torch.zeros((embedder.weight.shape[1] + 1, embedder.weight.shape[1] + 1), dtype=torch.float32, device=device)
        b = torch.zeros((embedder.weight.shape[1] + 1, len(all_domains)), dtype=torch.float32, device=device)

        for expert_id, (domain, dataloaders) in enumerate(all_dataloaders.items()):
            
            print(f"Processing domain {domain} with expert_id {expert_id}...")
            train_dataloader = dataloaders['train']
            
            for step, batch in enumerate(train_dataloader):
                
                with torch.no_grad():

                    input_ids = batch["input_ids"].to(device, non_blocking=True)            
                    X = embedder(input_ids)
                    X = X.reshape(-1, X.size(-1))
                    X_with_bias = torch.cat((X, torch.ones((X.shape[0], 1), dtype=torch.float32).to(X.device)), dim=1)
                    
                    one_hot = F.one_hot(
                        torch.as_tensor(expert_id, device=X_with_bias.device),
                        num_classes=len(all_domains)
                    ).float()
                    rows = X_with_bias.size(0)
                    Y = one_hot.unsqueeze(0).expand(rows, -1)

                    batch_A = X_with_bias.T @ X_with_bias
                    batch_b = X_with_bias.T @ Y
                    A += batch_A
                    b += batch_b

                    if (step + 1) % print_interval == 0:
                        print(f"Processed {step + 1} batches for domain {domain}.")
                    
                    if step + 1 >= max_num_steps:
                        break
        
        W = torch.linalg.solve(A + lambda_ * torch.eye(A.shape[0], device=A.device), b)
        bias = W[-1, :]
        W = W[:-1, :]
        norm = torch.norm(W, dim=0, keepdim=True)
        if torch.any(norm == 0.0):
            print("WARNING: 0 encountered in norm, substituting with 1e-6")
            norm[norm == 0.0] = 1e-6
        W = W / norm
        bias = bias / norm[0, :]

        router = torch.nn.Linear(W.shape[0], W.shape[1]).to(device)
        router.weight.data = W.T
        router.bias.data = bias.T

        return router, embedder
    
    raise ValueError(f"Strategy {strategy} not implemented.")


class ReasoningRouteWrapper(torch.nn.Module):

    def __init__(self, experts: List[LlamaWrapper | LlamaWrapperWithMLPSize], embedder: torch.nn.Linear,
                 router: torch.nn.Linear, strategy: str = 'majority_voting', entropy_top_k: int | None = None, model_size: str = "small",
                 all_embeddings=None, domain_ids=None, bert=None, encoding_tokenizer=None, decoding_tokenizer=None, k_knn=None):
        super().__init__()
        self.experts = torch.nn.ModuleList(experts)
        self.embedder = embedder
        if router is not None:
            self.router = router
        else:
            self.router = nn.Linear(embedder.weight.shape[1], len(experts)).to(embedder.weight.device)
        self.strategy = strategy
        self.entropy_top_k = entropy_top_k
        self.count_mode = False
        self.confusion_matrix = None
        self.decoding_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B") if decoding_tokenizer is None else decoding_tokenizer
        self.encoding_tokenizer = AutoTokenizer.from_pretrained(f"microsoft/deberta-v3-{model_size}") if encoding_tokenizer is None else encoding_tokenizer
        self.all_embeddings = all_embeddings
        self.domain_ids = domain_ids
        self.bert = bert
        self.k_knn = k_knn

    def _route(self, input_ids: torch.Tensor):
        if self.strategy in ('majority_voting', 'average'):
            X = self.embedder(input_ids)
            batch_size = X.size(0)
            X = X.reshape(-1, X.size(-1))
            outputs = self.router(X)
            if self.strategy == 'majority_voting':
                if self.entropy_top_k is not None:
                    outputs = outputs.view(batch_size, -1, len(self.experts))
                    probs = F.softmax(outputs, dim=2)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=2)
                    k = min(self.entropy_top_k, entropy.size(1))
                    _, idx = torch.topk(entropy, k=k, largest=False, dim=1)
                    mask = torch.zeros_like(entropy, dtype=torch.bool)
                    mask.scatter_(1, idx, True)
                    # expand indices to match last dim
                    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, len(self.experts))     # (16, 10, 5)
                    # gather along dim=1
                    outputs = outputs.gather(1, idx_expanded)                             # (16, 10, 5)
                predicted_expert = torch.argmax(outputs, dim=-1)
                assigned_experts = predicted_expert.view(batch_size, -1).mode(dim=1)[0]
            elif self.strategy == 'average':
                batch_output = outputs.view(batch_size, -1, len(self.experts)).mean(dim=1)
                assigned_experts = torch.argmax(batch_output, dim=1)
            else:
                raise ValueError(f"Invalid strategy: {self.strategy}")
        elif self.strategy == 'random_router':
            batch_size = input_ids.size(0)
            assigned_experts = torch.randint(0, len(self.experts), (batch_size,), device=input_ids.device)
        elif self.strategy == 'MoDEM':
            batch_size = input_ids.size(0)
            decoded = [self.decoding_tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            encoded = self.encoding_tokenizer(decoded, return_tensors='pt', padding=True, truncation=True).to(input_ids.device)
            outputs = self.router(encoded.input_ids)
            assigned_experts = torch.argmax(outputs.logits, dim=-1)
        elif self.strategy == 'kNN':
            assert self.all_embeddings is not None and self.domain_ids is not None, "Embeddings and domain IDs must be generated for kNN strategy."
            assert self.bert is not None, "BERT model must be loaded for kNN strategy."
            assert self.encoding_tokenizer is not None and self.decoding_tokenizer is not None, "Tokenizers must be loaded for kNN strategy."
            assert self.k_knn is not None, "k_knn must be specified for kNN strategy."
            batch_size = input_ids.size(0)
            decoded = [self.decoding_tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            encoded = self.encoding_tokenizer(decoded, return_tensors='pt', padding=True, truncation=True).to(input_ids.device)
            with torch.no_grad():
                query_embeddings = self.bert(encoded['input_ids']).last_hidden_state[:, 0, :]
            query_norm = F.normalize(query_embeddings, p=2, dim=1)
            cos_sim = torch.matmul(query_norm, self.all_embeddings.T)
            cos_dist = 1 - cos_sim
            knn_indices = torch.topk(cos_dist, k=self.k_knn, largest=False).indices
            knn_domain_ids = self.domain_ids[knn_indices]
            assigned_experts = torch.mode(knn_domain_ids, dim=1).values.to(input_ids.device)
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        return batch_size, assigned_experts

    def _group_by_expert(self, assigned_expert: torch.Tensor):
        expert_to_inputs = {}
        for batch_idx, expert_idx in enumerate(assigned_expert):
            expert_idx = expert_idx.item()
            if expert_idx not in expert_to_inputs:
                expert_to_inputs[expert_idx] = []
            expert_to_inputs[expert_idx].append(batch_idx)
        return expert_to_inputs
    
    def set_count_mode(self, mode):
        self.count_mode = mode
        if mode:
            self.confusion_matrix = torch.zeros((len(self.experts), len(self.experts)), dtype=torch.int32)

    def forward(self, input_ids: torch.Tensor, expert_id=None):
        
        if self.count_mode:
            assert expert_id is not None, "expert_id must be provided in count mode"

        batch_size, assigned_expert = self._route(input_ids)
        expert_to_inputs = self._group_by_expert(assigned_expert)
        if self.count_mode:
            for predicted_expert_idx in assigned_expert:
                self.confusion_matrix[expert_id, predicted_expert_idx] += 1  # type: ignore
    
        final_outputs = [None] * batch_size
        
        for expert_idx, batch_indices in expert_to_inputs.items():
            expert_input_ids = input_ids[batch_indices]
            expert_outputs = self.experts[expert_idx](expert_input_ids)
            for i, batch_idx in enumerate(batch_indices):
                final_outputs[batch_idx] = expert_outputs[i]
        
        try:
            return torch.stack(final_outputs)  # type: ignore
        except RuntimeError as e:
            # print(f"WARNING: error stacking final outputs: {e}. Cropping to minimum length.")
            min_length = min(output.shape[1] for output in final_outputs)  # type: ignore
            final_outputs = [output[:, :min_length] for output in final_outputs]  # type: ignore
            return torch.stack(final_outputs)  # type: ignore
    
    def generate(self, input_ids: torch.Tensor, attention_mask, max_new_tokens, do_sample, temperature, pad_token_id, expert_id=None):
        
        batch_size, assigned_expert = self._route(input_ids)
        expert_to_inputs = self._group_by_expert(assigned_expert)
        if self.count_mode:
            for predicted_expert_idx in assigned_expert:
                self.confusion_matrix[expert_id, predicted_expert_idx] += 1  # type: ignore
    
        final_outputs = [None] * batch_size
        
        for expert_idx, batch_indices in expert_to_inputs.items():
            expert_input_ids = input_ids[batch_indices]
            expert_outputs = self.experts[expert_idx].generate(  # type: ignore
                expert_input_ids,
                attention_mask=attention_mask[batch_indices],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample, 
                temperature=temperature,
                pad_token_id=pad_token_id
            )
            for i, batch_idx in enumerate(batch_indices):
                final_outputs[batch_idx] = expert_outputs[i]
        try:
            return torch.stack(final_outputs)  # type: ignore
        except RuntimeError as e:
            # print(f"WARNING: error stacking final outputs: {e}. Cropping to minimum length.")
            min_length = min(output.shape[1] for output in final_outputs)  # type: ignore
            final_outputs = [output[:, :min_length] for output in final_outputs]  # type: ignore
            return torch.stack(final_outputs)  # type: ignore