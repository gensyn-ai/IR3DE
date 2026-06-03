import os
import glob
import random
import math
from tqdm import tqdm
import importlib.util

import torch
import torch.distributed as dist
from torch.utils.data import get_worker_info, DataLoader
from datasets.io.parquet import ParquetDatasetReader
from datasets import load_dataset
from lm_eval.tasks import get_task_dict

from utils.distributed import is_distributed, get_rank


def eval_clm(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, tot_steps: int,
             device: torch.device, ev_type: str = 'Validation', dtype: torch.dtype | None = None):

    model.eval()
    ppl_loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    rank = get_rank()
    is_main = rank == 0

    with torch.no_grad():
        avg_loss = 0.0
        tot_ppl_loss = 0.0
        tot_num_tokens = 0
        steps = 0
        pbar = tqdm(
            # total=tot_steps,
            desc=ev_type) if is_main else None

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=dtype) if dtype is not None else torch.cuda.amp.autocast(enabled=False):
                outputs = model(input_ids=input_ids)
                num_tokens = labels.shape[0] * labels.shape[1]
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                ppl_loss = ppl_loss_fn(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1)
                )
            avg_loss += ppl_loss.item() / num_tokens
            tot_ppl_loss += ppl_loss.item()
            tot_num_tokens += num_tokens

            steps += 1
            if is_main:
                pbar.update(1)  # type: ignore

            if steps >= tot_steps:
                break

        if is_distributed():
            t = torch.tensor([avg_loss, tot_ppl_loss, tot_num_tokens, steps], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            avg_loss, tot_ppl_loss, tot_num_tokens, steps = t.tolist()

        avg_loss /= steps
        perplexity = math.exp(tot_ppl_loss / tot_num_tokens)
        if is_main and pbar is not None:
            pbar.close()

    if is_main:
        print(f"{ev_type} loss: {avg_loss:.4f}, {ev_type} perplexity: {perplexity:.4f}")

    model.train()

    return perplexity


def eval_clm_predicted_domains(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, tot_steps: int,
                               device: torch.device, ev_type: str = 'Validation', dtype: torch.dtype | None = None, num_domains: int = 5):

    model.eval()
    ppl_loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    rank = get_rank()
    is_main = rank == 0

    with torch.no_grad():
        tot_ppl_loss = [0.0 for _ in range(num_domains)]
        tot_num_tokens = [0 for _ in range(num_domains)]
        steps = 0
        pbar = tqdm(
            # total=tot_steps,
            desc=ev_type) if is_main else None

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            real_domain_labels = batch["domain_label"]

            with torch.cuda.amp.autocast(dtype=dtype) if dtype is not None else torch.cuda.amp.autocast(enabled=False):
                outputs = model(input_ids=input_ids)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Get unique domains in this batch
                unique_domains = torch.unique(real_domain_labels)
                
                for domain_id in unique_domains:
                    # Create mask for this domain
                    domain_mask = (real_domain_labels == domain_id)
                    
                    # Extract domain-specific data
                    domain_input_ids = input_ids[domain_mask]
                    domain_labels = labels[domain_mask] 
                    domain_outputs = outputs[domain_mask]
                    
                    if domain_input_ids.numel() == 0:  # Skip if no tokens for this domain
                        continue
                        
                    num_tokens = domain_labels.shape[0] * domain_labels.shape[1]
                    
                    ppl_loss = ppl_loss_fn(
                        domain_outputs.view(-1, domain_outputs.size(-1)),
                        domain_labels.view(-1)
                    )
                    
                    domain_id_int = domain_id.item()
                    tot_ppl_loss[domain_id_int] += ppl_loss.item()
                    tot_num_tokens[domain_id_int] += num_tokens

            steps += 1
            if is_main:
                pbar.update(1)  # type: ignore

            if steps >= tot_steps:
                break

        if is_distributed():
            # Flatten all per-domain metrics for reduction
            all_metrics = tot_ppl_loss + tot_num_tokens + [steps]
            t = torch.tensor(all_metrics, device=device, dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            all_metrics = t.tolist()
            
            # Reconstruct per-domain metrics
            tot_ppl_loss = all_metrics[:num_domains]
            tot_num_tokens = [int(x) for x in all_metrics[num_domains:2*num_domains]]
            steps = int(all_metrics[2*num_domains])

    if is_main and pbar is not None:
        pbar.close()

    model.train()

    return tot_ppl_loss, tot_num_tokens


def compute_metrics_clm_predicted_domains(num_domains: int, tot_ppl_loss: list[float], tot_num_tokens: list[int]):
    # Calculate per-domain results
    perplexities = []
    for domain_id in range(num_domains):
        if tot_num_tokens[domain_id] > 0:
            perplexity = math.exp(tot_ppl_loss[domain_id] / tot_num_tokens[domain_id])
        else:
            perplexity = float('inf')
        perplexities.append(perplexity)

    return perplexities


def get_raw_clm_datasets(dataset_name: str):
    if dataset_name == 'openwebtext':
        path = 'dataset/openwebtext/raw'

    elif dataset_name == 'math_l1':
        path = 'dataset/math_l1'
    elif dataset_name == 'cs_l1':
        path = 'dataset/cs_l1'
    elif dataset_name == 'physics_l1':
        path = 'dataset/physics_l1'
    elif dataset_name == 'History_and_events':
        path = 'dataset/History_and_events'
    elif dataset_name == 'Philosophy_and_thinking':
        path = 'dataset/Philosophy_and_thinking'
                
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")

    # Resolve dataset directory relative to workspace root (parent of DetGatingMoE-exp)
    workspace_root = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(workspace_root, path)

    # Pre-check: ensure train split exists to avoid opaque HF error
    train_glob = os.path.join(dataset_dir, 'train', '*.parquet')
    if not glob.glob(train_glob):
        raise ValueError(
            f"No Parquet files found for train split at: {train_glob}. "
            f"Verify your dataset path and splits."
        )

    datasets = {
        'train': ShardedDataset(train_glob),
        'validation': ShardedDataset(os.path.join(dataset_dir, 'validation', '*.parquet')),
        'test': ShardedDataset(os.path.join(dataset_dir, 'test', '*.parquet'))
    }

    return datasets


def get_dataloaders(dataset_name: str, batch_size: int, num_workers: int, drop_last_test: bool = False):

    datasets = get_raw_clm_datasets(dataset_name)

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=True),
        'validation': DataLoader(datasets['validation'], batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers),
        'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=drop_last_test)
    }

    return dataloaders


class ShardedDataset(torch.utils.data.IterableDataset):

    def __init__(self, path: str):
        self.path = path
        # Defer reader creation to __iter__ to avoid forking issues with workers
        self._reader = None

    def _ensure_reader(self):
        if self._reader is None:
            self._reader = ParquetDatasetReader(self.path, streaming=True).read()
        return self._reader

    def __iter__(self):
        reader = self._ensure_reader()

        # Discover distributed rank/world size if initialized
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Account for DataLoader workers per process
        wi = get_worker_info()
        if wi is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = wi.id
            num_workers = wi.num_workers

        # Global sharding across all ranks and workers
        shard_id = rank * num_workers + worker_id
        num_shards = world_size * num_workers

        for idx, sample in enumerate(reader):
            if (idx % num_shards) != shard_id:
                continue
            yield {
                key: torch.tensor(value, dtype=torch.int64, device="cpu")
                for key, value in sample.items()
            }

    # def __len__(self):
    #     return len(self.data_reader)


class PredictedDomainDataset(torch.utils.data.Dataset):
    
    def __init__(self, predicted_inputs, predicted_labels, domain_labels):
        self.inputs = predicted_inputs
        self.labels = predicted_labels
        self.domain_labels = domain_labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {"input_ids": self.inputs[idx], "label": self.labels[idx], "domain_label": self.domain_labels[idx]}
    

NUM_SHOTS = {
    'gsm8k': 8,
    'mathqa': 0,
    'hendrycks_math': 0,
    'm_mmlu': None,
    'humaneval': None,
    'ifeval': None,
    'm_arc': None
}


def get_reasoning_task(task_name):

    # Define multilingual ARC task group
    if task_name == "m_arc":
        multilingual_arc_tasks = [
            "arc_ar", "arc_bn", "arc_ca", "arc_de", "arc_es", "arc_eu", 
            "arc_fr", "arc_gu", "arc_hi", "arc_hr", "arc_hu", "arc_hy",
            "arc_id", "arc_it", "arc_kn", "arc_ml", "arc_mr", "arc_ne",
            "arc_nl", "arc_pt", "arc_ro", "arc_ru", "arc_sk", "arc_sr",
            "arc_sv", "arc_ta", "arc_te", "arc_uk", "arc_vi", "arc_zh"
        ]
        task_dict = get_task_dict(multilingual_arc_tasks)  # type: ignore
    else:
        task_dict = get_task_dict([task_name])

    if task_name == 'hendrycks_math':
        limit = None
        for _, tasks in task_dict.items():  # this is a fake for, its length is only 1
            for name, t in tasks.items():
                t.fewshot = 0
                t.bootstrap_iters = 0
    elif task_name == "m_mmlu":
        limit = 1764
    elif task_name in ("humaneval", "ifeval", "m_arc"):
        limit = None
    else:
        task = task_dict[task_name]
        task.fewshot = NUM_SHOTS[task_name]
        task.bootstrap_iters = 0
        limit = None
    
    return task_dict, limit


def set_total_tests(total):
    os.environ['HUMANEVAL_TOTAL_TESTS'] = str(total)


class CodeEvalImportHook:
    def find_spec(self, fullname, path, target=None):
        # Intercept execute imports (working logic)
        if 'execute' in fullname and ('code_eval' in fullname or 'evaluate' in str(path)):
            # Redirect to custom execute.py
            custom_path = '[path/to/your/repo]/IR3DE/utils/execute.py'
            if os.path.exists(custom_path):
                spec = importlib.util.spec_from_file_location(fullname, custom_path)
                print(f"Redirecting {fullname} to {custom_path}")
                return spec
        return None


MODEL_MAP = {
    "math": "MergeBench/Llama-3.2-3B_math",
    "multilingual": "MergeBench/Llama-3.2-3B_multilingual",
    "coding": "MergeBench/Llama-3.2-3B_coding",
    "instruction": "MergeBench/Llama-3.2-3B_instruction"
}

TASK_MAP = {
    "math": "gsm8k",
    "multilingual": "m_arc",
    "coding": "humaneval",
    "instruction": "ifeval"
}


def get_clm2_dataloaders(domain, batch_size, num_workers, tokenizer, max_num_tokens, max_num_samples):
    if domain == 'math':
        return get_clm2_math_dl(batch_size=batch_size, num_workers=num_workers, tokenizer=tokenizer, max_length=max_num_tokens, max_num_samples=max_num_samples)
    if domain == 'bio' or domain == 'legal':
        return get_clm2_bio_legal_dl(batch_size=batch_size, num_workers=num_workers, tokenizer=tokenizer, max_length=max_num_tokens, max_num_samples=max_num_samples, _type=domain)
    if domain == 'dialogue':
        return get_clm2_dialogue_dl(batch_size=batch_size, num_workers=num_workers, tokenizer=tokenizer, max_length=max_num_tokens, max_num_samples=max_num_samples)
    raise ValueError(f"Unsupported domain: {domain}")


def preprocess_samples(tokenizer, key, max_length):
    def preprocess_function(examples):
        out = tokenizer(examples[key], max_length=max_length+1, padding='max_length', truncation=True, return_tensors="pt")
        return {
            "input_ids": out['input_ids'][:, :-1],
            "label": out['input_ids'].clone()[:, 1:],
            "attention_mask": out['attention_mask'][:, :-1]}
    return preprocess_function


def get_clm2_math_dl(batch_size, num_workers, tokenizer, max_length, max_num_samples, return_raw=False, avoid_preprocessing=False):
    
    dataset = load_dataset("open-web-math/open-web-math")
    total_size = len(dataset['train'])
    
    assert max_num_samples <= len(dataset['train']), "max_num_samples cannot be greater than the total size of the dataset"
    all_indices = list(range(total_size))
    random.shuffle(all_indices)
    indices = all_indices[:max_num_samples]
    dataset = dataset['train'].select(indices)
    total_size = max_num_samples
    
    if not avoid_preprocessing:
        preprocess_function = preprocess_samples(tokenizer, key="text", max_length=max_length)
        dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
        dataset.set_format(type="torch", columns=["input_ids", "label", "attention_mask"])
    
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    
    indices = list(range(total_size))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    test_dataset = dataset.select(test_indices)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True),  # type: ignore
        'validation': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers),  # type: ignore
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)  # type: ignore
    }

    datasets = {
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    }
    if return_raw:
        return dataloaders, datasets
    return dataloaders
    

def get_clm2_bio_legal_dl(batch_size, num_workers, tokenizer, max_length, max_num_samples, _type='bio', return_raw=False, avoid_preprocessing=False):
    
    if _type == 'bio':
        dataset = load_dataset("allenai/peS2o")
    elif _type == 'legal':
        dataset = load_dataset("pile-of-law/pile-of-law", "all")
    else:
        raise ValueError(f"Unknown dataset type: {_type}")

    total_size = len(dataset['train']) + len(dataset['validation'])
    
    assert max_num_samples <= total_size, "max_num_samples cannot be greater than the total size of the dataset"

    all_train_indices = list(range(len(dataset['train'])))
    random.shuffle(all_train_indices)
    
    train_indices = all_train_indices[:int(0.7 * max_num_samples)]
    test_indices = all_train_indices[int(0.7 * max_num_samples):int(0.9 * max_num_samples)]
    train_dataset = dataset['train'].select(train_indices)
    test_dataset = dataset['train'].select(test_indices)
    
    all_val_indices = list(range(len(dataset['validation'])))
    random.shuffle(all_val_indices)
    
    val_indices = all_val_indices[:int(0.1 * max_num_samples)]
    val_dataset = dataset['validation'].select(val_indices)

    if not avoid_preprocessing:
        preprocess_function = preprocess_samples(tokenizer, key="text", max_length=max_length)
        train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
        val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=dataset["validation"].column_names)
        test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
        train_dataset.set_format(type="torch", columns=["input_ids", "label", "attention_mask"])
        val_dataset.set_format(type="torch", columns=["input_ids", "label", "attention_mask"])
        test_dataset.set_format(type="torch", columns=["input_ids", "label", "attention_mask"])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True),  # type: ignore
        'validation': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers),  # type: ignore
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)  # type: ignore
    }
    datasets = {
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    }
    if return_raw:
        return dataloaders, datasets
    return dataloaders


def get_clm2_dialogue_dl(batch_size, num_workers, tokenizer, max_length, max_num_samples, return_raw=False, avoid_preprocessing=False):
    
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
    dataset = dataset.map(lambda x: {
        "text": " ".join([m["content"] for m in x["messages"]])
    })
    
    total_size = len(dataset['train_sft']) + len(dataset['test_sft'])
    
    assert max_num_samples <= total_size, "max_num_samples cannot be greater than the total size of the dataset"

    all_train_indices = list(range(len(dataset['train_sft'])))
    random.shuffle(all_train_indices)
    
    train_indices = all_train_indices[:int(0.7 * max_num_samples)]
    val_indices = all_train_indices[int(0.7 * max_num_samples):int(0.8 * max_num_samples)]
    train_dataset = dataset['train_sft'].select(train_indices)
    val_dataset = dataset['train_sft'].select(val_indices)
    
    all_test_indices = list(range(len(dataset['test_sft'])))
    random.shuffle(all_test_indices)
    
    test_indices = all_test_indices[:int(0.2 * max_num_samples)]
    test_dataset = dataset['test_sft'].select(test_indices)

    if not avoid_preprocessing:
        preprocess_function = preprocess_samples(tokenizer, key="prompt", max_length=max_length)
        train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=dataset["train_sft"].column_names)
        val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=dataset["train_sft"].column_names)
        test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=dataset["test_sft"].column_names)
        train_dataset.set_format(type="torch", columns=["input_ids", "label", "attention_mask"])
        val_dataset.set_format(type="torch", columns=["input_ids", "label", "attention_mask"])
        test_dataset.set_format(type="torch", columns=["input_ids", "label", "attention_mask"])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True),  # type: ignore
        'validation': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers),  # type: ignore
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)  # type: ignore
    }
    datasets = {
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    }
    if return_raw:
        return dataloaders, datasets
    return dataloaders

def get_raw_clm2_datasets(batch_size, num_workers, tokenizer, max_length, max_num_samples, dataset_name: str):
    if dataset_name == 'math':
        _, datasets = get_clm2_math_dl(batch_size, num_workers, tokenizer, max_length, max_num_samples, return_raw=True, avoid_preprocessing=True)
        return datasets
    if dataset_name == 'bio' or dataset_name == 'legal':
        _, datasets = get_clm2_bio_legal_dl(batch_size, num_workers, tokenizer, max_length, max_num_samples, _type=dataset_name, return_raw=True, avoid_preprocessing=True)
        return datasets
    if dataset_name == 'dialogue':
         _, datasets = get_clm2_dialogue_dl(batch_size, num_workers, tokenizer, max_length, max_num_samples, return_raw=True, avoid_preprocessing=True)
         return datasets
    raise NotImplementedError(f"Dataset {dataset_name} not supported.")