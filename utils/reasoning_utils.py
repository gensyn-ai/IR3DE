import os
import random

import torch
from torch.utils.data import DataLoader
from lm_eval.tasks import get_task_dict


TASK_MAP = {
    "math": "gsm8k",
    "multilingual": "m_arc",
    "coding": "humaneval",
    "instruction": "ifeval"
}


MODEL_MAP = {
    "math": "MergeBench/Llama-3.2-3B_math",
    "multilingual": "MergeBench/Llama-3.2-3B_multilingual",
    "coding": "MergeBench/Llama-3.2-3B_coding",
    "instruction": "MergeBench/Llama-3.2-3B_instruction"
}


NUM_SHOTS = {
    'gsm8k': 8,
    'mathqa': 0,
    'hendrycks_math': 0,
    'm_mmlu': None,
    'humaneval': None,
    'ifeval': None,
    'm_arc': None
}

MODEL_MAP = {
    "math": "MergeBench/Llama-3.2-3B_math",
    "multilingual": "MergeBench/Llama-3.2-3B_multilingual",
    "coding": "MergeBench/Llama-3.2-3B_coding",
    "instruction": "MergeBench/Llama-3.2-3B_instruction"
}


def get_task(task_name):

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

    if task_name in ("humaneval", "ifeval", "m_arc"):
        limit = None
    else:
        task = task_dict[task_name]
        task.fewshot = NUM_SHOTS[task_name]
        task.bootstrap_iters = 0
        limit = None
    
    return task_dict, limit


class IRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_length=1025, dataset_name='gsm8k'):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.dataset)

    def get_prompt(self, idx):
        if self.dataset_name == 'gsm8k':
            prompt = f"{self.dataset[idx]['question']}"
        elif self.dataset_name == 'm_arc':
            prompt = f"{self.dataset[idx]['instruction']}"
        elif self.dataset_name == 'humaneval' or self.dataset_name == 'ifeval':
            prompt = f"{self.dataset[idx]['prompt']}"
        else:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")
        return prompt

    def __getitem__(self, idx):
        prompt = self.get_prompt(idx)
        out = self.tokenizer(prompt, max_length=self.max_length, padding='max_length', truncation=True)
        return {
            "input_ids": torch.tensor(out['input_ids'][:-1]),
            "label": torch.tensor(out['input_ids'][1:]),
            "attention_mask": torch.tensor(out['attention_mask'][:-1])
        }


def get_m_arc_merged_dataset(max_num_samples, task, split):
    remaining_samples = max_num_samples
    dataset = []
    while remaining_samples > 0:
        num_samples_per_arc = max(1, int(remaining_samples / len(task.keys())))  # type: ignore
        for arc_task_name in task.keys():  # type: ignore
            dataset.extend(list(task[arc_task_name].dataset[split])[:num_samples_per_arc])
            remaining_samples -= num_samples_per_arc
            if remaining_samples <= 0:
                break
    return dataset


def get_dataloaders(domain, max_num_samples, tokenizer, batch_size, num_workers):

    with torch.no_grad():

        tn = TASK_MAP[domain]
        task_dict, _ = get_task(tn)

        if tn == "m_arc":
            train_dataset = get_m_arc_merged_dataset(max_num_samples, task_dict, split='train')
            test_dataset = get_m_arc_merged_dataset(max_num_samples, task_dict, split='test')
        else:
            task = task_dict[tn]
            if tn == "humaneval":
                key1, key2 = 'test', 'test'  # humaneval doesn't have a train/test split, so we use the test set for both (as we are not generating any answer from the prompts, just using them for embedding extraction)
            elif tn == "ifeval":
                key1, key2 = 'train', 'train'  # ifeval doesn't have a train/test split, so we use the train set for both (as we are not generating any answer from the prompts, just using them for embedding extraction)
            else:
                key1, key2 = 'train', 'test'
            train_dataset = list(task.dataset[key1])
            test_dataset = list(task.dataset[key2])

        if len(train_dataset) < max_num_samples:
            original_dataset = train_dataset.copy()
            while len(train_dataset) < max_num_samples:
                remaining_needed = max_num_samples - len(train_dataset)
                if remaining_needed >= len(original_dataset):
                    train_dataset.extend(original_dataset)
                else:
                    train_dataset.extend(original_dataset[:remaining_needed])
                    break
        elif len(train_dataset) > max_num_samples:
            random.shuffle(train_dataset)
            train_dataset = train_dataset[:max_num_samples]
        
        train_dataset = IRDataset(train_dataset, tokenizer, dataset_name=tn)
        test_dataset = IRDataset(test_dataset, tokenizer, dataset_name=tn)

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True),
            'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
        }

        return dataloaders


def set_total_tests(total):
    os.environ['HUMANEVAL_TOTAL_TESTS'] = str(total)