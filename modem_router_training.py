import argparse
from tqdm import tqdm

import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from utils.datasets_utils import get_raw_clm2_datasets, get_raw_clm_datasets


def get_args():
    parser = argparse.ArgumentParser(description="MoDEM Router Training Script")
    parser.add_argument("--setting", type=str, default='clm2', choices=['reasoning', 'clm', 'clm2'], help="Which setting to run the router training for")
    parser.add_argument("--max_num_samples", type=int, default=1750, help="Maximum number of samples per task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length for tokenization")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--val_interval", type=int, default=1, help="Number of epochs between validation checks")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'large'], help='Size of the DeBERTa model to use for the router')
    args = parser.parse_args()
    return args


class MergedDataset(torch.utils.data.Dataset):

    def __init__(self, task_dicts, split, tokenizer, max_length=1024, setting='reasoning'):
        self.task_dicts = task_dicts
        self.split = split
        self.tasks = list(task_dicts.keys())
        self.task_lengths = {task: len(task_dicts[task][split]) for task in self.tasks}
        self.total_length = sum(self.task_lengths.values())
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.setting = setting
        self.idx_map = list(range(len(self)))

    def __len__(self):
        return self.total_length

    def get_prompt(self, sample, task_id):
        if self.setting == 'reasoning':
            if task_id == 0:  # math
                return sample['query']
            if task_id == 1:  # multilingual
                return sample['inputs']
            if task_id == 2:  # coding
                return sample['problem']
            if task_id == 3:  # instruction
                return sample['prompt']
            raise ValueError(f"Unsupported task_id: {task_id}")
        if self.setting == 'clm2':
            return sample['text']
        return sample

    def __getitem__(self, idx):
        cumulative_length = 0
        real_idx = self.idx_map[idx]
        for task_id, task in enumerate(self.tasks):
            if real_idx < cumulative_length + self.task_lengths[task]:
                sample_idx = real_idx - cumulative_length
                prompt = self.get_prompt(self.task_dicts[task][self.split][sample_idx], task_id)
                encoding = self.tokenizer(prompt, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
                return {
                    "input_ids": encoding['input_ids'].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                    "label": torch.tensor(task_id)
                }
            cumulative_length += self.task_lengths[task]
        raise IndexError("Index out of range")
    
    def shuffle(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.idx_map = torch.randperm(len(self)).tolist()


class UnshardedDataset(Dataset):

    def __init__(self, sharded_dataset, tokenizer, max_num_samples=1000):
        self.sharded_dataset = sharded_dataset
        self.tokenizer = tokenizer
        self.dataset = []
        for sample in tqdm(self.sharded_dataset):
            self.dataset.append(sample)
            if len(self.dataset) >= max_num_samples:
                break

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        original_text = self.tokenizer.decode(self.dataset[idx]['input_ids'], skip_special_tokens=True)
        return original_text
    

def get_task_dicts(setting, max_num_samples, seed, batch_size=None, num_workers=None, max_length=None):
    if setting == 'clm':
        task_dicts = {}
        for domain in ['cs_l1', 'math_l1', 'physics_l1', 'History_and_events', 'Philosophy_and_thinking']:
            decoding_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
            datasets = get_raw_clm_datasets(domain)
            task_dicts[domain] = {}
            for partition, dataset in datasets.items():
                print(f"Processing domain {domain}, partition {partition}")
                const = 0.7 if partition == 'train' else 0.1 if partition == 'validation' else 0.2
                task_dicts[domain][partition] = UnshardedDataset(dataset, tokenizer=decoding_tokenizer, max_num_samples=int(const * max_num_samples))
    elif setting == 'clm2':
        task_dicts = {}
        for domain in ['math', 'bio', 'legal', 'dialogue']:
            decoding_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
            decoding_tokenizer.pad_token = decoding_tokenizer.eos_token
            dataset = get_raw_clm2_datasets(batch_size=batch_size, num_workers=num_workers, tokenizer=decoding_tokenizer, max_length=max_length, max_num_samples=max_num_samples, dataset_name=domain)
            train_dataset, test_dataset = dataset['train'].train_test_split(test_size=0.2, seed=seed).values()  # type: ignore
            train_dataset, val_dataset = train_dataset.train_test_split(test_size=1/8, seed=seed).values()
            task_dicts[domain] = {
                "train": train_dataset,
                "val": val_dataset,
                "test": test_dataset
            }
    elif setting == 'reasoning':
        task_dicts = {
            "math": load_dataset("hkust-nlp/dart-math-hard")['train'],
            "multilingual": load_dataset("CohereLabs/aya_dataset")['train'],
            "coding": load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K")['train'],
            "instruction": load_dataset("allenai/tulu-3-sft-personas-instruction-following")['train']
        }
        for task_name, dataset in task_dicts.items():
            print(f"Processing task: {task_name}")
            if len(dataset) > max_num_samples:
                dataset = dataset.shuffle(seed=seed).select(range(max_num_samples))
            train_dataset, test_dataset = dataset.train_test_split(test_size=0.2, seed=seed).values()
            train_dataset, val_dataset = train_dataset.train_test_split(test_size=1/8, seed=seed).values()
            task_dicts[task_name] = {  # type: ignore
                "train": train_dataset,
                "val": val_dataset,
                "test": test_dataset
            }
    else:
        raise ValueError(f"Unsupported setting: {setting}")
    
    return task_dicts


def main():

    args = get_args()
    set_seed(args.seed)

    task_dicts = get_task_dicts(args.setting, args.max_num_samples, args.seed, batch_size=args.batch_size, num_workers=args.num_workers, max_length=args.max_length)
    
    tokenizer = AutoTokenizer.from_pretrained(f"microsoft/deberta-v3-{args.model_size}")
    train_dataset = MergedDataset(task_dicts, split='train', tokenizer=tokenizer, max_length=args.max_length, setting=args.setting)
    train_dataset.shuffle(seed=args.seed)
    
    val_dataset = MergedDataset(task_dicts, split='validation' if args.setting == 'clm' else 'val', tokenizer=tokenizer, max_length=args.max_length, setting=args.setting)
    test_dataset = MergedDataset(task_dicts, split='test', tokenizer=tokenizer, max_length=args.max_length, setting=args.setting)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers, drop_last=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        f"microsoft/deberta-v3-{args.model_size}",
        num_labels=len(task_dicts)
    )
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    model.to('cuda')
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    warmup_steps = int(0.1 * len(train_dataloader) * args.num_epochs)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(
                optimizer,
                start_factor=0.1,   # initial lr = 0.1 * base_lr
                end_factor=1.0,
                total_iters=warmup_steps
            ),
            CosineAnnealingLR(
                optimizer,
                T_max=len(train_dataloader) * args.num_epochs - warmup_steps,
                eta_min=0.0
            )
        ],
        milestones=[warmup_steps]
    )
    best_accuracy = 0.0
    best_val_model_state = None
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}...")
        epoch_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['label'].to('cuda')
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print(f"Loss: {epoch_loss/len(train_dataloader):.4f}")  # type: ignore
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch['input_ids'].to('cuda')
                    attention_mask = batch['attention_mask'].to('cuda')
                    labels = batch['label'].to('cuda')
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
            accuracy = correct / total * 100
            print(f"Validation Accuracy after epoch {epoch+1}: {accuracy:.2f}%")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_val_model_state = model.state_dict()
                print(f"New best model found at epoch {epoch+1} with accuracy {accuracy:.2f}%")
            model.train()

    model.load_state_dict(best_val_model_state)
    model.eval()
    correct, total = 0, 0
    confusion_matrix = torch.zeros(len(task_dicts), len(task_dicts), dtype=torch.int32)
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['label'].to('cuda')
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            for t, p in zip(labels.view(-1), predictions.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    test_accuracy = correct / total * 100
    print(f"Test Accuracy of the best model: {test_accuracy:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix)

    print("Saving the best model...")
    torch.save(best_val_model_state, f"checkpoints/MoDEM_router_size={args.model_size}_setting={args.setting}_lr={args.lr}_numepochs={args.num_epochs}.pth")
    print(f"Best model saved as 'checkpoints/MoDEM_router_size={args.model_size}_setting={args.setting}_lr={args.lr}_numepochs={args.num_epochs}.pth'")


if __name__ == '__main__':
    main()