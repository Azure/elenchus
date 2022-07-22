from datasets import load_dataset
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
import argparse
from tqdm.auto import tqdm
from datasets import load_metric
import datasets
from dataset import MyDataset

def eval(dataloader, model):
    metric = load_metric("glue", "mrpc")
    model.eval()
    for batch in dataloader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    
    return metric.compute()

def train(dataloader, model, optimizer, lr_scheduler):
    model.train()
    for batch in dataloader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    
    return
            
def main(args):
    
    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    raw_datasets = load_dataset("glue", "mrpc")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    tokenized_datasets["train"].column_names

    train_sampler = RandomSampler(tokenized_datasets["train"], replacement=False, num_samples=args.batch_size)
    eval_sampler = RandomSampler(tokenized_datasets["validation"], replacement=False, num_samples=args.batch_size * 2)

    train_dataloader = DataLoader(
        tokenized_datasets["train"], batch_size=args.batch_size, collate_fn=data_collator, sampler=train_sampler
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=args.batch_size, collate_fn=data_collator, sampler=eval_sampler
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, num_labels=args.num_labels)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    num_training_steps = args.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(args.num_epochs):
        train(train_dataloader, model, optimizer, lr_scheduler)
        if epoch % args.eval_interval == 0:
            eval_results = eval(eval_dataloader, model)
            print(eval_results)
            
        progress_bar.update(1)

    # final evaluation
    eval_results = eval(eval_dataloader, model)
    print(eval_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, default="bert-base-uncased")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--eval_interval", type=int, default=5)
    args = parser.parse_args()
    main(args)