import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
from transformers import  DataCollatorWithPadding
from datasets import load_metric
from dataset import SQLDataset
from tqdm.auto import tqdm

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
            
def main(args):
    
    raw_datasets = {}
    raw_datasets['train'] = SQLDataset(args, split='train')
    raw_datasets['validation'] = SQLDataset(args, split='validation')

    data_collator = DataCollatorWithPadding(tokenizer=raw_datasets['train'].tokenizer)

    train_dataloader = DataLoader(
        raw_datasets["train"], batch_size=args.batch_size, collate_fn=data_collator, num_workers=args.num_workers, persistent_workers=True
    )
    eval_dataloader = DataLoader(
        raw_datasets["validation"], batch_size=args.batch_size, collate_fn=data_collator, shuffle=False, num_workers=args.num_workers, persistent_workers=True
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

    progress_bar = tqdm(range(args.num_epochs))
    
    for epoch in range(args.num_epochs):
        if epoch % args.eval_interval == 0:
            eval_results = eval(eval_dataloader, model)
            print("epoch:", epoch, eval_results)

        train(train_dataloader, model, optimizer, lr_scheduler)

        progress_bar.update(1)
    
    # final evaluation
    eval_results = eval(eval_dataloader, model)
    print("epoch:", epoch, eval_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches-per-worker", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default="bert-base-uncased")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--dataset_root", default="data", help="path to to dataset root")
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--config-file", default="config.json", help="config file with SQL configuration parameters")
    args = parser.parse_args()
    main(args)