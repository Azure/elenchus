import argparse
import json
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
from transformers import  DataCollatorWithPadding
from datasets import load_metric
from dataset import SQLDataset
from tqdm.auto import tqdm

def eval(dataloader, model, subset='qnli'):
    metric = load_metric("glue", subset)
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
    with open(args.config_file, "r") as f:
        config = json.load(f)
        
    raw_datasets = {}
    raw_datasets['train'] = SQLDataset(args, split=config['data']['train'])
    raw_datasets['validation'] = SQLDataset(args, split=config['data']['validation'])

    data_collator = DataCollatorWithPadding(tokenizer=raw_datasets['train'].tokenizer)

    persistent_workers = config['model']['num_workers'] > 0
    train_dataloader = DataLoader(
        raw_datasets["train"], batch_size=config['model']['batch_size'], collate_fn=data_collator, num_workers=config['model']['num_workers'], persistent_workers=persistent_workers
    )
    eval_dataloader = DataLoader(
        raw_datasets["validation"], batch_size=config['model']['batch_size'], collate_fn=data_collator, shuffle=False, num_workers=config['model']['num_workers'], persistent_workers=persistent_workers
    )

    model = AutoModelForSequenceClassification.from_pretrained(config['model']['checkpoint'], num_labels=config['data']['num_labels'])

    optimizer = AdamW(model.parameters(), lr=config['model']['lr'])

    num_training_steps = config['model']['num_epochs'] * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(config['model']['num_epochs']))
    
    for epoch in range(config['model']['num_epochs']):
        if epoch % config['model']['eval_interval'] == 0:
            eval_results = eval(eval_dataloader, model, subset=config['data']['glue_subset'])
            print("epoch:", epoch, eval_results)

        train(train_dataloader, model, optimizer, lr_scheduler)

        progress_bar.update(1)
    
    # final evaluation
    eval_results = eval(eval_dataloader, model, subset=config['data']['glue_subset'])
    print("epoch:", epoch, eval_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="config.json", help="config file with SQL configuration parameters")
    args = parser.parse_args()
    main(args)