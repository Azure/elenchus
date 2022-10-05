import os
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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def eval(dataloader, model, config):
    if config['data']['metric'] is not None:
        metric = load_metric(config['data']['metric'])
    else:
        try:
            metric = load_metric(config['data']['dataset'], config['data']['subset'])
        except FileNotFoundError:
            print("WARNING: couldn't load desired eval metric, using accuracy.")
            metric = load_metric('accuracy')
    
    model.eval()
    for batch in dataloader:
        batch = {k: v.to(config['model']['device'] ) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    
    return metric.compute()

def train(dataloader, model, optimizer, lr_scheduler, config):
    model.train()
    for batch in dataloader:
        batch = {k: v.to(config['model']['device']) for k, v in batch.items()}
        
        outputs = model(**batch)

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
            
def main(args):
    with open(args.config_file, "r") as f:
        config = json.load(f)

    # if we are fine-tuning BERT model, rather than training a small model, we should try to use the GPU
    if config['data']['use_embeddings'] == False:
        config['model']['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        config['model']['device'] = torch.device("cpu")
    
    raw_datasets = {}
    raw_datasets['train'] = SQLDataset(args, split=config['data']['train'])
    raw_datasets['validation'] = SQLDataset(args, split=config['data']['validation'])
    
    # define data collator
    if config['data']['use_embeddings'] == True:
        data_collator = None
    else:
        data_collator = DataCollatorWithPadding(tokenizer=raw_datasets['train'].tokenizer)

    persistent_workers = config['model']['num_workers'] > 0
    train_dataloader = DataLoader(
        raw_datasets["train"], batch_size=config['model']['batch_size'], collate_fn=data_collator, num_workers=config['model']['num_workers'], persistent_workers=persistent_workers
    )
    eval_dataloader = DataLoader(
        raw_datasets["validation"], batch_size=config['model']['batch_size'], collate_fn=data_collator, shuffle=False, num_workers=config['model']['num_workers'], persistent_workers=persistent_workers
    )

    # define model
    if config['data']['use_embeddings']:
        from models import MLP
        input_dim = raw_datasets['train'].get_embedding_size()
        hidden_dim = config['data']['num_labels'] // 2
        output_dim = config['data']['num_labels']
        model = MLP(input_dim, hidden_dim, output_dim)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(config['model']['checkpoint'], num_labels=config['data']['num_labels'])
        param_list = list(model.named_parameters())
        # freeze all layers except the classifier weights
        for p, param in enumerate(param_list):
            if param[0].startswith('classifier') == False:
                param[1].requires_grad = False

    model.to(config['model']['device'] )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['model']['lr'],
        weight_decay=config['model']['weight_decay'],
        amsgrad=True)

    num_training_steps = config['model']['num_epochs'] * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(config['model']['num_epochs']))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, timestamp))

    for epoch in range(config['model']['num_epochs']):
        if epoch % config['model']['eval_interval'] == 0:
            eval_results = eval(eval_dataloader, model, config)
            # print("epoch:", epoch, eval_results)
            writer.add_scalar('eval/acc', eval_results['accuracy'], epoch)

        train(train_dataloader, model, optimizer, lr_scheduler, config)

        progress_bar.update(1)
    
    # final evaluation
    eval_results = eval(eval_dataloader, model, config)
    print("epoch:", epoch, eval_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="config.json", help="config file with SQL configuration parameters")
    parser.add_argument("--log-dir", default="runs", help="path to log directory")
    
    args = parser.parse_args()

    main(args)