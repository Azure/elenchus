import os
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
import pyodbc
from sqlalchemy import create_engine
import json
import urllib
import torch

def generate_embedding(sentences, model, tokenizer, device='cuda'):
    tokens_batch = tokenizer.batch_encode_plus(sentences, truncation=True, padding=True, return_tensors="pt")
    for key in tokens_batch.keys():
        tokens_batch[key] = tokens_batch[key].to(device)
    with torch.no_grad():
        outputs = model(**tokens_batch)
    
    json_out = []
    for i in range(len(sentences)):
        json_out.append(
            json.dumps(outputs.logits[i].cpu().numpy().tolist())
        )

    return json_out

def create_sql_engine(config):
    conn = f"""Driver={config['sql']['driver']};Server=tcp:{config['sql']['server']},1433;Database={config['sql']['database']};
    Uid={config['sql']['username']};Pwd={config['sql']['password']};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"""

    params = urllib.parse.quote_plus(conn)
    conn_str = 'mssql+pyodbc:///?autocommit=true&odbc_connect={}'.format(params)
    engine = create_engine(conn_str,echo=False,fast_executemany=True, pool_size=1000, max_overflow=100)

    print('connection is ok')

    return engine

def add_clustered_index(table, engine):
    print("adding clustered index")

    stmt = "DROP INDEX IF EXISTS %s_idx ON %s" % (table, table)
    _ = engine.execute(stmt)

    # primary index as to be NOT NULL
    stmt = "ALTER TABLE %s alter column idx bigint NOT NULL" % table
    _ = engine.execute(stmt)

    # add primary key
    stmt = """ALTER TABLE %s
            ADD CONSTRAINT %s_idx PRIMARY KEY CLUSTERED (idx)""" % (table, table)
    _ = engine.execute(stmt)

def test_table(table, engine):
    print("Testing connection to SQL server.")
    stmt = "SELECT * FROM %s" % table
    res = engine.execute(stmt)
    row = res.fetchone()
    print(row)
    
with open("config.json", "r") as f:
    config = json.load(f)

if config['data']['use_embeddings'] == True:    
    
    model = AutoModelForSequenceClassification.from_pretrained(config['model']['checkpoint'])
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Identity(in_features)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(config['model']['checkpoint'])

raw_datasets = load_dataset(config['data']['dataset'], config['data']['subset'])

os.makedirs('data', exist_ok=True)

engine = create_sql_engine(config)

import time
batch_size = 64
sentences = []

for split in ['test']: #raw_datasets.keys():
    df_dict = {"idx":[]}
    for column_name in raw_datasets.column_names[split]:
        df_dict[column_name] = []
    start = time.time()
    for i, example in enumerate(raw_datasets[split]):
        for column_name in raw_datasets.column_names[split]:
            df_dict[column_name].append(example[column_name])
        df_dict['idx'].append(i)
        if config['data']['use_embeddings'] == True:
            sentences.append(example[config['data']['text_column']])
            if i % batch_size == 0 and i > 0:
                embeddings = generate_embedding(sentences, model, tokenizer, device)
                if 'embedding' not in df_dict.keys():
                    df_dict['embedding'] = []
                df_dict['embedding'] += embeddings
                sentences = []

        if i % 10000 == 0 and i > 0:
            print("Split: %s," % split, "%0.1f%%" % (i/len(raw_datasets[split]) * 100), ", %0.3f s/sentence" % ((time.time() - start)/i))

    if len(sentences) > 0:
        embeddings = generate_embedding(sentences, model, tokenizer, device)
        df_dict['embedding'] += embeddings

    df = pd.DataFrame(df_dict)

    table_name = config['data']['dataset'] + split
    try:
        print("creating table")
        print("table name:", table_name)
        df.to_sql(table_name, con=engine, if_exists='replace', index=False, method='multi', chunksize=100)
    except Exception as e:
        print(e)
        print("failed")

    add_clustered_index(table_name, engine)

    test_table(table_name, engine)

    engine.dispose()