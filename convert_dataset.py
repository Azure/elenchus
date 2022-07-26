import os
from datasets import load_dataset
import pandas as pd
import pyodbc
from sqlalchemy import create_engine
import json
import urllib

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


raw_datasets = load_dataset("glue", "mrpc")

os.makedirs('data', exist_ok=True)

engine = create_sql_engine(config)

for split in ['train', 'validation', 'test']:
    sentence1 = []
    sentence2 = []
    labels = []
    idx = []
    for i, example in enumerate(raw_datasets[split]):
        sentence1.append(example["sentence1"])
        sentence2.append(example["sentence2"])
        labels.append(example["label"])
        idx.append(i)
    df = pd.DataFrame({"sentence1": sentence1, "sentence2": sentence2, "labels": labels, "idx": idx})

    table = config['sql']['table_prefix'] + split
    try:
        print("creating table")
        print("table name:", table)
        df.to_sql(table, con=engine, if_exists='replace', index=False, method='multi', chunksize=100)
    except Exception as e:
        print(e)
        print("failed")

    add_clustered_index(table, engine)

    test_table(table, engine)
