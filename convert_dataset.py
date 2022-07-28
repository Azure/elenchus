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

raw_datasets = load_dataset("glue", config['data']['glue_subset'])

os.makedirs('data', exist_ok=True)

engine = create_sql_engine(config)

for split in raw_datasets.keys():
    df_dict = {}
    for column_name in raw_datasets.column_names[split]:
        df_dict[column_name] = []
    for i, example in enumerate(raw_datasets[split]):
        for column_name in raw_datasets.column_names[split]:
            df_dict[column_name].append(example[column_name])
    df = pd.DataFrame(df_dict)

    table = config['data']['glue_subset'] + split
    try:
        print("creating table")
        print("table name:", table)
        df.to_sql(table, con=engine, if_exists='replace', index=False, method='multi', chunksize=100)
    except Exception as e:
        print(e)
        print("failed")

    add_clustered_index(table, engine)

    test_table(table, engine)

    engine.dispose()