import argparse
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

def delete_tables(config):
    engine = create_sql_engine(config)
    for split in ['train', 'validation', 'test']:
        table = config['sql']['table_prefix'] + split
        try:
            print("Dropping table")
            print("table name:", table)
            # drop table
            stmt = "DROP TABLE IF EXISTS %s" % (table)
            _ = engine.execute(stmt)

        except Exception as e:
            print(e)
            print("failed")

    engine.dispose()

def delete_db(config):
    engine = create_sql_engine(config)
    db_name = config['sql']['database']

    try:
        print("Dropping database")
        print("database name:", db_name)
        # drop table
        stmt = "DROP DATABASE IF EXISTS [%s]" % (db_name)
        _ = engine.execute(stmt)

    except Exception as e:
        print(e)
        print("failed")
    engine.dispose()


def main(args):
    with open("config.json", "r") as f:
        config = json.load(f)

    if args.db:
        delete_db(config)
    elif args.tables:
        delete_tables(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-db", help="delete the whole database?", action="store_true")
    parser.add_argument("-tables", help="delete tables only?", action="store_true")
    parser.add_argument("--config-file", default="config.json", help="config file with SQL configuration parameters")
    args = parser.parse_args()
    main(args)