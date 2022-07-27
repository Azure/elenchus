from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
import pyodbc
from sqlalchemy import create_engine
from sqlalchemy.exc import InterfaceError, OperationalError, TimeoutError, DBAPIError
import urllib
import json
import time
import torch

class SQLDataset(IterableDataset):
    def __init__(self, args, split=""):
        self.args = args

        with open(args.config_file, "r") as f:
            config = json.load(f)

        self.table = config['sql']['table_prefix'] + split


        self.batch_size = args.batch_size
        self.batches_per_worker = args.batches_per_worker
        self.num_workers = args.num_workers

        self.len = None

        self.init_sql_engine(config)
        self.tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    def init_sql_engine(self, config):
        conn = f"""Driver={config['sql']['driver']};Server=tcp:{config['sql']['server']},1433;Database={config['sql']['database']};
        Uid={config['sql']['username']};Pwd={config['sql']['password']};Encrypt=yes;TrustServerCertificate=yes;Connection Timeout=1;"""

        params = urllib.parse.quote_plus(conn)
        conn_str = 'mssql+pyodbc:///?autocommit=true&odbc_connect={}'.format(params)
        self.sql_engine = create_engine(conn_str,echo=False,echo_pool=False,fast_executemany=True,pool_size=1,pool_pre_ping=True)
        self.sql_engine.dispose(close=True)
        
    def execute_sql_query(self, stmt, max_attempts=100, nrows='all', engine=None):
        if engine == None:
            engine = self.sql_engine

        for attempt in range(max_attempts):
            try:
                res = engine.execute(stmt)
                if nrows == 'one':
                    rows = res.fetchone()
                    return rows
                elif nrows == 'all':
                    rows = res.fetchall()
                    return rows
                else:
                    return
            except InterfaceError as e:
                raise e
            except OperationalError as e:
                pass
            except TimeoutError as e:
                pass
            except DBAPIError as e:
                pass

            delay = (attempt + 1) * .2
            print("Retrying in %f seconds" % delay)
            time.sleep(delay)


        if attempt == max_attempts:
            print("SQL query failed")
            raise e

    def __len__(self):
        if self.len == None:
            stmt = "SELECT MAX(idx) FROM %s" % self.table
            self.len = self.execute_sql_query(stmt)[0][0]
            
        return self.len

    def load_data(self, indices):
        stmt = "SELECT * FROM %s WHERE idx IN (%s)" % (self.table, indices)

        rows = self.execute_sql_query(stmt)
        return rows
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            partition_size = len(self)
        else:
            worker_id = worker_info.id
            partition_size = len(self) // self.args.num_workers
        perm = torch.randperm(partition_size) + worker_id * partition_size

        n_batches = min(self.batches_per_worker, partition_size // self.args.batch_size)

        for batch in range(n_batches):
            sample_indices_t = perm[self.batch_size * batch:self.batch_size * (batch + 1)]
            sampled_indices = ",".join("{0}".format(n) for n in sample_indices_t)
            samples = self.load_data(sampled_indices)

            for sample in samples:
                tokenized = self.tokenize_function(sample)
                tokenized['labels'] = sample[2]
                yield tokenized

    def tokenize_function(self, example):
        return self.tokenizer(example[0], example[1], truncation=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="data", help="path to to dataset root")
    parser.add_argument("--split", default="validation", help="dataset split (train, validation, test)")
    parser.add_argument("--config-file", default="config.json", help="config file with SQL configuration parameters")
    parser.add_argument("--checkpoint", type=str, default="bert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches-per-worker", type=int, default=100)
    args = parser.parse_args()

    dataset = SQLDataset(args, split=args.split)

    print("Dasaset size:", len(dataset))
