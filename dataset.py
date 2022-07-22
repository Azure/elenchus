from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import os

class MyDataset(Dataset):
    def __init__(self, args, split):
        

        df = pd.read_pickle(os.path.join(args.dataset_root, split + ".pkl.gz"))

        self.data = []
        for row in df.itertuples():
            self.data.append({"sentence1": row[1], "sentence2": row[2], "label": row[3], "idx": row[4]})

        #     print()
        # sentence1 = [
        #     'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
        #     "Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion .",
        #     "They had published an advertisement on the Internet on June 10 , offering the cargo for sale , he added .",
        #     "Around 0335 GMT , Tab shares were up 19 cents , or 4.4 % , at A $ 4.56 , having earlier set a record high of A $ 4.57 .",
        #     "The stock rose $ 2.11 , or about 11 percent , to close Friday at $ 21.51 on the New York Stock Exchange .",
        #     "Revenue in the first quarter of the year dropped 15 percent from the same period a year earlier .",
        #     "The Nasdaq had a weekly gain of 17.27 , or 1.2 percent , closing at 1,520.15 on Friday .",
        #     "The DVD-CCA then appealed to the state Supreme Court .",
        #     "That compared with $ 35.18 million , or 24 cents per share , in the year-ago period ."
        # ]
        # sentence2 = [
        #     'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .',
        #     "Yucaipa bought Dominick 's in 1995 for $ 693 million and sold it to Safeway for $ 1.8 billion in 1998 .",
        #     "On June 10 , the ship 's owners had published an advertisement on the Internet , offering the explosives for sale .",
        #     "Tab shares jumped 20 cents , or 4.6 % , to set a record closing high at A $ 4.57 .",
        #     "PG & E Corp. shares jumped $ 1.63 or 8 percent to $ 21.03 on the New York Stock Exchange on Friday .",
        #     "With the scandal hanging over Stewart 's company , revenue the first quarter of the year dropped 15 percent from the same period a year earlier .",
        #     "The tech-laced Nasdaq Composite .IXIC rallied 30.46 points , or 2.04 percent , to 1,520.15 .",
        #     "The DVD CCA appealed that decision to the U.S. Supreme Court .",
        #     "Earnings were affected by a non-recurring $ 8 million tax benefit in the year-ago period ."
        # ]
        # label = [1, 0, 1, 0, 1, 1, 0, 1, 0]
        # idx = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        
        # self.data = []
        # for s1, s2, l, i in zip(sentence1, sentence2, label, idx):
        #     self.data.append({"sentence1": s1, "sentence2": s2, "label": l, "idx": i})

        self.tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokenized = self.tokenize_function(self.data[idx]) # , 
        tokenized['labels'] = self.data[idx]["label"]
        return tokenized

    def tokenize_function(self, example):
        return self.tokenizer(example["sentence1"], example["sentence2"], truncation=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="data", help="path to to dataset root")
    parser.add_argument("--split", default="validation", help="dataset split (train, validation, test)")
    args = parser.parse_args()

    dataset = MyDataset(args)