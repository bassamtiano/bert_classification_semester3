import re

import torch
from torch.utils.data import TensorDataset
import pandas as pd

import pytorch_lightning as pl

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from transformers import BertTokenizer


class PreprocessorHatespeech(pl.LightningDataModule):
    def __init__(self, max_length = 100, batch_size = 30) -> None:
        super(PreprocessorHatespeech, self).__init__()
        self.tokenizers = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')
        self.max_length = max_length
        self.batch_size = batch_size

        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

    def clean_str(self, string):
        string = string.lower()
        string = re.sub(r"[^A-Za-z0-9(),!?\'\-`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\n", "", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = string.strip()
        # Menghilangkan imbuhan
        return self.stemmer.stem(string)

    def load_data(self):
        data = pd.read_csv("data/re_dataset.csv", encoding="latin-1")
        data = data.dropna(how = "any")

        # Ambil nama kolom
        self.hspc_label = list(data.keys())[1:]
        
        # Buat Label ke Id untuk variable Y
        self.label2id = {}
        for i_hspc, k_hspc in enumerate(self.hspc_label):
            self.label2id[k_hspc] = i_hspc

        # Mengambil id baris yang tidak memiliki label
        condition_empty_label = data[
            (
                (data['HS'] == 0) & 
                (data['Abusive'] == 0) &
                (data['HS_Individual'] == 0) &
                (data['HS_Group'] == 0) & 
                (data['HS_Religion'] == 0) & 
                (data['HS_Race'] == 0) & 
                (data['HS_Physical'] == 0) & 
                (data['HS_Gender'] == 0) &
                (data['HS_Other'] == 0) &
                (data['HS_Weak'] == 0) &
                (data['HS_Moderate'] == 0) &
                (data['HS_Strong'] == 0)
            )
        ].index

        data = data.drop(condition_empty_label)

        tweets = data["Tweet"].apply(lambda x: self.clean_str(x))
        tweets = tweets.values.tolist()
        
        labels = data.drop(["Tweet"], axis = 1)
        labels = labels.values.tolist()
        
        x_input_ids = []

        for tw in tweets:
            ids_tweet = self.tokenizers(text = tw,
                                        max_length = self.max_length,
                                        padding = "max_length",
                                        truncation = True)
            x_input_ids.append(ids_tweet)

        x_input_ids = torch.tensor(x_input_ids)
        y = torch.tensor(labels)

        tensor_dataset = TensorDataset(x_input_ids, y)

        # 1. Pisahkan Train & Validation (80%) dan Test (20%) dari tensor_dataset
        # 2. Pisahkan Train (90%) dan Validation (10%) dari Train & Validation di tahap 1
        # 3. Di dapatkan dataset train, validation, dan test
        


if __name__ == '__main__':
    pre  = PreprocessorHatespeech()
    pre.load_data()