import enum
import pickle
import torch
import os

import re

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import pytorch_lightning as pl
import pandas as pd

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import BertTokenizer

from tqdm import tqdm

class PreprocessorClass(pl.LightningDataModule):

    def __init__(self, 
                 preprocessed_dir,
                 train_data_dir = "data/training.res",
                 test_data_dir = "data/testing.res",
                 batch_size = 10,
                 max_length = 100):
        
        super(PreprocessorClass, self).__init__()

        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir

        self.label2id = {
            'bola': 0,
            'news': 1,
            'bisnis': 2,
            'tekno': 3,
            'otomotif': 4
        }

        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

        self.tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')
        self.max_length = max_length
        self.preprocessed_dir = preprocessed_dir

        self.batch_size = batch_size

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

    def load_data(self,):
        with open(self.train_data_dir, "rb") as tdr:
            train_pkl = pickle.load(tdr)
            train = pd.DataFrame({'title': train_pkl[0], 'label': train_pkl[1]})
        with open(self.test_data_dir, "rb") as tsdr:
            test_pkl = pickle.load(tsdr)
            test = pd.DataFrame({'title': test_pkl[0], 'label': test_pkl[1]})
        
        # Mengetahui apa saja label yang ada di dalam dataset
        label_yang_ada = train["label"].drop_duplicates()

        # Konversi dari label text (news) ke label id (1)
        train.label = train.label.map(self.label2id)
        test.label = test.label.map(self.label2id)

        return train, test
    
    def arrange_data(self, data, type):
        # Yang di lakukan
        # 1. Cleaning sentence
        # 2. Tokenizing
        # 3. Arrange ke dataset (training, validation, testing)

        x_input_ids, x_token_type_ids, x_attention_mask, y = [], [], [], []
        for baris, dt in enumerate(tqdm(data.values.tolist())):
            title = self.clean_str(dt[0])
            label = dt[1]

            binary_lbl = [0] * len(self.label2id)
            binary_lbl[label] = 1

            tkn = self.tokenizer(text = title,
                                 max_length = self.max_length, 
                                 padding = "max_length",
                                 truncation = True)

            x_input_ids.append(tkn['input_ids'])
            x_token_type_ids.append(tkn['token_type_ids'])
            x_attention_mask.append(tkn['attention_mask'])
            y.append(binary_lbl)
            
            if baris > 10:
                break

        x_input_ids = torch.tensor(x_input_ids)
        x_token_type_ids = torch.tensor(x_token_type_ids)
        x_attention_mask = torch.tensor(x_attention_mask)
        y = torch.tensor(y)

        tensor_dataset = TensorDataset(x_input_ids,
                                       x_token_type_ids,
                                       x_attention_mask,
                                       y)

        if type == "train":
            # Standard split : Train (80%), Validation (20%)
            train_tensor_dataset, valid_tensor_dataset = torch.utils.data.random_split(tensor_dataset, [
                                                            round(len(x_input_ids) * 0.8), 
                                                            len(x_input_ids) - round(len(x_input_ids) * 0.8)
                                                         ])
            
            torch.save(train_tensor_dataset, f"{self.preprocessed_dir}/train.pt")
            torch.save(valid_tensor_dataset, f"{self.preprocessed_dir}/valid.pt")

            return train_tensor_dataset, valid_tensor_dataset
        else:
            torch.save(tensor_dataset, f"{self.preprocessed_dir}/test.pt")
            return tensor_dataset

    def preprocessor(self):
        train, test = self.load_data()

        if not os.path.exists(f"{self.preprocessed_dir}/train.pt") or not os.path.exists(f"{self.preprocessed_dir}/valid.pt"):
            print("Create Train and Validation dataset")
            train_data, valid_data = self.arrange_data(data = train, type = "train")
        else:
            print("Load Preprocessed train and validation data")
            train_data = torch.load(f"{self.preprocessed_dir}/train.pt")
            valid_data = torch.load(f"{self.preprocessed_dir}/valid.pt")

        if not os.path.exists(f"{self.preprocessed_dir}/test.pt"):
            print("Create test dataset")
            test_data = self.arrange_data(data = test, type = "test")
        else:
            print("Load Preprocessed test data")
            test_data = torch.load(f"{self.preprocessed_dir}/test.pt")

        return train_data, valid_data, test_data

    def setup(self, stage = None):
        train_data, valid_data, test_data = self.preprocessor()
        print(valid_data)
        if stage == "fit":
            self.train_data = train_data
            self.valid_data = valid_data
        elif stage == "predict":
            self.test_data = test_data

    def train_dataloader(self):
        sampler = RandomSampler(self.train_data)
        return DataLoader(
            dataset = self.train_data,
            batch_size = self.batch_size,
            sampler = sampler,
            num_workers = 1
        )
    
    def val_dataloader(self):
        sampler = SequentialSampler(self.valid_data)
        return DataLoader(
            dataset = self.valid_data,
            batch_size = self.batch_size,
            sampler = sampler,
            num_workers = 1
        )

    def predict_dataloader(self):
        sampler = SequentialSampler(self.test_data)
        return DataLoader(
            dataset = self.test_data,
            batch_size = self.batch_size,
            sampler = sampler,
            num_workers = 1
        )
    
# if __name__ == '__main__':
#     Pre = PreprocessorClass(preprocessed_dir = "data/preprocessed")
#     Pre.setup(stage = "fit")
#     train_data = Pre.train_dataloader()
    