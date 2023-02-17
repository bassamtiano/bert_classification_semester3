import torch

from datasets import load_dataset

import pytorch_lightning as pl

# https://huggingface.co/datasets/indonlp/indonlu

from transformers import BertTokenizer

from torch.utils.data import TensorDataset, DataLoader

class PreprocessorIndoNLU(pl.LightningDataModule):
    def __init__(self,
                 max_length,
                 n_classes,
                 batch_size):
        super(PreprocessorIndoNLU, self).__init__()
        # Superclass pytorch lightning data module wajib
        self.tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')
        self.max_length = max_length
        self.n_classes = n_classes
        self.batch_size = batch_size
        
    
    def load_data(self, dataset_type = "train"):
        indonlu_data = load_dataset("indonlp/indonlu", "emot", split = dataset_type)
        
        # Tipe data array
        # List, Dictionary, Set, Tuple
        
        # x = kalimat, y = label
        x_all, y_all = [], []
        for dt in indonlu_data:
            x = dt["tweet"]
            y = dt["label"]
            
            # max_length = Maximum panjang (jumlah kata) dalam kalimat
            # padding = menambahkan size (jumlah kata) dalam kalimat sesuai dengan size max length
            # truncation = pemotongan kata agar sama dengan max length
            x = self.tokenizer(text = x,
                               max_length = self.max_length,
                               padding = "max_length",
                               truncation = True)["input_ids"]
            x_all.append(x)
            
            # y = 3
            # y = [0, 0, 0, 1, 0]
            y_bin = [0] * self.n_classes
            y_bin[y] = 1
            y = y_bin
            y_all.append(y)
        
        # Menjadikan List menjadi tensor
        x_tensor = torch.tensor(x_all)
        y_tensor = torch.tensor(y_all)
        
        # tensor di bungkus menjadi tensor dataset
        tensor_dataset = TensorDataset(x_tensor, y_tensor)
        return tensor_dataset
    
    def setup(self, stage = None):
        train_data = self.load_data(dataset_type = "train")
        valid_data = self.load_data(dataset_type = "validation")
        test_data = self.load_data(dataset_type = "test")
        
        if stage == "fit":
            self.train_data = train_data
            self.valid_data = valid_data
        elif stage == "test":
            self.test_data = test_data
    
    def train_dataloader(self):
        return DataLoader(
            dataset = self.train_data,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 4
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset = self.valid_data,
            batch_size = self.batch_size,
            num_workers = 4
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset = self.test_data,
            batch_size = self.batch_size,
            num_workers = 4
        )
        
if __name__ == '__main__':
    pre = PreprocessorIndoNLU(max_length = 100,
                              n_classes = 5,
                              batch_size = 40)
    
    pre.setup("fit")
    print(pre.train_dataloader())
    print(pre.val_dataloader())