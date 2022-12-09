import random

import torch
import torch.nn as nn

import pytorch_lightning as pl

from transformers import BertModel
from sklearn.metrics import classification_report

class MultiClassModel(pl.LightningModule):
    def __init__(self,
                 dropout,
                 n_out,
                 lr) -> None:
        super(MultiClassModel, self).__init__()

        torch.manual_seed(1)
        random.seed(1)

        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased')
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(dropout)

        # jumlah label = 5
        self.classifier = nn.Linear(768, n_out)

        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()

    
    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_out = self.bert(input_ids = input_ids,
                             attention_mask = attention_mask,
                             token_type_ids = token_type_ids)
        hidden_state = bert_out[0]
        pooler = hidden_state[:, 0]
        # Outout size (batch size = 30 baris, sequence length = 100 kata / token, hidden_size = 768 tensor jumlah vector representation dari bert)

        pooler = self.pre_classifier(pooler)
        # pre classifier untuk mentransfer weight output ke epoch selanjutnya
        pooler = torch.nn.Tanh()(pooler)
        # kontrol hasil pooler min -1 max 1


        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        # classifier untuk memprojeksikan hasil pooler (768) ke jumlah label (5)

        return output

    def configure_optimizers(self):
        # Proses training lebih cepat
        # Tidak memakan memori berlebih
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = train_batch
        
        out = self(input_ids = x_input_ids,
                   attention_mask = x_attention_mask,
                   token_type_ids = x_token_type_ids)
        # ke tiga parameter di input dan diolah oleh method / function forward

        loss = self.criterion(out, target = y.float())

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()

        report = classification_report(true, pred, output_dict = True, zero_division = 0)

        self.log("accuracy", report["accuracy"], prog_bar = True)
        self.log("loss", loss)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = valid_batch
        
        out = self(input_ids = x_input_ids,
                   attention_mask = x_attention_mask,
                   token_type_ids = x_token_type_ids)
        # ke tiga parameter di input dan diolah oleh method / function forward

        loss = self.criterion(out, target = y.float())

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()

        report = classification_report(true, pred, output_dict = True, zero_division = 0)

        self.log("accuracy", report["accuracy"], prog_bar = True)
        self.log("loss", loss)

        return loss
    
    def predict_step(self, pred_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = pred_batch
        
        out = self(input_ids = x_input_ids,
                   attention_mask = x_attention_mask,
                   token_type_ids = x_token_type_ids)
        # ke tiga parameter di input dan diolah oleh method / function forward
        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()

        return pred, true
