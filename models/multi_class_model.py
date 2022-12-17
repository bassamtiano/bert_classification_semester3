import random

import torch
import torch.nn as nn

import pytorch_lightning as pl

from transformers import BertModel
from sklearn.metrics import classification_report

from torchmetrics import Accuracy

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

        self.accuracy = Accuracy(task="multiclass")

    
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

        # pred = out.argmax(1).cpu()
        # true = y.argmax(1).cpu()

        self.accuracy(out, y)
        # report = classification_report(true, pred, output_dict = True, zero_division = 0)

        self.log("accuracy", self.accuracy, prog_bar = True)
        self.log("loss", loss)

        return {"loss": loss, "predictions": out, "labels": y}

    def validation_step(self, valid_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = valid_batch
        
        out = self(input_ids = x_input_ids,
                   attention_mask = x_attention_mask,
                   token_type_ids = x_token_type_ids)
        # ke tiga parameter di input dan diolah oleh method / function forward

        loss = self.criterion(out, target = y.float())

        # pred = out.argmax(1).cpu()
        # true = y.argmax(1).cpu()

        # report = classification_report(true, pred, output_dict = True, zero_division = 0)
        self.accuracy(out, y)

        self.log("accuracy", self.accuracy, prog_bar = True)
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

        return {"predictions": pred, "labels": true}

    def training_epoch_end(self, outputs):
        labels = []
        predictions = []

        for output in outputs:
            for out_lbl in output["labels"].detach().cpu():
                labels.append(out_lbl)
            for out_pred in output["predictions"].detach().cpu():
                predictions.append(out_pred)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        # Hitung akurasi
        accuracy = Accuracy(task = "multiclass")
        acc = accuracy(predictions, labels)

        # Print Akurasinya
        print("Overall Training Accuracy : ", acc)

    def on_predict_epoch_end(self, outputs):
        labels = []
        predictions = []

        for output in outputs:
            # print(output[0]["predictions"][0])
            # print(len(output))
            # break
            for out in output:
                for out_lbl in out["labels"].detach().cpu():
                    labels.append(out_lbl)
                for out_pred in out["predictions"].detach().cpu():
                    predictions.append(out_pred)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        accuracy = Accuracy(task = "multiclass")
        acc = accuracy(predictions, labels)
        print("Overall Testing Accuracy : ", acc)