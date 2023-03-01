import random

import torch
import torch.nn as nn

import pytorch_lightning as pl

from transformers import BertModel
from sklearn.metrics import classification_report

from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, PrecisionRecallCurve

class MultiClassModel(pl.LightningModule):
    def __init__(self,
                 dropout,
                 n_out,
                 lr) -> None:
        super(MultiClassModel, self).__init__()

        torch.manual_seed(1)
        random.seed(1)

        self.num_classes = n_out

        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased')
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(dropout)

        # jumlah label = 5
        self.classifier = nn.Linear(768, n_out)

        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()

        self.accuracy = MulticlassAccuracy(task="multiclass", num_classes = self.num_classes)
        self.f1 = MulticlassF1Score(task = "multiclass", 
                          average = "micro",
                          num_classes = self.num_classes)
        # self.precission_recall = PrecisionRecallCurve(task = "multiclass", num_classes = self.num_classes)

    # Model
    def forward(self, input_ids):
        bert_out = self.bert(input_ids = input_ids)
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
        x_input_ids, y = train_batch
        
        out = self(input_ids = x_input_ids)
        # ke tiga parameter di input dan diolah oleh method / function forward
        f1_score = self.f1(out, y.float())
        loss = self.criterion(out, target = y.float())

        # pred = out
        # true = y

        # acc = self.accuracy(out, y)
        
        # precission, recall, _ = self.precission_recall(out, y)
        # report = classification_report(true, pred, output_dict = True, zero_division = 0)

        # self.log("accuracy", acc, prog_bar = True)
        self.log("f1_score", f1_score, prog_bar = True)
        self.log("loss", loss)

        # return {"loss": loss, "predictions": out, "F1": f1_score, "labels": y}
        return {"loss": loss}

    def validation_step(self, valid_batch, batch_idx):
        x_input_ids, y = valid_batch
        
        out = self(input_ids = x_input_ids)
        # ke tiga parameter di input dan diolah oleh method / function forward

        loss = self.criterion(out, target = y.float())

        # pred = out
        # true = y

        # report = classification_report(true, pred, output_dict = True, zero_division = 0)
        # acc = self.accuracy(out, y)
        f1_score = self.f1(out, y)
        
        self.log("f1_score", f1_score, prog_bar = True)
        # self.log("accuracy", acc, prog_bar = True)
        self.log("loss", loss)

        return loss
    
    def predict_step(self, pred_batch, batch_idx):
        x_input_ids, y = pred_batch
        
        out = self(input_ids = x_input_ids)
        # ke tiga parameter di input dan diolah oleh method / function forward
        pred = out
        true = y

        return {"predictions": pred, "labels": true}

    # def training_epoch_end(self, outputs):
    #     labels = []
    #     predictions = []

    #     for output in outputs:
    #         for out_lbl in output["labels"].detach().cpu():
    #             labels.append(out_lbl)
    #         for out_pred in output["predictions"].detach().cpu():
    #             predictions.append(out_pred)

    #     labels = torch.stack(labels).int()
    #     predictions = torch.stack(predictions)

    #     # Hitung akurasi
        
    #     # accuracy = Accuracy(task = "multiclass", num_classes = self.num_classes)
    #     acc = self.accuracy(predictions, labels)
    #     f1_score = self.f1(predictions, labels)
    #     # Print Akurasinya
    #     print("Overall Training Accuracy : ", acc , "| F1 Score : ", f1_score)

    # def on_predict_epoch_end(self, outputs):
    #     labels = []
    #     predictions = []

    #     for output in outputs:
    #         # print(output[0]["predictions"][0])
    #         # print(len(output))
    #         # break
    #         for out in output:
    #             for out_lbl in out["labels"].detach().cpu():
    #                 labels.append(out_lbl)
    #             for out_pred in out["predictions"].detach().cpu():
    #                 predictions.append(out_pred)

    #     labels = torch.stack(labels).int()
    #     predictions = torch.stack(predictions)
        
    #     acc = self.accuracy(predictions, labels)
    #     f1_score = self.f1(predictions, labels)
#     print("Overall Testing Accuracy : ", acc , "| F1 Score : ", f1_score)