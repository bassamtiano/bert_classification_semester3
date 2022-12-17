import argparse

from utils.preprocessor_class import PreprocessorClass
from models.multi_class_model import MultiClassModel

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

def collect_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1)

    parser.add_argument("--max_length", type=int,  default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=10)

    parser.add_argument("--preprocessed_dir", type=str, default="data/preprocessed")
    parser.add_argument("--train_data_dir", type=str, default="data/training.res")
    parser.add_argument("--test_data_dir", type=str, default="data/testing.res")

    return parser.parse_args()
    

if __name__ == '__main__':
    args = collect_parser()

    print(args.batch_size)

    dm = PreprocessorClass(preprocessed_dir = args.preprocessed_dir,
                           train_data_dir = args.train_data_dir,
                           test_data_dir = args.test_data_dir,
                           batch_size = args.batch_size,
                           max_length = args.max_length)

    # Learning rate diganti 1e-3 ke 1e-5
    model = MultiClassModel(
        n_out = 5,
        dropout = 0.3,
        lr = 1e-5
    )

    logger = TensorBoardLogger("logs", name="bert-multi-class")

    trainer = pl.Trainer(
        accelerator = args.accelerator,
        num_nodes = args.num_nodes,
        max_epochs = args.max_epochs,
        default_root_dir = "checkpoints/class",
        logger = logger
    )

    trainer.fit(model, datamodule = dm)
    hasil = trainer.predict(model = model, datamodule = dm)


