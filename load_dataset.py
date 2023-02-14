from datasets import load_dataset

# https://huggingface.co/datasets/indonlp/indonlu

class PreprocessorIndoNLU():
    
    
    def load_data(self, dataset_type = "train"):
        indonlu_data = load_dataset("indonlp/indonlu", "emot", split = dataset_type)
        
        
if __name__ == "__main__":
    
    pre = PreprocessorIndoNLU()
    
    pre.load_data()