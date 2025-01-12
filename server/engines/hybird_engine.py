from FlagEmbedding import BGEM3FlagModel
from abc import ABC, abstractmethod

class HybirdEngine(ABC):
    @abstractmethod
    def create(self, input):
        pass

class BGEM3Flag(HybirdEngine):
    def __init__(self, model_name='BAAI/bge-m3'):
        self.model = BGEM3FlagModel(model_name,  use_fp16=True)

    def create(self, input):
        output = self.model.encode(input, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        dense_vecs = output['dense_vecs']
        sparse_vecs = output['lexical_weights']

        return dense_vecs, sparse_vecs
    
class HybirdFactory():
    @staticmethod
    def get_engine(model_type: str) -> HybirdEngine:
        models = {
            "bge-m3": BGEM3Flag('BAAI/bge-m3'),
        }
        return models.get(model_type.lower(), None)

if __name__ == "__main__":
    factory = HybirdFactory()
    engine = factory.get_engine("bge-m3")
    
    text = "Hugging Face is way more fun with friends and colleagues!"
    dense, sparse = engine.create(text)
    print("dense", len(dense))
    print("sparse", len(sparse))