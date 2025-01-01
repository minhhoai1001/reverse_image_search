import numpy as np
from PIL import Image
from numpy.linalg import norm
from abc import ABC, abstractmethod
from transformers import AutoImageProcessor, AutoModel

class HuggingFaceEngine(ABC):
    def __init__(self, model_name=''):
        self.processor  = AutoImageProcessor.from_pretrained(model_name)
        self.model      = AutoModel.from_pretrained(model_name)

    @abstractmethod
    def create(self, input):
        pass

class DinoV2(HuggingFaceEngine):
    def __init__(self, model_name='facebook/dinov2-base'):
        super(DinoV2, self).__init__(model_name)
    
    def create(self, input):
        image   = Image.open(input)
        inputs  = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state
        embeddings = embeddings.mean(dim=1)
        vector = embeddings.detach().cpu().numpy()
        return vector[0]

class HuggingFaceFactory():
    @staticmethod
    def get_engine(model_type: str) -> HuggingFaceEngine:
        models = {
            "dinov2-base": DinoV2(),
        }
        return models.get(model_type.lower(), None)

if __name__ == "__main__":
    factory = HuggingFaceFactory()
    engine = factory.get_engine("dinov2-base")
    if engine:
        input = "/home/nextai_2/vlm/images.jpg"
        data = engine.create(input)
        print("Size vector :", len(data))