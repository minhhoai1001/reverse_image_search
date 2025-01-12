import requests, io, base64
from pathlib import Path
from PIL import Image
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
    
    def is_path(string: str) -> bool:
        try:
            # Check if the string is a valid file path
            path = Path(string)
            # Heuristic: Paths often contain separators or extensions
            if path.is_absolute() or any(part in string for part in ('/', '\\', '.')):
                return True
            return False
        except Exception:
            return False
    
    def create_from_image(self, image: Image.Image):
        return image

    def create_from_string(self, input_string: str):
        # Check if the input is a URL or a file path
        if input_string.startswith("http://") or input_string.startswith("https://"):
            # If it's a URL, fetch the image content
            response = requests.get(input_string)
            image = Image.open(io.BytesIO(response.content))
        else:
            # Otherwise, assume it's a file path
            image = Image.open(input_string)
        
        return image

    def create_from_base64(self, input_base64: bytes):
        # Handle base64 input (assuming input is already base64-encoded)
        image_data = base64.b64decode(input_base64)
        image = Image.open(io.BytesIO(image_data))
        return image

    def create(self, input):
        if isinstance(input, Image.Image):
            image = self.create_from_image(input)
        elif isinstance(input, str):
            if self.is_path(input):
                image = self.create_from_string(input)
            else:
                raise ValueError("Unsupported input type")
                
        elif isinstance(input, bytes):
            image = self.create_from_base64(input)
        else:
            raise ValueError("Unsupported input type")

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