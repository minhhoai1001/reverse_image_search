import os, io
import base64
from PIL import Image
from openai import OpenAI
from abc import ABC, abstractmethod

MAX_WIDTH = 720

class vLLMEngine(ABC):
    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:8080/v1",
            api_key="EMPTY",
        )

    @abstractmethod
    def create(self, input):
        pass

class Qwen2VL(vLLMEngine):
    def create_from_image(self, image: Image.Image):
        # Check if the width is greater than 1280
        if image.width > MAX_WIDTH:
            # Calculate the new height while maintaining the aspect ratio
            ratio = MAX_WIDTH / image.width
            new_width = MAX_WIDTH
            new_height = int(image.height * ratio)
            # Resize the image
            image = image.resize((new_width, new_height), Image.LANCZOS)

        # Convert the PIL image to a base64-encoded string
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return base64_image

    def create_from_string(self, input_string: str):
        # Assume the input string is a file path or URL
        with open(input_string, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        return base64_image

    def create_from_base64(self, input_base64: bytes):
        # Handle base64 input (assuming input is already base64-encoded)
        return input_base64.decode("utf-8")

    def create(self, input):
        if isinstance(input, Image.Image):
            base64_image = self.create_from_image(input)
        elif isinstance(input, str):
            base64_image = self.create_from_string(input)
        elif isinstance(input, bytes):
            base64_image = self.create_from_base64(input)
        else:
            raise ValueError("Unsupported input type")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ]

        try:
            completion = self.client.chat.completions.create(
                model="Qwen/Qwen2-VL-2B-Instruct",
                messages=messages,
                max_tokens=256,
            )

            text = completion.choices[0].message
            return text.content
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

class BGEBASE(vLLMEngine):
    def create(self, input):
        try:
            completion = self.client.embeddings.create(
                model="BAAI/bge-base-en-v1.5",
                input=input,
                encoding_format="float"
            )

            embeded = completion.data[0].embedding
            return embeded
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

class vLLMFactory():
    @staticmethod
    def get_engine(model_type: str) -> vLLMEngine:
        models = {
            "bge-base-en-v1.5": BGEBASE(),
            "qwen2_vl": Qwen2VL(),
        }
        return models.get(model_type.lower(), None)

if __name__ == "__main__":
    factory = vLLMFactory()
    engine1 = factory.get_engine("bge-base-en-v1.5")
    if engine1:
        input1 = "Hugging Face is way more fun with friends and colleagues!"
        data = engine1.create(input1)
        print("Size vector 1:", len(data))
    
    engine2 = factory.get_engine("qwen2_vl")
    if engine2:
        image_path = "/home/nextai_2/vlm/images.jpg"

        # Convert the image to a Base64 string
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        data = engine2.create(base64_image)
        print("Text:", data)