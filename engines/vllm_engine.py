import os
import base64
from openai import OpenAI
from abc import ABC, abstractmethod

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
    def create(self, input):
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
                        "image_url": {"url": f"data:image/jpeg;base64,{input}"},
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