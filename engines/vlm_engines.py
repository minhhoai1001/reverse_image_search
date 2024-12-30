import os
import base64
from openai import OpenAI
from abc import ABC, abstractmethod

# Abstract Base Class for VLM Engines
class VLMEngine(ABC):
    @abstractmethod
    def create(self, base64_image: str) -> str:
        pass

# Concrete Class for vllm_Qwen2_vl
class VLLMQwen2VL(VLMEngine):
    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:8080/v1",
            api_key="EMPTY",
        )
    
    def create(self, base64_image: str) -> str:
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

# Factory Class to Create VLM Engine Instances
class VLMEngineFactory:
    @staticmethod
    def get_engine(engine_type: str) -> VLMEngine:
        if engine_type == "vllm_qwen2_vl":
            return VLLMQwen2VL()
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")

# Example Usage
if __name__ == "__main__":
    # Path to the local image
    image_path = "/home/nextai_2/vlm/images.jpg"

    # Convert the image to a Base64 string
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    # Create engine instance using factory
    engine = VLMEngineFactory.get_engine("vllm_qwen2_vl")
    
    # Use the engine to create a description
    result = engine.create(base64_image)
    print("Result:", result)
