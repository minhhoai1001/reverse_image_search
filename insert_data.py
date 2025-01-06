import sys, yaml, uuid, base64, os
from tqdm import tqdm
from PIL import Image
from engines.hf_engine import HuggingFaceFactory
from engines.vllm_engine import vLLMFactory
from engines.qdrant_engine import QdrantEngine

import numpy as np
from numpy.linalg import norm

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(img_path, config):
    host = f"{config['qdrant']['host']}:{config['qdrant']['port']}"
    qdrant_client = QdrantEngine(host)
    qdrant_client.create_collection(config['qdrant']['text_collection'], 768)
    qdrant_client.create_collection(config['qdrant']['image_collection'], 768)

    vllm = vLLMFactory()
    hf = HuggingFaceFactory()
    engine_text = vllm.get_engine(config['model']['text'])
    engine_img = vllm.get_engine(config['model']['vlm'])
    engine_hf = hf.get_engine(config['model']['img'])

    if engine_text and engine_img:
        paths = os.listdir(img_path)
        for path in tqdm(paths):
            image_path = os.path.join(img_path, path)
            image = Image.open(image_path)
            text = engine_img.create(image)
            data = engine_text.create(text)

            payload = {"text": text}
            id = str(uuid.uuid4())
            qdrant_client.upsert_points("text_embedded", id, payload, data)

            embedding = engine_hf.create(image)
            payload = {}
            qdrant_client.upsert_points("image_embedded", id, payload, embedding)
            
            image.save(f"./data/{id}.jpg")
    else:
        print("Cannot connect vLLM engine !")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "imgs"
    config = read_config("./config.yaml")
    print(config)
    main(img_path, config)