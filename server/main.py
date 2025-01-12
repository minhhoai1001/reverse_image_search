import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel

from engines.hybird_engine import HybirdFactory
from engines.hf_engine import HuggingFaceFactory

app = FastAPI()
hybird_factory = HybirdFactory()
hf_factory = HuggingFaceFactory()

hybird_engine   = hybird_factory.get_engine("bge-m3")
img_engine      = hf_factory.get_engine("dinov2-base")

class HybirdRequest(BaseModel):
    text: str

class ImageModel(BaseModel):
    base64: bytes


@app.get("/health")
async def root():
    return {"message": "Hello World"}

@app.post("/api/hybird_embeddeding")
async def hybird_embeddeding(request: HybirdRequest):
    text = request.text
    dense, sparse = hybird_engine.create(text)
    sparse = {
        "indices": [int(key) for key in sparse.keys()],
        "values": [float(value) for value in sparse.values()],
    }
    response = {
        "dense": dense.tolist(),
        "sparse": sparse
    }

    return {"embeddings": response}


@app.post("/api/image_embeddeding")
async def hybird_embeddeding(request: ImageModel):
    img = request.base64
    embed = img_engine.create(img)

    return {"embeddings": embed.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)