import base64
import requests
from io import BytesIO
from PIL import Image

class fastAPIAdapter():
    def __init__(self, url="http://localhost:8001"):
        self.url = url

    def hybird_embeddeding(self, text: str):
        payload = {
            "text": text
        }
        response = requests.post(f"{self.url}/api/hybird_embeddeding", json=payload)
        
        if response.status_code == 200:
            embed = response.json()["embeddings"]
            
            return embed
        
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None, None
        
    def image_embeddeding(self, image: Image.Image):
        buffer = BytesIO()
        format = image.format if image.format else "PNG"
        image.save(buffer, format=format)
        buffer.seek(0)
    
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        payload = {
            "base64": img_base64
        }
        response = requests.post(f"{self.url}/api/image_embeddeding", json=payload)
        if response.status_code == 200:
            embed = response.json()["embeddings"]
            
            return embed
        
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None