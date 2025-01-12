import uuid
import gradio as gr
from adapter.fastAPI import fastAPIAdapter
from adapter.vllm import vLLMFactory
from adapter.qdrant import QdrantAdapter

fastAPI = fastAPIAdapter()
factory = vLLMFactory()
vllm = factory.get_engine("qwen2_vl")
qdrant_client = QdrantAdapter()
qdrant_client.create_collection("hybird_embeded", 1024, hybird=True)
qdrant_client.create_collection('image_embedded', 768)

def process_inputs(image, text):
    description = vllm.create(image)
    img_embed = fastAPI.image_embeddeding(image)
    
    if description:
        if text:
            description = description +"; " + text
        
        vector = fastAPI.hybird_embeddeding(description)
        if vector and img_embed:
            id = str(uuid.uuid4())
            payload = {"text": description}
            qdrant_client.upsert_points("hybird_embeded", id, payload, vector)
            payload = {"text": text}
            qdrant_client.upsert_points("image_embedded", id, payload, img_embed)
        return description
    else:
        return "Cannot connect to vLLM server"

def toggle_button_state(image):
    # Enable the button only if an image is uploaded
    return gr.update(interactive=bool(image))

# Create the Gradio app
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            # Image input on the left
            image_input = gr.Image(type="pil", label="Upload an Image")
        with gr.Column():
            # Text input on the right
            text_input = gr.Textbox(label="Enter Text information")

    # Outputs
    output_text = gr.Textbox(label="Image Description")

    # Button to trigger processing
    process_button = gr.Button("Process", interactive=False)
    
    # Update button state based on image input
    image_input.change(
        toggle_button_state,
        inputs=[image_input],
        outputs=[process_button],
    )

    # Define the button action
    process_button.click(
        process_inputs,
        inputs=[image_input, text_input],
        outputs=[output_text]
    )

# Launch the app
demo.launch(server_port=8111)
