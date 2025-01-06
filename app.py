import gradio as gr

from engines.qdrant_engine import QdrantEngine
from engines.vllm_engine import vLLMFactory
from engines.hf_engine import HuggingFaceFactory

vllm = vLLMFactory()
hf = HuggingFaceFactory()
engine_img = vllm.get_engine("qwen2_vl")
engine_text = vllm.get_engine("bge-base-en-v1.5")
engine_hf = hf.get_engine("dinov2-base")

qdrant_client = QdrantEngine("localhost:6333")

def process_input(input_type, image, text):
    if input_type == "Image" and image is not None:
        # `image` will be a PIL.Image object
        embed = engine_hf.create(image)
        
        info = qdrant_client.search_similar("image_embedded", embed)
        for data in info:
            print(data.id, data.score)
        return f"Size vector {len(embed)}"

    elif input_type == "Text" and text:
        print(text)
        embed = engine_text.create(text)
        info = qdrant_client.search_similar("text_embedded", embed)
        for data in info:
            print(data.id, data.score)
        return f"Size vector {len(embed)}"
    else:
        return "Invalid input or no input provided!"

with gr.Blocks() as demo:
    with gr.Row():
        input_type = gr.Dropdown(choices=["Image", "Text"], label="Input Type", value="Image")
        image_input = gr.Image(label="Upload Image", type="pil", visible=True)
        text_input = gr.Textbox(label="Enter Text", visible=False)

    def toggle_input(choice):
        if choice == "Image":
            return gr.update(visible=True), gr.update(visible=False)
        elif choice == "Text":
            return gr.update(visible=False), gr.update(visible=True)
        return gr.update(visible=False), gr.update(visible=False)

    input_type.change(
        toggle_input, inputs=[input_type], outputs=[image_input, text_input]
    )

    submit_button = gr.Button("Submit")
    output = gr.Textbox(label="Output")

    submit_button.click(
        process_input,
        inputs=[input_type, image_input, text_input],
        outputs=output
    )

demo.launch()
