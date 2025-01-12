import gradio as gr

from adapter.fastAPI import fastAPIAdapter
from adapter.vllm import vLLMFactory
from adapter.qdrant import QdrantAdapter

factory = vLLMFactory()
vllm = factory.get_engine("qwen2_vl")
fastAPI = fastAPIAdapter()
qdrant_client = QdrantAdapter("localhost:6333")

def process_input(input_type, image, text):
    if input_type == "Image" and image is not None:
        # `image` will be a PIL.Image object
        embed = fastAPI.image_embeddeding(image)
        print("====>", len(embed))
        info = qdrant_client.search_similar("image_embedded", embed)
        for data in info:
            print(data.id, data.score)
        return f"Size vector {len(embed)}"

    elif input_type == "Text" and text:
        print(text)
        embed = fastAPI.hybird_embeddeding(text)
        info = qdrant_client.query_points("hybird_embeded", embed, limit=5)
        for data in info:
            print(data.id, data.score)
        return f"Size vector {len(embed)}"
    else:
        return "Invalid input or no input provided!"

with gr.Blocks() as demo:
    with gr.Row():
        input_type = gr.Dropdown(choices=["Text", "Image"], label="Input Type", value="Text")
        image_input = gr.Image(label="Upload Image", type="pil", visible=False)
        text_input = gr.Textbox(label="Enter Text", visible=True)

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
