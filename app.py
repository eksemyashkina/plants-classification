from typing import Dict
import gradio as gr
import json
import PIL.Image, PIL.ImageOps
import torch
import torchvision.transforms.functional as F

from src.models.resnet50 import ResNet
from src.models.mobilenet_v2 import MobileNetV2


num_classes = 30
model1 = ResNet(weights_path="weights/checkpoint-best-resnet.pth", num_classes=num_classes)
model1.eval()
model2 = MobileNetV2(weights_path="weights/checkpoint-best-mobilenet.pth", num_classes=num_classes)
model2.eval()


with open("labels.json", "r") as f:
    class_labels = json.load(f)
label_mapping = {v: k for k, v in class_labels.items()}


def predict(img, model_choice) -> Dict[str, float]:
    model = model1 if model_choice == "ResNet" else model2
    width, height = img.size
    max_dim = max(width, height)
    padding = (max_dim - width, max_dim - height)
    img = PIL.ImageOps.expand(img, padding, (255, 255, 255))
    img = img.resize((224, 224))
    img = F.to_tensor(img)
    img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = img.unsqueeze(0)

    with torch.inference_mode():
        logits = model.forward(img)
        probs = torch.nn.functional.softmax(logits, dim=1)
        top_probs, top_indices = probs[0].topk(3)

    top_classes = {label_mapping[idx.item()]: prob.item() for idx, prob in zip(top_indices, top_probs)}
    return top_classes


examples = [
    ["assets/banana.jpg"],
    ["assets/pineapple.jpg"],
    ["assets/mango.jpg"],
    ["assets/melon.jpg"],
    ["assets/orange.jpg"],
    ["assets/eggplant.jpg"],
    ["assets/black.jpg"],
    ["assets/white.jpg"]
]


with gr.Blocks() as demo:
    gr.Markdown("## Plant Classification")
    with gr.Row():
        with gr.Column():
            pic = gr.Image(label="Upload Plant Image", type="pil", height=300, width=300)
            model_choice = gr.Dropdown(choices=["ResNet", "MobileNetV2"], label="Select Model", value="ResNet")
            with gr.Row():
                with gr.Column(scale=1):
                    predict_btn = gr.Button("Predict")
                with gr.Column(scale=1):
                    clear_btn = gr.Button("Clear")
        
        with gr.Column():
            output = gr.Label(label="Top 3 Predicted Classes")

    predict_btn.click(fn=predict, inputs=[pic, model_choice], outputs=output, api_name="predict")
    clear_btn.click(lambda: (None, None), outputs=[pic, output])
    gr.Examples(examples=examples, inputs=[pic])

demo.launch()
