
# 1. Import the required packages
import torch
import gradio as gr 

from typing import Dict
from transformers import pipeline

# 2. Define our function to use with our model.

def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

DEVICE = set_device()

# 1. Create a function to take a String input 

def food_not_food_classifier(text: str) -> Dict[str, float]:
    # Setup food not food text classifier
    food_not_food_classifier_pipeline = pipeline(task="text-classification",
                                        model="mdarefin/learn_hf_food_not_food_text_classifier-distilbert-base-uncased",
                                        batch_size=32,
                                        device=DEVICE,
                                        top_k=None) # top_k = None => Return all possible labels
    
    # Get the outputs from our pipeline
    outputs = food_not_food_classifier_pipeline(text)[0]

    # Format output from Gradio

    output_dict = {}
    for item in outputs:
        output_dict[item["label"]] = item["score"]

    return output_dict

# 3. Create a Gradio interface with details about our app
description = """
A text classifier to determine if a sentence is about food or not food. 

Fine-tuned from [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) on a [small dataset of food and not food text](https://huggingface.co/datasets/mrdbourke/learn_hf_food_not_food_image_captions).

See [source code](https://github.com/Adnan-edu/hugging_custom_ai_model).
"""

demo = gr.Interface(fn=food_not_food_classifier, 
             inputs="text", 
             outputs=gr.Label(num_top_classes=2), # show top 2 classes
             title="🍗🚫🥑 Food or Not Food Text Classifier",
             description=description,
             examples=[["I whipped up a fresh batch of code, but it seems to have a syntax error."],
                       ["A delicious photo of a plate of scrambled eggs, bacon and toast."]])

# 4. Launch the interface
if __name__ == "__main__":
    demo.launch()
