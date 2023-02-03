import logging
from transformers import pipeline
import torch 

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def init():
    global model 

    device = 0 if torch.cuda.is_available() else -1
    model = pipeline("fill-mask", model='bert-base-uncased', device=device)

def inference(model_inputs: dict) -> dict:
    global model 

    prompt = model_inputs.get("prompt", None)
    logging.info(prompt)
    if prompt is  None:
        return {
            "msg": "No prompt provided"
        }

    result = model(prompt)
    logging.info(result)
    
    return result