from transformers import pipeline
import torch 

def init():
    global model 

    device = 0 if torch.cuda.is_available() else -1
    model = pipeline("fill-mask", model='bert-base-uncased', device=device)

def inference(model_inputs: dict) -> dict:
    global model 

    prompt = model_inputs.get("prompt", None)
    if prompt ==  None:
        return {
            "msg": "No prompt provided"
        }

    result = model(prompt)
    
    return result