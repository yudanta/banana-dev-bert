import torch
from transformers import pipeline

from decorators import logger, timeit


@timeit
def init():
    logger.info("initalizing the model...")
    global model

    device = 0 if torch.cuda.is_available() else -1
    model = pipeline("fill-mask", model="bert-base-uncased", device=device)
    logger.info("initializing model done!")


@timeit
def inference(model_inputs: dict) -> dict:
    global model

    prompt = model_inputs.get("prompt", None)
    if prompt is None:
        return {"msg": "No prompt provided"}

    result = model(prompt)

    return result
