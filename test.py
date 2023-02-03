import banana_dev as banana
import json
import os 
import dotenv

dotenv.load_dotenv(".env")

api_key = os.environ.get("BANANA_DEV_API_KEY", None)
model_key = os.environ.get("BERT_DEV_TEST_MODEL_KEY", None)

payload = {
    "prompt": "Paris is the [MASK] of France."
}

out = banana.run(api_key, model_key, payload)
print(out)
