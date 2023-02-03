from fastapi  import FastAPI
from pydantic import  BaseModel
import subprocess
import app as user_model
import  uvicorn

class  PromptInput(BaseModel):
    prompt: str = ""

http_api = FastAPI()

http_api.get("/healthcheck")
async def healthcheck():
    gpu = False 
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode  == 0:
        gpu = True

    return {
        "state": "healthy",
        "gpu": gpu
    }

http_api.post("/")
async def inference(prompt: PromptInput):
    try:
        output = user_model.inference(prompt)
    except Exception as e:
        print(e)

    return output

if __name__ == "__main__":
    uvicorn.run(http_api, host="0.0.0.0", port=8000)