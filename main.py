import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI
from pydantic import BaseModel
import accelerate

# Initialize FastAPI app
app = FastAPI()

model_id = "Qwen/QwQ-32B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Using half-precision (FB16/FP16)
    device_map="auto"  # Automatically loads to GPU if available, otherwise CPU
)


# Define a data model for the API request
class InferenceRequest(BaseModel):
    prompt: str


# API endpoint to generate text based on a prompt
@app.post("/generate")
def generate_text(request: InferenceRequest):
    # Tokenize the input prompt and shift tensors to the same device as the model
    inputs = tokenizer(request.prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate text using the model.
    outputs = model.generate(**inputs, max_new_tokens=512)

    # Decode the generated tokens back into text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}


# Entry point for development using uvicorn
# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run(app, host="0.0.0.0", port=8000)