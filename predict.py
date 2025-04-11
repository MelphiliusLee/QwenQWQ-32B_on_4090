from gear import BasePredictor, Input, ConcatenateIterator
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model and tokenizer into memory for efficient prediction."""
        # Replace "Qwen/QwQ-32B" with the actual model identifier if needed.
        model_name = "Qwen/QwQ-32B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use half-precision (FP16)
            device_map="auto"
        )

    async def predict(
            self,
            prompt: str = Input(description="Prompt input", default="你好")
    ) -> ConcatenateIterator[str]:
        """
        Run inference on the input prompt and stream back the generated text.
        """
        # Tokenize the prompt and move input tensors to the model's device.
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}

        # Perform inference with a generation call.
        # You can adjust parameters like max_new_tokens, do_sample, temperature, etc. as needed.
        output_ids = self.model.generate(**inputs, max_new_tokens=128)

        # Decode the output tokens into a string.
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Optional: If you want to stream the output, you can split the text.
        # For example, streaming word by word:
        words = generated_text.split()
        for word in words:
            # Yield each word as it is generated.
            yield word
