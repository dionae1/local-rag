from llm.base import LLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TransformersLLM(LLM):
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ):
        super().__init__(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loaded LLM '{model_name}' on device '{device}'")

    @torch.no_grad()
    def generate_text(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated = output[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        return text.strip()
