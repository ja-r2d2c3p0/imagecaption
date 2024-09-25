from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from typing import Dict
import torch
import numpy as np

class PaliGemmaPredictor:
    def __init__(self, prompt="caption en", image_col="image"):        
        self.prompt = prompt
        self.image_col = image_col
        self.model_id = "google/paligemma-3b-mix-224"
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(self.model_id).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        images = list(batch[self.image_col])
        prompts = [self.prompt] * len(images)
        model_inputs = self.processor(text=prompts, images=images, return_tensors="pt")
        input_len = model_inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = torch.tensor(self.model.generate(**model_inputs, max_new_tokens=100, do_sample=False))
            mask = torch.tensor([i>=input_len for i in range(generation.shape[1])]).repeat(generation.shape[0],1)
            indices = torch.nonzero(mask, as_tuple=True)
            decoded = self.processor.batch_decode(generation[indices].reshape(generation.shape[0],-1), skip_special_tokens=True)
        
        return {
            "captions": decoded,
            "path": batch['path'].tolist()
        }
