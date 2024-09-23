from vllm import LLM, SamplingParams, TextPrompt
from typing import Dict, List
import numpy as np
from PIL import Image

class LLMPredictor:
    def __init__(self, sampling_params: SamplingParams=SamplingParams(), prompt: str ="caption en", image_col: str="image", model_id: str = "google/paligemma-3b-mix-224"):
        self.prompt = prompt
        self.image_col = image_col
        self.model_id = model_id
        self.sampling_params = sampling_params

        # Create an LLM.
        self.llm = LLM(
            model=self.model_id
        )

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        images = batch[self.image_col]
        inputs = [TextPrompt(**{"prompt":self.prompt, "multi_modal_data": {"image": Image.fromarray(images[i])}}) for i in range(images.shape[0])]
        outputs = self.llm.generate(inputs, sampling_params=self.sampling_params)
        generated_text = []
        for output in outputs:
            generated_text.append(' '.join([o.text for o in output.outputs]))
        return {
            "captions": generated_text,
            "path": batch['path'].tolist()
        }