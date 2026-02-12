import re
import copy
import torch
import PIL
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs


def apply_prompt_template(prompt):
    s = (
        '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
        "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
        f'<|user|>\n{prompt}<|end|>\n<|assistant|>\n'
    )
    return s 

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {}
        self.default_gen_kwargs = {
            "temperature": 0.05,
            "do_sample": False, 
            "max_new_tokens": 1024, 
            "top_p": None, 
            "num_beams": 1
        }
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        self.model = AutoModelForVision2Seq.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            **self.model_kwargs
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, 
            trust_remote_code=True, 
            use_fast=False, 
            legacy=False
        )
        
        self.image_processor = AutoImageProcessor.from_pretrained(
            args.model_name_or_path, 
            trust_remote_code=True
        )
        
        # Update special tokens
        self.tokenizer = self.model.update_special_tokens(self.tokenizer)
        
        # Move model to GPU and set eval mode
        self.model = self.model.to(device='cuda', dtype=self.dtype)
        self.model.eval()
        
        # Configure tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.eos_token = '<|end|>'
        
    def _parse_input(self, message: dict):
        prompt = message["prompt"]
        return apply_prompt_template(prompt)

    def _generate_response(self, inputs, image_sizes):
        if image_sizes:
            generated_text = self.model.generate(
                **inputs, 
                image_size=[image_sizes],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **self.gen_kwargs
            )
        else:
            generated_text = self.model.generate(
                **inputs, 
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **self.gen_kwargs
            )
        
        prediction = self.tokenizer.decode(generated_text[0], skip_special_tokens=True).split("<|end|>")[0]
        
        return prediction
    
    def run_sample(self, sample: dict):
        message = sample["messages"][0]
        ori_sample = copy.deepcopy(sample)
        prompt = self._parse_input(message)
        
        # Process images
        images = message.get("media", [])
        image_list = []
        image_sizes = []
        
        if images:
            for img in images:
                image_list.append(self.image_processor([img], image_aspect_ratio='anyres')["pixel_values"].to(device='cuda', dtype=self.dtype))
                image_sizes.append(img.size)
                inputs = {"pixel_values": [image_list]}
        else:
            inputs = {"pixel_values": None}
        
        # Process text
        language_inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs.update(language_inputs)
        
        # Move to CUDA
        for name, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[name] = value.to(device='cuda')

        if not self.args.score_target:
            response = self._generate_response(inputs, image_sizes if images else None)
            ori_sample["messages"].append({"role": "assistant", "response": response})
        else:
            # Handle scoring if needed
            pass

        return ori_sample

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
