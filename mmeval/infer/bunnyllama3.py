import re
import copy
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.float16
        self.default_model_kwargs = {"device_map": "auto", "trust_remote_code": True}
        self.default_gen_kwargs = {"max_new_tokens": 100}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)
    
    def load_model(self, args):
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            **self.model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True
        )
    
    def _parse_input(self, message: dict):
        prompt = message["prompt"]
        q_chunks = re.split(r'(<(?:image|video)>)', prompt)
        media_list = message.get('media', [])
        images = []
        PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: "
        media_idx = 0

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                media_file = media_list[media_idx]
                media_idx += 1
                images.append(media_file)
                PROMPT += "<image>"
            else:
                PROMPT += chunk
                
        PROMPT += " ASSISTANT:"
        return PROMPT, images
    
    def _generate_response(self, text, images):
        if images:
            image_tensor = self.model.process_images(images, self.model.config).to(dtype=self.model.dtype, device=self.model.device)
            text_chunks = [self.tokenizer(chunk).input_ids for chunk in text.split('<image>')]
            
            # Reconstruct input_ids with image tokens (-200)
            input_ids = text_chunks[0]
            for i in range(1, len(text_chunks)):
                input_ids = input_ids + [-200] + text_chunks[i][1:]  # Remove BOS token from subsequent chunks
            
            input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.model.device)
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                **self.gen_kwargs
            )[0]
        else:
            # Text-only generation
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
            output_ids = self.model.generate(
                input_ids,
                images=None,
                **self.gen_kwargs
            )[0]
        response = self.tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        return response

    def run_sample(self, sample: dict):
        message = sample["messages"][0]
        ori_sample = copy.deepcopy(sample)
        text, images = self._parse_input(message)
        
        if not self.args.score_target:
            response = self._generate_response(text, images)
            ori_sample["messages"].append({"role": "assistant", "response": response})
        else:
            ori_sample.update(self._score_choices(text, images, message))
        
        return ori_sample
    
    def _score_choices(self, text, images, message):
        pass

if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()