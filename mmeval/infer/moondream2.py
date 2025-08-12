"""
Moondream2 is a VLM that can take in single image and text.
https://huggingface.co/vikhyatk/moondream2
"""
import re
import copy
import torch
import numpy as np
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(args)
    
    def load_model(self, args):
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            revision="2025-06-21",
            trust_remote_code=True,
            device_map={"": str(self.device)}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        messages = self._parse_input(sample)
        
        prompt = ""
        image_path = None
        for content in messages[0]["content"]:
            if content["type"] == "text":
                prompt += content["text"]
            elif content["type"] == "image":
                image_path = content["image"]
        
        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(prompt, image_path)
        else:
            ori_sample.update(self._score_choices(prompt, image_path, sample))

        return ori_sample

    def _generate_response(self, text, image):
        if image:
            if isinstance(image, str):
                image = Image.open(image)
            output = self.model.query(image, text)
            return output["answer"]
        else:
            raise ValueError("Moondream2 requires an image input")

    def _score_choices(self, text, image_path, sample):
        pass

    def _parse_input(self, sample:dict):
        prompt = sample["prompt"]
        # placeholder <>, can be image, video, audio, etc.
        q_chunks = re.split(r'(<[^>]*>)', prompt)
        media = copy.deepcopy(sample['media'])

        messages = [
            {
                "role": "user",
                "content": []
            }
        ]

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                media_file = media.pop(0)
                messages[0]["content"].append(
                    {
                        "type": "image",
                        "image": media_file
                    }
                )       
            else:
                messages[0]["content"].append(
                    {
                        "type": "text",
                        "text": chunk
                    }
                )

        return messages

    

if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()