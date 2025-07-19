"""
Moondream1 is a VLM that can take in single image and text.
https://huggingface.co/vikhyatk/moondream1
"""
import re
import copy
import torch
import numpy as np
import transformers
from transformers import AutoModelForCausalLM, CodeGenTokenizerFast as Tokenizer
from PIL import Image

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(args)
    
    def load_model(self, args):
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, 
            trust_remote_code=True,
        ).to(self.device)
        self.tokenizer = Tokenizer.from_pretrained(args.model_name_or_path)
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        messages = self.parse_input(sample)
        
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

    def _generate_response(self, text, image_path):
        if image_path:
            image = Image.open(image_path)
            enc_image = self.model.encode_image(image)
        output_text = self.model.answer_question(enc_image, text, self.tokenizer)

        return output_text

    def _score_choices(self, text, image_path, sample):
        return NotImplementedError

    def parse_input(self, sample:dict):
        question = sample["prompt"]
        # placeholder <>, can be image, video, audio, etc.
        q_chunks = re.split(r'(<[^>]*>)', question)
        images = copy.deepcopy(sample['media'])

        messages = [
            {
                "role": "user",
                "content": []
            }
        ]

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            
            # Check if chunk contains any placeholder
            if chunk in constants.all:
                
                # Moondream1 only supports images
                assert chunk == constants.image, f"Moondream1 only supports images, got {chunk}"

                media_file = images.pop(0)
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