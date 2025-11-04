"""
Fuyu-8B is a multi-modal text and image transformer trained by Adept AI.
https://huggingface.co/adept/fuyu-8b
"""
import re
import copy
import torch
import numpy as np
from PIL import Image
from transformers import FuyuForCausalLM, AutoProcessor, AutoTokenizer

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

BEGINNING_OF_ANSWER_STRING = "<0x04>"

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {"max_new_tokens": 100, "do_sample": False}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(args)
    
    def load_model(self, args):
        self.model = FuyuForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=self.dtype, **self.model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        messages = self.parse_input(sample)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(messages, sample['media'])
        else:
            ori_sample.update(self._score_choices(messages, sample['media'], sample))

        return ori_sample

    def _generate_response(self, messages, media):
        inputs = self.processor(text=messages, images=media, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs, **self.gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return output_text

    def _score_choices(self, messages, media, sample):
        contents = sample.get("choices")
        full = [messages + BEGINNING_OF_ANSWER_STRING+content for content in contents]
        full_encoded = [self.processor(text=i, images=media, return_tensors="pt").to(self.device) for i in full]
        prompt_encoded = self.processor(text=messages, images=media, return_tensors="pt").to(self.device)

        target_toks = target_tokens(self.tokenizer, contents)

        scorer = IncrementalLMScorer(self.model, self.device, tokenizer=self.tokenizer)
        scores = scorer.conditional_score(target_toks, full_encoded, prompt_encoded)
        
        return {
            "score": scores,
            "response": contents[np.argmax(scores)]
        }


    def parse_input(self, sample:dict):
        question = sample["prompt"]
        question = question.replace(constants.image, "")
        return question
    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()