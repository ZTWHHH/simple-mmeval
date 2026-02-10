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
        self.dtype = getattr(args, "dtype") or "auto"
        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {"max_new_tokens": 100, "do_sample": False}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(args)
    
    def load_model(self, args):
        self.model = FuyuForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=self.dtype, **self.model_kwargs)
        # Resolve "auto" dtype to actual model dtype
        if self.dtype == "auto":
            self.dtype = self.model.dtype
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    
    def run_sample(self, sample: dict):
        message = sample["messages"][0]
        ori_sample = copy.deepcopy(sample)
        messages = self.parse_input(message)
        media = message.get('media', [])

        if not self.args.score_target:
            response = self._generate_response(messages, media if media else None)
            ori_sample["messages"].append({"role": "assistant", "response": response})
        else:
            ori_sample.update(self._score_choices(messages, media if media else None, message))

        return ori_sample

    def _generate_response(self, messages, media):
        if not media:
            raise ValueError("Fuyu requires an image input, but no image was provided in the sample")
        
        inputs = self.processor(text=messages, images=media, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, **self.gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return output_text

    def _score_choices(self, messages, media, message):
        contents = message.get("choices")
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


    def parse_input(self, message:dict):
        question = message["prompt"]
        question = question.replace(constants.image, "")
        return question
    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()