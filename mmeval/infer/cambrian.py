import os, sys, importlib
d = os.path.abspath(os.path.dirname(__file__))
popped = sys.path.pop(0) if sys.path and os.path.abspath(sys.path[0]) == d else None
try:
    m = importlib.import_module('cambrian')  
    sys.modules['cambrian'] = m        
finally:
    if popped is not None:
        sys.path.insert(0, popped)

import copy
import torch

import os
import numpy as np
import random

from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.model.builder import load_pretrained_model
from cambrian.utils import disable_torch_init
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

conv_mode_map = {
    "cambrian-phi3-3b": "phi3",
    "cambrian-8b": "llama_3",
    "cambrian-34b": "chatml_direct",
    "cambrian-13b": "vicuna_v1",
}

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_gen_kwargs = {"max_new_tokens": 512, "temperature": 0, "num_beams": 1, "use_cache": True, "do_sample": False}
        self.model_kwargs = parse_model_kwargs(args)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        model_path = os.path.expanduser(args.model_name_or_path)
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, self.model_name, **self.model_kwargs, device_map=self.device)
        
    def _parse_input(self, message:dict):
        prompt = message["prompt"]
        prompt = prompt.replace("<image>", "")

        return prompt
    
    def process(self, question, image):
        if image:
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + question
        else:
            qs = question

        conv = conv_templates[conv_mode_map[self.model_name]].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        if image:
            image_sizes = [image.size]
            image_tensor = process_images([image], self.image_processor, self.model.config)
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        else:
            image_sizes = None
            image_tensor = None
            input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.model.device)

        return input_ids, image_tensor, image_sizes, prompt
    
    def _generate_response(self, input_ids, image_tensor, image_sizes):
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                **self.gen_kwargs)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return outputs

    def run_sample(self, sample: dict):
        message = sample["messages"][0]
        ori_sample = copy.deepcopy(sample)
        question = self._parse_input(message)
        media_list = message.get("media", [])
        image = media_list[0] if media_list else None
        
        input_ids, image_tensor, image_sizes, prompt = self.process(question, image)

        if not self.args.score_target:
            response = self._generate_response(input_ids, image_tensor, image_sizes)
            ori_sample["messages"].append({"role": "assistant", "response": response})
        else:
            pass

        return ori_sample

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
