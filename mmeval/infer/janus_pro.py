import os, sys, importlib
d = os.path.abspath(os.path.dirname(__file__))
popped = sys.path.pop(0) if sys.path and os.path.abspath(sys.path[0]) == d else None
try:
    m = importlib.import_module('janus')  
    sys.modules['janus'] = m        
finally:
    if popped is not None:
        sys.path.insert(0, popped)

import copy
import torch

from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_gen_kwargs = {
            "max_new_tokens": 512,
            "do_sample": False,
            "use_cache": True
        }
        self.model_kwargs = parse_model_kwargs(args)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)
        
        super().__init__(args)

    def load_model(self, args):
        # specify the path to the model
        model_path = args.model_name_or_path
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.vl_gpt = self.vl_gpt.to(self.dtype).cuda().eval()
        
    def _parse_input(self, message:dict):
        prompt = message["prompt"]
        media_list = message.get('media', [])
        content = prompt.replace("<image>", "<image_placeholder>")

        conversation = [
            {
                "role": "<|User|>",
                "content": content,
                "images": media_list,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        return conversation

    def _generate_response(self, inputs_embeds, attention_mask):
        # run the model to get the response
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.gen_kwargs
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

        return answer
    
    def run_sample(self, sample: dict):
        message = sample["messages"][0]
        ori_sample = copy.deepcopy(sample)
        pil_images = message.get("media", [])
        conversation = self._parse_input(message)

        # load images and prepare for inputs
        if pil_images:
            prepare_inputs = self.vl_chat_processor(
                conversations=conversation, images=pil_images, force_batchify=True
            ).to(self.vl_gpt.device, self.dtype)
            inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            attention_mask = prepare_inputs.attention_mask
        else:
            sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.vl_chat_processor.sft_format,
                system_prompt=self.vl_chat_processor.system_prompt,
            )
            input_ids = torch.LongTensor(self.tokenizer.encode(sft_format)).unsqueeze(0).to(self.vl_gpt.device)
            inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(input_ids)
            attention_mask = torch.ones_like(input_ids)

        if not self.args.score_target:
            response = self._generate_response(inputs_embeds, attention_mask)
            ori_sample["messages"].append({"role": "assistant", "response": response})
        else:
            pass

        return ori_sample

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
