import re
import copy
import torch

from transformers import AutoModelForCausalLM, AutoProcessor

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or "auto"
        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {"max_new_tokens": 100, "do_sample": False, "temperature": 0.0}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)
        
    def load_model(self, args):
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            _attn_implementation='eager', # set to flash_attention_2 if flash attention is supported
            **self.model_kwargs
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            num_crops=4
        )
        
    def _parse_input(self, sample:dict):
        prompt = sample["prompt"]
        # placeholder <>, can be image, video, etc.
        q_chunks = re.split(r'(<(?:image|video)>)', prompt)
        media = copy.deepcopy(sample['media'])

        # Build the prompt with image placeholders for Phi-3.5 Vision
        text_content = []
        placeholder_content = []
        image_counter = 1
        images = []
        
        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                if media:
                    images.append(media.pop(0))
                    placeholder_content.append(f"<|image_{image_counter}|>")
                    image_counter += 1
            else:
                text_content.append(chunk)
        
        # Combine placeholders and text
        full_content = "".join(placeholder_content) + "\n" + "".join(text_content) if placeholder_content else "".join(text_content)
        
        messages = [
            {
                "role": "user",
                "content": full_content
            }
        ]

        return messages, images

    def _generate_response(self, inputs):
        with torch.inference_mode():
            generate_ids = self.model.generate(
                **inputs,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                **self.gen_kwargs
            )
            
            # Remove input tokens
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

        return response
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        messages, images = self._parse_input(ori_sample)

        # Apply chat template
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process with images if available, otherwise just text
        if images:
            inputs = self.processor(prompt, images, return_tensors="pt").to(self.model.device)
        else:
            inputs = self.processor(prompt, return_tensors="pt").to(self.model.device)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(inputs)
        else:
            pass

        return ori_sample

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()