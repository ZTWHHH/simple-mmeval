import re
import copy
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {"torch_dtype": self.dtype}
        self.default_gen_kwargs = {"max_new_tokens": 512, "do_sample": False}
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
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # Store the device for later use
        self.device = next(self.model.parameters()).device
        
        # Set the tokenizer as an attribute of the model for the chat method
        self.model.tokenizer = self.tokenizer
        
        # Ensure model is in eval mode
        self.model.eval()

    def _parse_input(self, sample: dict):
        prompt = sample["prompt"]
        q_chunks = re.split(r'(<(?:image|video)>)', prompt)
        media = copy.deepcopy(sample['media'])

        messages = []
        content = []

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                media_file = media.pop(0)
                if isinstance(media_file, Image.Image):
                    image = media_file.convert('RGB')
                else:
                    image = Image.open(media_file).convert('RGB')
                content.append({"type": "image", "image": image})
            elif chunk == constants.video:
                # Video processing would need to be implemented based on model capabilities
                media_file = media.pop(0)
                # For now, treat as image (first frame)
                content.append({"type": "text", "text": "[VIDEO]"})
            else:
                content.append({"type": "text", "text": chunk})
        
        messages.append({"role": "user", "content": content})
        return messages

    def generate_output(self, sample, **generation_kwargs):
        messages = self._parse_input(sample)
    
        # Extract text and image from the parsed messages
        text_content = ""
        image = None
        
        for msg in messages:
            for content in msg["content"]:
                if content["type"] == "text":
                    text_content += content["text"]
                elif content["type"] == "image":
                    image = content["image"]
        
        # Use the model's multimodal generation capability
        if hasattr(self.model, 'chat'):
            # InternLM-XComposer should handle PIL images directly (similar to other models)
            # Let the model handle its own preprocessing internally
            response, _ = self.model.chat(text_content, image=image, history=None, **self.gen_kwargs, **generation_kwargs)
        else:
            # Text-only generation
            inputs = self.tokenizer(text_content, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    **self.gen_kwargs,
                    **generation_kwargs
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if text_content in response:
                response = response.replace(text_content, "").strip()
        
        return response
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        
        if not self.args.score_target:
            ori_sample["response"] = self.generate_output(ori_sample)
        else:
            # Handle scoring target if needed
            pass
        
        return ori_sample


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()