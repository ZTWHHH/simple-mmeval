import re
import copy

import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from moviepy.editor import VideoFileClip

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs

THINK_MODE = True

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_gen_kwargs = {"max_new_tokens": 3072}
        self.model_kwargs = parse_model_kwargs(args)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)
        self.num_frames = 8

        super().__init__(args)

    def load_model(self, args):
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            **self.model_kwargs
        ).cuda()
        self.text_tokenizer = self.model.text_tokenizer

    def _extract_video_frames(self, video_path):
        with VideoFileClip(video_path) as clip:
            total_frames = int(clip.fps * clip.duration)
            indices = [int(i * total_frames / self.num_frames) for i in range(self.num_frames)]
            frames = [Image.fromarray(clip.get_frame(idx / clip.fps)) for idx in indices]
        return frames

    def _parse_input(self, message: dict):
        question = message["prompt"]
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        media_list = copy.deepcopy(message.get("media", []))

        content = []
        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                image = media_list.pop(0)
                content.append({"type": "image", "image": image})
            elif chunk == constants.video:
                video = media_list.pop(0)
                frames = self._extract_video_frames(video)
                content.append({"type": "video", "video": frames})
            else:
                content.append({"type": "text", "text": chunk})

        messages = [{"role": "user", "content": content}]
        return messages

    def _generate_response(self, input_ids, pixel_values, grid_thws):
        with torch.inference_mode():
            self.gen_kwargs["eos_token_id"] = self.text_tokenizer.eos_token_id
            self.gen_kwargs["pad_token_id"] = self.text_tokenizer.pad_token_id

            if THINK_MODE:
                outputs = self.model.generate(
                    inputs=input_ids,
                    pixel_values=pixel_values,
                    grid_thws=grid_thws,
                    enable_thinking=True,
                    enable_thinking_budget=True,
                    thinking_budget=2048,
                    **self.gen_kwargs
                )
            else:
                outputs = self.model.generate(
                    inputs=input_ids,
                    pixel_values=pixel_values,
                    grid_thws=grid_thws,
                    do_sample=True,
                    eos_token_id=self.text_tokenizer.eos_token_id,
                    pad_token_id=self.text_tokenizer.pad_token_id,
                    **self.gen_kwargs
                )
        
        response = self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

    def run_sample(self, sample: dict):
        message = sample["messages"][0]
        ori_sample = copy.deepcopy(sample)

        media_list = message.get("media", [])

        if media_list:
            messages = self._parse_input(message)
        else:
            # Text-only input
            messages = [{"role": "user", "content": message["prompt"]}]

        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=True
        )
        input_ids = input_ids.cuda()
        pixel_values = pixel_values.cuda() if pixel_values is not None else None
        grid_thws = grid_thws.cuda() if grid_thws is not None else None

        if not self.args.score_target:
            response = self._generate_response(input_ids, pixel_values, grid_thws)
            ori_sample["messages"].append({"role": "assistant", "response": response})
        else:
            pass

        return ori_sample


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
