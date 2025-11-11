import re
import copy
import os
import time
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args

load_dotenv()


class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.default_gen_kwargs = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 512,
        }
        
        if hasattr(args, 'max_new_tokens') and args.max_new_tokens:
            self.default_gen_kwargs["max_output_tokens"] = args.max_new_tokens
        
        super().__init__(args)
        
    def load_model(self, args):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model_name = args.model_name_or_path.split("/")[-1]
        self.model = genai.GenerativeModel(self.model_name)
        self.client = genai

    def parse_input(self, sample: dict):
        question = sample["prompt"]
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        media_list = copy.deepcopy(sample['media'])

        contents = []

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                image_path = media_list.pop(0)
                image = Image.open(image_path)
                contents.append(image)
            elif chunk == constants.video:
                video_path = media_list.pop(0)
                video_file = self.client.upload_file(path=video_path)
                
                while video_file.state.name == "PROCESSING":
                    time.sleep(1)
                    video_file = self.client.get_file(video_file.name)
                
                if video_file.state.name == "FAILED":
                    raise ValueError(f"Video processing failed: {video_file.state.name}")
                
                contents.append(video_file)
            else:
                contents.append(chunk)

        return contents

    def _generate_response(self, contents):
        generation_config = genai.types.GenerationConfig(**self.default_gen_kwargs)
        
        response = self.model.generate_content(
            contents,
            generation_config=generation_config
        )
        
        return response.text

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        contents = self.parse_input(ori_sample)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(contents)
        else:
            raise NotImplementedError("Score target mode not supported for Google Gemini API models")

        return ori_sample


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()

