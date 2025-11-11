import re
import copy
import os
import base64
import io

from anthropic import Anthropic
from dotenv import load_dotenv
from PIL import Image

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args

load_dotenv()


def encode_image(image):
    """Encode PIL Image to base64 with media type"""
    buffered = io.BytesIO()
    
    # Convert RGBA/LA/P images to RGB
    if image.mode in ('RGBA', 'LA', 'P'):
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'P':
            image = image.convert('RGBA')
        if image.mode in ('RGBA', 'LA'):
            rgb_image.paste(image, mask=image.split()[-1])
        else:
            rgb_image.paste(image)
        image = rgb_image
    
    image.save(buffered, format="JPEG")
    image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
    media_type = "image/jpeg"
    
    return image_data, media_type


class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.default_gen_kwargs = {"max_tokens": 512}
        
        if hasattr(args, 'max_new_tokens') and args.max_new_tokens:
            self.default_gen_kwargs["max_tokens"] = args.max_new_tokens
        
        super().__init__(args)
        
    def load_model(self, args):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        self.client = Anthropic(api_key=api_key)
        self.model_name = args.model_name_or_path.split("/")[-1]

    def parse_input(self, sample: dict):
        question = sample["prompt"]
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        media_list = copy.deepcopy(sample['media'])

        content = []

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                image = media_list.pop(0)
                image_data, media_type = encode_image(image)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                })
            elif chunk == constants.video:
                raise NotImplementedError("Anthropic Claude does not support video input")
            else:
                content.append({
                    "type": "text",
                    "text": chunk
                })

        return content

    def _generate_response(self, content):
        message = self.client.messages.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            **self.default_gen_kwargs
        )
        
        return message.content[0].text

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        content = self.parse_input(ori_sample)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(content)
        else:
            raise NotImplementedError("Score target mode not supported for Anthropic API models")

        return ori_sample


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()

