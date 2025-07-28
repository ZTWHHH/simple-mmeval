import re
import copy
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args

class TaskRunner(Task):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self, args):
        model_path = f"Cylingo/{args.model_name_or_path}"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        messages = self.parse_input(sample)
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(text, image_inputs, video_inputs)
        else:
            ori_sample.update(self._score_choices(text, image_inputs, video_inputs, sample))

        return ori_sample

    def _generate_response(self, text, image_inputs, video_inputs):
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return output_text

    def _score_choices(self, text, image_inputs, video_inputs, sample):
        raise NotImplementedError("Scoring is not supported.")

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
            
            if any(p in chunk for p in constants.all):
                
                # TODO: Qwen2.5-VL might support other modality
                assert chunk == constants.image, f"Unsupported placeholder {chunk}"

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
