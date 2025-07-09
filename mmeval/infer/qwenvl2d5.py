import re
import copy
import torch
import numpy as np
import transformers
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer

from qwen_vl_utils import process_vision_info

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

class TaskRunner(Task):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
    
    def load_model(self, args):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name_or_path, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    def run_sample(self, sample:dict):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ori_sample = copy.deepcopy(sample)
        messages = self.parse_input(sample)
        
    
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        # Inference
        if not self.args.output_scores:
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            ori_sample["response"] = output_text

        else:
            contents = sample.get("choices")

            full = [text+content for content in contents]  # full conversation for each choice
            full_encoded = [self.processor(text=i, images=image_inputs, videos=video_inputs, return_tensors="pt").to(device) for i in full]
            prompt_encoded = self.processor(text=text, images=image_inputs, videos=video_inputs, return_tensors="pt").to(device)
            target_toks = target_tokens(self.tokenizer, contents)
            scorer = IncrementalLMScorer(self.model, device, tokenizer=self.tokenizer)
            scores = scorer.conditional_score(target_toks, full_encoded, prompt_encoded)  # inputs are used to truncate/locate the prompt and choices' contents
            ori_sample["score"] = scores
            ori_sample["response"] = contents[np.argmax(np.array(scores))]  # model most preferred choice
        return ori_sample

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
