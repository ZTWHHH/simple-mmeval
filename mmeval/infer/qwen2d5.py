import argparse
import json
import os
import re
import tqdm
import copy

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

from mmeval.infer.task import Task
from mmeval.utils import spec_tokens 

class TaskRunner(Task):
    def __init__(self, args):
        super().__init__(args)
        self.load_model(args)
    
    def load_model(self, args):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype="auto", device_map="auto")
        processor = AutoProcessor.from_pretrained(args.model_path)
        self.model = model
        self.processor = processor

    def run_sample(self, sample:dict):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ori_sample = copy.deepcopy(sample)
        messages = self.parse_input(sample)
        
    
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        ori_sample["response"] = output_text
        return ori_sample

    def parse_input(self, sample:dict):
        question = sample["questions"]
        # placeholder <>, can be image, video, audio, etc.
        q_chunks = re.split(r'(<[^>]*>)', question)
        images = copy.deepcopy(sample['modality'])

        messages = [
            {
                "role": "user",
                "content": []
            }
        ]

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            
            if any(p in chunk for p in spec_tokens.all):
                assert chunk == spec_tokens.image, f"Unsupported placeholder {chunk}"

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
    parser = argparse.ArgumentParser(description='Qwen2D5 evaluation')
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--save_freq', type=int, default=3)
    parser.add_argument('--max_retry', type=int, default=3)
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    
    args = parser.parse_args()

    model_evaluator = TaskRunner(args)

    model_evaluator.inference_dataset()

        
