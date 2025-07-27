import re
import copy
import torch
import numpy as np
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

class TaskRunner(Task):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DEBUG] Using device: {self.device}")
    
    def load_model(self, args):
        print(f"[DEBUG] Loading model from: {args.model_name_or_path}")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            args.model_name_or_path, 
            device_map="auto",
            torch_dtype="auto"
        )
        self.processor = Blip2Processor.from_pretrained(args.model_name_or_path)
        self.tokenizer = self.processor.tokenizer
        print(f"[DEBUG] Model loaded successfully")
    
    def run_sample(self, sample: dict):
        print(f"[DEBUG] Processing sample ID: {sample.get('id', 'unknown')}")
        ori_sample = copy.deepcopy(sample)
        try:
            image, question = self.parse_input(sample)
            print(f"[DEBUG] Question: {question[:50]}...")
            
            if not self.args.score_target:
                response = self._generate_response(image, question)
                print(f"[DEBUG] Generated response: {response}")
                ori_sample["response"] = response
            else:
                ori_sample.update(self._score_choices(image, question, sample))
        except Exception as e:
            print(f"[ERROR] Exception in run_sample: {e}")
            import traceback
            traceback.print_exc()
            ori_sample["response"] = ""

        return ori_sample

    def _generate_response(self, image, question):
        print(f"[DEBUG] Generating response for question")
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        print(f"[DEBUG] Input shape: {inputs.input_ids.shape}")
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        print(f"[DEBUG] Generated IDs shape: {generated_ids.shape}")
        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        print(f"[DEBUG] Decoded text: {output_text}")
        
        return output_text

    def _score_choices(self, image, question, sample):
        contents = sample.get("choices")
        full = [question + " " + content for content in contents]
        
        full_encoded = []
        for text in full:
            encoded = self.processor(image, text, return_tensors="pt").to(self.device)
            full_encoded.append(encoded)
        
        prompt_encoded = self.processor(image, question, return_tensors="pt").to(self.device)
        target_toks = target_tokens(self.tokenizer, contents)
        
        scorer = IncrementalLMScorer(self.model, self.device, tokenizer=self.tokenizer)
        scores = scorer.conditional_score(target_toks, full_encoded, prompt_encoded)
        
        return {
            "score": scores,
            "response": contents[np.argmax(scores)]
        }

    def parse_input(self, sample: dict):
        question = sample["prompt"]
        q_chunks = re.split(r'(<[^>]*>)', question)
        images = copy.deepcopy(sample['media'])
        
        processed_question = ""
        image = None
        
        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            
            if any(p in chunk for p in constants.all):
                assert chunk == constants.image, f"BLIP2 only supports image input, got {chunk}"
                
                media_file = images.pop(0)
                if isinstance(media_file, str):
                    image = Image.open(media_file).convert('RGB')
                else:
                    image = media_file
                    
            else:
                processed_question += chunk
        
        return image, processed_question.strip()


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
