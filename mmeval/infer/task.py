import os
import tqdm

from mmeval.data import load_dataset
from mmeval.utils.res_handler import ResponseHandler

class Task:
    def __init__(self, args):


        self.max_retry = args.max_retry
        self.max_retry_sample = args.max_retry_sample
        self.out_dir = args.out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.res_handler = ResponseHandler(args)
        self.dataset = load_dataset(args)

        self.load_model(args)

    def run_sample(self, sample:dict):
        raise NotImplementedError("run_sample is not implemented")

    def run_batch(self, samples:list):
        raise NotImplementedError("run_batch is not implemented")

    def load_model(self, args):
        raise NotImplementedError("load_model is not implemented")

    def inference_dataset(self):

        run_count = 0
        while not self.res_handler.check_complete(self.dataset) and run_count < self.max_retry:
            run_count += 1

            for sample in tqdm.tqdm(self.dataset, total=len(self.dataset), desc=f"Running {self.dataset.name}"):
                
                cnt = 0
                try:
                    ret = self.run_sample(sample)
                    self.res_handler.save(ret)

                except Exception as e:
                    print(f"Encountered Error: {e}")
                    cnt += 1
                    if cnt >= self.max_retry_sample:
                        print("Max retries reached, skip example.")
                        continue