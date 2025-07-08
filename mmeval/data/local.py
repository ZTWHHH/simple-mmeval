import json
import os

class LocalJSONDataset:
    def __init__(self, args):
        self.args = args
        self.parallel_per_task = args.parallel_per_task
        self.rank = args.rank

        self.data_file = args.infile
        data = json.load(open(self.data_file, "r"))
        self.img_dir = args.img_dir

        data_list = []
        for sample in data:
            media = [os.path.join(self.img_dir, f) for f in sample["media"]]
            sample["media"] = media
            data_list.append(sample)
        
        self.data = data_list[self.rank::self.parallel_per_task]
    @property
    def name(self):
        return f"local@{self.data_file.split('/')[-1]}"
    
    def __iter__(self):
        return iter(self.data)
    
    def __next__(self):
        return next(self.data)  

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    