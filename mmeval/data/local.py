import json
from mmeval.data.base import BaseDataset


class LocalJSONDataset(BaseDataset):
    """Dataset class for loading local JSON files."""
    
    def __init__(self, args):
        self.data_file = args.infile
        self.media_dir = args.img_dir 
        super().__init__(args)

    def _load_raw_data(self, args):
        with open(self.data_file, "r") as f:
            data = json.load(f)
        for i, sample in enumerate(data):
            assert "eval-id" not in sample, "eval-id already exists"
            sample["eval-id"] = i
        
        return data, None  # No dataset-specific template

    def _process_sample(self, idx: int):
        sample = dict(self._raw_dataset[idx])
        media_list = sample.get("media")
        sample["messages"] = self._process_messages(sample["messages"], media_list)
        return sample

    def __repr__(self):
        if self.parallel_per_task > 1:
            return f"local@{self.data_file.split('/')[-1]}(rank={self.rank}/{self.parallel_per_task}, local={len(self)}, global={self.global_length})"
        else:
            return f"local@{self.data_file.split('/')[-1]}(samples={len(self)})"
    
    def __str__(self):
        return self.__repr__()
