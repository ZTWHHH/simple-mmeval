import json

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from mmeval.data.base import BaseDataset


class MMEvalHFDataset(BaseDataset):
    """Dataset loader for HuggingFace datasets in mm-eval format."""

    def __init__(self, args):
        self.dataset_name = args.dataset.split("@", 1)[1]
        self.subset = getattr(args, "subset", None)
        self.split = args.split
        self.circular = args.circular
        self.resize = args.resize  # Used by base._process_messages for image resizing
        if self.resize is not None:
            print(f"Resizing images to {self.resize}x{self.resize}")
        super().__init__(args)

    def _load_raw_data(self, args) -> tuple:
        meta_path = hf_hub_download(
            repo_id=self.dataset_name,
            filename="metadata.json",
            repo_type="dataset",
        )
        with open(meta_path, encoding="utf-8") as f:
            subsets = json.load(f).get("subsets") or {}
        if not subsets:
            raise ValueError(f"{self.dataset_name} has no subsets")
        if self.subset is None:
            if len(subsets) != 1:
                raise ValueError(
                    f"{self.dataset_name} has {len(subsets)} subsets "
                    f"({list(subsets)}); pass --subset to pick one"
                )
            self.subset = next(iter(subsets))
        elif self.subset not in subsets:
            raise ValueError(
                f"{self.dataset_name}: subset {self.subset!r} not in {list(subsets)}"
            )
        dataset_template = subsets[self.subset].get("prompt_template")

        ds = load_dataset(self.dataset_name, name="default", split=self.split)
        return ds, dataset_template

    def convert_circular(self, **kwargs) -> any:
        """Prepare dataset for circular evaluation."""
        raise NotImplementedError("convert_circular not implemented.")

    def _process_sample(self, idx: int):
        sample = dict(self._raw_dataset[idx])

        # Add eval-id if not present
        if "eval-id" not in sample:
            sample["eval-id"] = idx

        messages = sample["messages"]
        message_list = json.loads(messages) if isinstance(messages, str) else messages

        # Get media from sample level (HF datasets may store as single image or list)
        media = sample.get("media")
        if media is None:
            media_list = []
        elif isinstance(media, list):
            media_list = media
        else:
            media_list = [media]

        # Ensure sample["media"] is always a list
        sample["media"] = media_list

        # Process all messages with sample-level media indexed by placeholder order
        sample["messages"] = self._process_messages(message_list, media_list)

        return sample
