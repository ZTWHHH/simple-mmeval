import json
from datasets import load_dataset
from mmeval.data.base import BaseDataset


class MMEvalHFDataset(BaseDataset):
    """Dataset loader for HuggingFace datasets in mm-eval format."""

    def __init__(self, args):
        self.dataset_name = args.dataset.split("@")[1] if "@" in args.dataset else args.dataset
        self.split = args.split
        self.circular = args.circular
        self.resize = args.resize  # Used by base._process_message for image resizing
        self.template_arg = args.template
        if self.resize is not None:
            print(f"Resizing images to {self.resize}x{self.resize}")
        super().__init__(args)

    def _load_raw_data(self, args):
        # Load metadata subset to get jinja_template for the current split
        metadata_ds = load_dataset(self.dataset_name, name="metadata", split=self.split)
        jinja_template = metadata_ds[0]["jinja_template"] if len(metadata_ds) > 0 else None
        
        # User template takes priority
        user_template = self._load_template(self.template_arg)
        if user_template is not None:
            jinja_template = user_template
        
        # Load default subset with the current split for data
        ds = load_dataset(self.dataset_name, name="default", split=self.split)
        return ds, jinja_template

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
        media = sample.pop("media", None)
        if media is None:
            media_list = []
        elif isinstance(media, list):
            media_list = media
        else:
            media_list = [media]

        # Process all messages with sample-level media indexed by placeholder order
        sample["messages"] = self._process_messages(message_list, media_list)

        return sample
