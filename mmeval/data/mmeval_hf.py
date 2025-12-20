import re
import json
from datasets import load_dataset
from mmeval.data.base import BaseDataset


class MMEvalHFDataset(BaseDataset):
    """Dataset loader for HuggingFace datasets in mm-eval format."""

    def __init__(self, args):
        self.dataset_name = args.dataset.split("@")[1] if "@" in args.dataset else args.dataset
        self.split = args.split
        self.circular = args.circular
        self.resize = args.resize
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

    def _process_message(self, msg: dict):
        """Process a single message dict, building prompt and processing media."""
        # Priority: user template -> existing prompt -> error
        prompt = msg.get("prompt")
        if self._prompt_template is not None:
            try:
                prompt = self.build_prompt(self._prompt_template, msg)
            except Exception as e:
                if prompt is None:
                    raise ValueError(f"Template rendering failed: {e}")
        if prompt is None:
            raise ValueError("No prompt found and no template provided")

        # Normalize to list
        image = msg.get("image", None)
        video = msg.get("video", None)

        if image is None:
            image_list = []
        elif isinstance(image, list):
            image_list = [self.load_image(img) for img in image]
        else:
            image_list = [self.load_image(image)]

        if video is None:
            video_list = []
        elif isinstance(video, list):
            video_list = video
        else:
            video_list = [video]

        media = []
        placeholder_list = re.findall(r"<(video|image)>", prompt)
        for tag in placeholder_list:
            if tag == "image" and image_list:
                img = image_list.pop(0)
                # Resize image if resize parameter is set
                if self.resize is not None:
                    img = self.resize_image(img, self.resize)
                media.append(img)
            elif tag == "video" and video_list:
                media.append(video_list.pop(0))

        return {
            "prompt": prompt,
            "media": media,
            **{k: v for k, v in msg.items() if k not in ("prompt", "image", "video")}
        }

    def _process_sample(self, idx: int):
        sample = self._raw_dataset[idx]
        
        # Get image from sample level (HF datasets store image at sample level)
        sample_image = sample.get("image", None)
        
        # Handle different message field names: "messages" or "conversation"
        if "messages" in sample:
            messages_list = sample["messages"]
        elif "conversation" in sample:
            conv = sample["conversation"]
            # Parse JSON string if needed
            messages_list = json.loads(conv) if isinstance(conv, str) else conv
        else:
            # Wrap sample itself as single message
            messages_list = [dict(sample)]
        
        # Inject sample-level image into each message if not present
        processed_messages = []
        for msg in messages_list:
            msg_dict = dict(msg)  # Always make a copy
            if sample_image is not None and "image" not in msg_dict:
                msg_dict["image"] = sample_image
            processed_messages.append(self._process_message(msg_dict))

        return {
            "eval-id": idx,
            "messages": processed_messages,
        }
