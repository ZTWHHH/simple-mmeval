import re
import json
from datasets import load_dataset, Image
from mmeval.data.base import BaseDataset

class MMEvalHFDataset(BaseDataset):
    """Dataset loader for HuggingFace datasets in mm-eval format."""

    def __init__(self, args):
        self.dataset_name = args.dataset.split("@")[1] if "@" in args.dataset else args.dataset
        self.split = args.split
        self.circular = args.circular
        self.resize = args.resize
        if self.resize is not None:
            print(f"Resizing images to {self.resize}x{self.resize}")
        super().__init__(args)

    def _load_raw_data(self, args):
        ds = load_dataset(self.dataset_name, split=self.split)
        dataset = json.loads(ds[0]['data'])
        return dataset['content'], dataset['jinja_template']

    def convert_circular(self, **kwargs) -> any:
        """Prepare dataset for circular evaluation."""
        raise NotImplementedError("convert_circular not implemented.")

    def _process_sample(self, idx: int):
        sample = self._raw_dataset[idx]['conversations'][0]['value']
        prompt = self.build_prompt(self._prompt_template, sample)

        # Normalize to list
        image = sample.get("image", None)
        video = sample.get("video", None)

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
                    img = self.load_image(img)
                    img = self.resize_image(img, self.resize)
                media.append(img)
            elif tag == "video" and video_list:
                media.append(video_list.pop(0))

        return {
            "eval-id": idx,
            "prompt": prompt,
            "media": media,
            **{k: v for k, v in sample.items() if k not in ("prompt", "image", "video")}
        }