import re
from datasets import load_dataset, Image
from mmeval.data.base import BaseDataset

class MMEvalHFDataset(BaseDataset):
    """Dataset loader for HuggingFace datasets in mm-eval format."""

    def __init__(self, args):
        self.dataset_name = args.dataset.split("@")[1] if "@" in args.dataset else args.dataset
        self.split = args.split
        self.circular_eval = args.circular_eval
        super().__init__(args)

    def _load_raw_data(self, args):
        ds = load_dataset(self.dataset_name, split=self.split)
        if "image" in ds.column_names:
            ds = ds.cast_column("image", Image(decode=True))
            print(f"Loaded dataset {self.dataset_name} with image column.")
        if "video" in ds.column_names:
            ds = ds.cast_column("video", Image(decode=True))
        return ds

    def convert_circular(self, **kwargs) -> any:
        """Prepare dataset for circular evaluation."""
        raise NotImplementedError("convert_circular not implemented.")

    def _process_sample(self, idx: int):
        sample = self._raw_dataset[idx]
        prompt = sample["prompt"] if "prompt" in sample else sample["question"]

        # Normalize to list
        image = sample.get("image", None)
        video = sample.get("video", None)

        if image is None:
            image_list = []
        elif isinstance(image, list):
            image_list = image
        else:
            image_list = [image]

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
                media.append(image_list.pop(0))
            elif tag == "video" and video_list:
                media.append(video_list.pop(0))

        # Conditional logic for circular_eval
        if self.circular_eval:
            prompt_clean = self.convert_circular(idx=idx, sample=sample)
        else:
            # Extract <option_index:...:END> and <option_content:...:END> pairs
            index_matches = re.findall(r"<option_index:(.*?):END>", prompt)
            content_matches = re.findall(r"<option_content:(.*?):END>", prompt)
            options = {
                idx_val.strip(): content_val.strip()
                for idx_val, content_val in zip(index_matches, content_matches)
            }
            # Replace <option_index:...:END> and <option_content:...:END> with their contents in the prompt
            def replace_option(match):
                return match.group(1)
            prompt_clean = re.sub(r"<option_index:(.*?):END>", replace_option, prompt)
            prompt_clean = re.sub(r"<option_content:(.*?):END>", replace_option, prompt_clean)

        # create choices for score_target
        choices = [f"{idx_key}. {options[idx_key]}" for idx_key in options]

        return {
            "eval-id": idx,
            "prompt": prompt_clean,
            "media": media,
            "options": options,
            "choices": choices,
            **{k: v for k, v in sample.items() if k not in ("prompt", "image", "video")}
        }