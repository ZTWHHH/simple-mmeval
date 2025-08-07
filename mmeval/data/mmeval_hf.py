import re
from datasets import load_dataset
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
        return ds

    def convert_circular(self, **kwargs) -> any:
        """Prepare dataset for circular evaluation."""
        raise NotImplementedError("convert_circular not implemented.")

    def _process_sample(self, idx: int):
        sample = self._raw_dataset[idx]
        prompt = sample["prompt"] if "prompt" in sample else sample["question"]

        # media: always a list, follow <image> and <video> order in prompt
        media = []
        image_list = sample["image"] if "image" in sample and sample["image"] is not None else []
        video_list = sample["video"] if "video" in sample and sample["video"] is not None else []

        if not isinstance(image_list, list):
            image_list = [image_list]
        if not isinstance(video_list, list):
            video_list = [video_list]

        img_idx, vid_idx = 0, 0
        placeholder_list = re.findall(r"<(image|video)>", prompt)
        for ph in placeholder_list:
            if ph == "<image>":
                if img_idx < len(image_list):
                    media.append(image_list[img_idx])
                    img_idx += 1
            elif ph == "<video>":
                if vid_idx < len(video_list):
                    media.append(video_list[vid_idx])
                    vid_idx += 1

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