import os
import re
import ast
import string
import pandas as pd
from typing import Dict, Any
from dotenv import load_dotenv
from mmeval.data.base import BaseDataset

load_dotenv(dotenv_path=".env", override=True)

IMG_PLACEHOLDER_RE = re.compile(
    r"""
        <img[^>]*>            |  # any <img …>
        <image[^>]*>          |  # catch-all <image …>  (covers <image 1>, <image_2>, …)
        <imagehere>           |  # <ImageHere>
        <img_plh>             |  # <IMG_PLH>
        <img_context>            # <IMG_CONTEXT>

    """,
    re.IGNORECASE | re.VERBOSE,
)

class TSVDataset(BaseDataset):
    """Dataset class for loading TSV files.
    
    This class provides a bridge between TSV files and the mmeval Dataset interface.
    """
    
    def __init__(self, args):
        """Initialize the TSV dataset with parallel processing support.
        
        Parameters
        ----------
        args: argparse.Namespace
            Arguments from argparse containing dataset configuration
        """
        self.dataset_dir = os.getenv('DATASET_DIR')
        self.dataset_name = args.dataset
        super().__init__(args)
    
    def _load_raw_data(self, args) -> Any:
        """Load raw data from TSV files.
        
        Returns
        -------
        Any
            Pandas DataFrame containing the dataset
        """
        tsv_file = os.path.join(self.dataset_dir, f"{self.dataset_name}.tsv")
        dataset = pd.read_csv(tsv_file, sep='\t')

        if "eval-id" not in dataset.columns:
            dataset["eval-id"] = range(len(dataset))
        
        # Handle single file datasets
        return dataset

    def _extract_media(self, sample: Dict[str, Any]) -> list:
        """Extract media from sample using image_url or base64 image data.
        
        Parameters
        ----------
        sample : Dict[str, Any]
            Sample dictionary
        index : int
            Sample index for error reporting
            
        Returns
        -------
        list
            List of PIL Image objects
        """
        media = []
        
        # Priority 1: Check for image_url
        if 'image_url' in sample and pd.notna(sample['image_url']):
            image_url = sample['image_url']
            # Handle multiple image paths stored as string representation of list
            if image_url.startswith('[') and image_url.endswith(']'):
                image_url_list = ast.literal_eval(image_url)
                for image_url in image_url_list:
                    media.append(self.load_image(image_url))
            else:
                # Single image url
                media.append(self.load_image(image_url))
                        
        # Priority 2: Check for base64 image data
        elif 'image' in sample and pd.notna(sample['image']):
            image = sample['image']
            if image.startswith('[') and image.endswith(']'):
                image_list = ast.literal_eval(image)
                for image in image_list:
                    media.append(self.load_image(image))
            else:
                # Single base64 image
                media.append(self.load_image(image))
        
        return media

    def _process_sample(self, index: int) -> Dict[str, Any]:
        """Process a raw TSV sample to match format.
        
        Parameters
        ----------
        index : int
            Global index of the sample in the dataset
            
        Returns
        -------
        Dict[str, Any]
            Processed sample with unified format
        """
        sample = self._raw_dataset.iloc[index].to_dict()
        media = self._extract_media(sample)
        question = str(sample['question'])

        # Normalize image placeholders
        if IMG_PLACEHOLDER_RE.search(question):
            prompt = IMG_PLACEHOLDER_RE.sub("<image>", question)
        elif media:
            prompt = f'{"<image>" * len(media)} {question}'.strip()

        # Build choices prompt
        choices = {
            choice_index: sample[choice_index] for choice_index in string.ascii_uppercase
            if choice_index in sample and not pd.isna(sample[choice_index])
        }

        if choices:
            prompt = prompt + "\nOptions:\n" + "\n".join(f"{k}. {v}" for k, v in choices.items())
            sample["choices"] = choices

        # Add hint if available
        hint = sample.get("hint", None)
        if hint:
            prompt += f"\nHint: {hint}"

        sample['media'] = media
        sample['prompt'] = prompt
        
        # Clean up original fields to reduce memory usage
        sample.pop("image", None) 
        
        return sample

    def __repr__(self):
        if self.parallel_per_task > 1:
            return f"{self.dataset_name}(rank={self.rank}/{self.parallel_per_task}, local={len(self)}, global={self.global_length})"
        else:
            return f"{self.dataset_name}(samples={len(self)})"
    
    def __str__(self):
        return self.__repr__()