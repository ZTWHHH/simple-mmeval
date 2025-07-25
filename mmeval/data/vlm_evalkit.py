import os
import re
import ast
import sys
import base64
import string
import pandas as pd
from PIL import Image
from io import BytesIO
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from .dataset import Dataset


load_dotenv()

__all__ = ["VLMEVALKIT_DATASET_LIST", "VLMEvalKitDataset", "VLMEVALKIT_MULTIPART_DATASET_CONFIG", "VLMEVALKIT_CONCAT_DATASET_SETS"]

# VLMEvalKit supported datasets
VLMEVALKIT_DATASET_LIST = [
    '3DSRBench',
    'A-Bench_TEST',
    'A-Bench_VAL',
    'A-OKVQA',
    'A4Bench',
    'AI2D_TEST',
    'AI2D_TEST_NO_MASK',
    'AMBER',
    'AesBench_TEST',
    'AesBench_VAL',
    'BLINK',
    'CMMU_MCQ',
    'CRPE_EXIST',
    'CharXiv_descriptive_val',
    'CharXiv_reasoning_val',
    'ChartQA_TEST',
    'Creation_MMBench',
    'DocVQA_TEST',
    'DocVQA_VAL',
    'GOBench',
    'GQA_TestDev_Balanced',
    'HRBench4K',
    'HRBench8K',
    'InfoVQA_TEST',
    'InfoVQA_VAL',
    'LEGO',
    'LLaVABench',
    'LogicVista',
    'MIA-Bench',
    'MLLMGuard_DS',
    'MM-IFEval',
    'MM-Math',
    'MMBench_dev_ar',
    'MMBench_dev_cn',
    'MMBench_dev_en',
    'MMBench_dev_pt',
    'MMBench_dev_ru',
    'MMBench_dev_tr',
    'MMCR',
    'MME',
    'MMMB',
    'MMMB_ar',
    'MMMB_cn',
    'MMMB_en',
    'MMMB_pt',
    'MMMB_ru',
    'MMMB_tr',
    'MMSci_DEV_Captioning_image_only',
    'MMSci_DEV_MCQ',
    'MMStar',
    'MMT-Bench_ALL',
    'MMT-Bench_VAL',
    'MMVP',
    'MMVet',
    'MMVet_Hard',
    'MTL_MMBench_DEV',
    'MTVQA_TEST',
    'MUIRBench',
    'MathVerse_MINI',
    'MathVerse_MINI_Text_Dominant',
    'MathVerse_MINI_Text_Lite',
    'MathVerse_MINI_Vision_Dominant',
    'MathVerse_MINI_Vision_Intensive',
    'MathVerse_MINI_Vision_Only',
    'MathVision',
    'MathVision_MINI',
    'MathVista_MINI',
    'MedXpertQA_MM_test',
    'MicroBench',
    'MicroVQA',
    'NaturalBenchDataset',
    'OCRBench',
    'OlympiadBench',
    'OmniMedVQA',
    'POPE',
    'PathMMU_TEST',
    'PathMMU_VAL',
    'PathVQA_TEST',
    'PathVQA_VAL',
    'Q-Bench1_TEST',
    'Q-Bench1_VAL',
    'R-Bench-Dis',
    'R-Bench-Ref',
    'RealWorldQA',
    'SEEDBench2',
    'SEEDBench2_Plus',
    'SEEDBench_IMG',
    'ScienceQA_TEST',
    'ScienceQA_VAL',
    'TableVQABench',
    'TaskMeAnything_v1_imageqa_random',
    'TextVQA_VAL',
    'VCR_EN_EASY_ALL',
    'VCR_EN_HARD_ALL',
    'VCR_ZH_EASY_ALL',
    'VCR_ZH_HARD_ALL',
    'VL-RewardBench',
    'VStarBench',
    'VisOnlyQA-VLMEvalKit',
    'VizWiz',
    'WeMath',
    'WeMath_COT',
    'WildVision',
    'WorldMedQA-V',
    'atomic_dataset',
    'electro_dataset',
    'hle',
    'mechanics_dataset',
    'optics_dataset',
    'quantum_dataset',
    'statistics_dataset'
]

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

VLMEVALKIT_MULTIPART_DATASET_CONFIG = {
    "MicroBench": {"filename_pattern": "microbench_part_{}", "start_idx": 1, "end_idx": 14},
    "XLRS-Bench-lite": {"filename_pattern": "XLRS-Bench-lite_part{}", "start_idx": 0, "end_idx": 14}, 
    "OmniEarth-Bench": {"filename_pattern": "OmniEarth-Bench_MCQ_part{}", "start_idx": 0, "end_idx": 14},
    "OmniMedVQA": {"filename_pattern": "omnimedbench_part_{}", "start_idx": 1, "end_idx": 14}
}

VLMEVALKIT_CONCAT_DATASET_SETS = {
    'MMMB': ['MMMB_ar', 'MMMB_cn', 'MMMB_en', 'MMMB_pt', 'MMMB_ru', 'MMMB_tr'],
    'MTL_MMBench_DEV': [
        'MMBench_dev_ar', 'MMBench_dev_cn', 'MMBench_dev_en',
        'MMBench_dev_pt', 'MMBench_dev_ru', 'MMBench_dev_tr'
    ]
}

def normalize_image_placeholders(text: str, num_images: int) -> str:
    """
    Normalize various image placeholder formats to standard <image> format.
    
    Args:
        text: Input text that may contain various image placeholder formats
        num_images: Number of images to expect
        
    Returns:
        Text with normalized <image> placeholders
    """
    text = text or ""
    
    # Fast path: normalize existing placeholders
    if IMG_PLACEHOLDER_RE.search(text):
        return IMG_PLACEHOLDER_RE.sub("<image>", text)
    
    if num_images > 0:
        image_prefix = '<image>' * num_images
        return image_prefix + ' ' + text if text else image_prefix
    
    return text


class VLMEvalKitDataset(Dataset):
    """Dataset class for loading VLMEvalKit datasets from TSV files.
    
    This class provides a bridge between VLMEvalKit's dataset TSV files and the mmeval Dataset interface.
    It supports loading datasets from TSV files with image_url or base64 image data.
    """
    
    def __init__(self, args):
        """Initialize the VLMEvalKit dataset with parallel processing support.
        
        Parameters
        ----------
        args: argparse.Namespace
            Arguments from argparse containing dataset configuration
        """
        # Store args for use in _load_raw_data
        self.dataset_name = args.dataset
        self.dataset_dir = os.getenv('DATASET_DIR')
        self.parallel_num = args.parallel_per_task
        self.rank = args.rank
        
        # Initialize parent class with parallel processing parameters
        super().__init__(
            self.dataset_name, 
            parallel_num=self.parallel_num, 
            rank=self.rank
        )

    def _decode_base64_image(self, image: str) -> Image.Image:
        """Decode base64 image string to PIL Image object."""
        try:
            image_data = base64.b64decode(image)
            return Image.open(BytesIO(image_data))
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {e}")

    def _extract_media_from_sample(self, sample: Dict[str, Any], index: int) -> list:
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
            List of media URLs or PIL Image objects
        """
        media = []
        
        # Priority 1: Check for image_url
        if 'image_url' in sample and pd.notna(sample['image_url']):
            image_url = sample['image_url']
            # Handle multiple image paths stored as string representation of list
            if isinstance(image_url, str):
                if image_url.startswith('[') and image_url.endswith(']'):
                    image_url_list = ast.literal_eval(image_url)
                    if isinstance(image_url_list, list):
                        # Process each image url in the list
                        for image_url in image_url_list:
                            media.append(image_url)
                else:
                    # Single image url
                    media.append(image_url)
            else:
                raise ValueError(f"Unsupported image url in sample {index}: {image_url}")
                        
        # Priority 2: Check for base64 image data
        elif 'image' in sample and pd.notna(sample['image']):
            image = sample['image']
            if isinstance(image, str):
                if image.startswith('[') and image.endswith(']'):
                    image_list = ast.literal_eval(image)
                    if isinstance(image_list, list):
                        # Process each image in the list
                        for image in image_list:
                            media.append(self._decode_base64_image(image))
                else:
                    # Single base64 image
                    media.append(self._decode_base64_image(image))
            else:
                raise ValueError(f"Unsupported image in sample {index}: {image}")

        if not media:
            raise ValueError(f"No image found in sample {index}")
        
        return media
    
    def _build_choices_prompt(self, sample: Dict[str, Any]) -> tuple:
        """Build choices dictionary and prompt text for multiple-choice questions.
        
        Parameters
        ----------
        sample : Dict[str, Any]
            Sample dictionary
            
        Returns
        -------
        tuple
            (choices_dict, choice_prompt_text)
        """
        choices = {
            choice_index: sample[choice_index]
            for choice_index in string.ascii_uppercase
            if choice_index in sample and not pd.isna(sample[choice_index])
        }

        choice_prompt = ""
        if choices:
            choice_prompt = "\nOptions:"
            for choice_index, choice_content in choices.items():
                choice_prompt += f"\n{choice_index}. {choice_content}"
        
        return choices, choice_prompt

    def _load_tsv_file(self, dataset_name: str) -> pd.DataFrame:
        """Load a TSV file using pandas."""
        try:
            tsv_file = os.path.join(self.dataset_dir, f"{dataset_name}.tsv")
            return pd.read_csv(tsv_file, sep='\t')
        except Exception as e:
            raise RuntimeError(f"Failed to load TSV file '{tsv_file}': {e}")
    
    def _load_single_tsv_file(self, dataset_name: str) -> pd.DataFrame:
        """Load a single TSV file for the given dataset.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset or URL
            
        Returns
        -------
        pd.DataFrame
            Loaded DataFrame
        """
        # Try to find TSV file for this dataset
        
        dataset = self._load_tsv_file(dataset_name)

        # Ensure the dataset has required columns
        if 'index' not in dataset.columns:
            dataset['index'] = range(len(dataset))
            
        return dataset

    def _load_multipart_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Download and load multipart dataset files for VLMEvalKit specific datasets.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset to load
            
        Returns
        -------
        pd.DataFrame
            Loaded and merged DataFrame from all parts
        """        
        config = VLMEVALKIT_MULTIPART_DATASET_CONFIG[dataset_name]
        pattern = config["filename_pattern"]
        start_idx = config["start_idx"]
        end_idx = config["end_idx"]
        
        # Load and merge all parts into DataFrames
        dataframes = []
        for part_idx in range(start_idx, end_idx+1):
            dataset = self._load_tsv_file(pattern.format(part_idx))
            dataframes.append(dataset)
        
        # Concatenate all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Ensure the dataset has required columns
        if 'index' not in combined_df.columns:
            combined_df['index'] = range(len(combined_df))
        
        return combined_df

    def _load_concat_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load and concatenate multiple datasets for composite datasets like MMMB.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset to load
        
        Returns
        -------
        pd.DataFrame
            Concatenated DataFrame from all parts
        """
        dataset_list = VLMEVALKIT_CONCAT_DATASET_SETS[dataset_name]
        dataframes = []
        for sub_dataset_name in dataset_list:
            sub_dataset = self._load_tsv_file(sub_dataset_name)
            sub_dataset['sub_dataset'] = [sub_dataset_name] * len(sub_dataset)
            dataframes.append(sub_dataset)

        combined_df = pd.concat(dataframes, ignore_index=True)

        # Ensure the dataset has required column
        if 'index' not in combined_df.columns:
            combined_df['index'] = range(len(combined_df))

        return combined_df

    def _load_raw_data(self) -> Any:
        """Load raw data from TSV files.
        
        Returns
        -------
        Any
            Pandas DataFrame containing the dataset
        """
        # Validate environment
        if not self.dataset_dir:
            raise ValueError("DATASET_DIR environment variable is not set")
    
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        # Handle multipart datasets
        if self.dataset_name in ["MicroBench", "XLRS-Bench-lite", "OmniEarth-Bench", "OmniMedVQA"]:
            return self._load_multipart_dataset(self.dataset_name)
        
        # Handle concat datasets
        if self.dataset_name in ["MMMB", "MTL_MMBench_DEV"]:
            return self._load_concat_dataset(self.dataset_name)
        
        # Handle single file datasets
        return self._load_single_tsv_file(self.dataset_name)

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
        # Get raw item from pandas DataFrame
        raw_item = self._raw_dataset.iloc[index]
        
        # Convert pandas Series to dict
        sample = raw_item.to_dict()
        
        # Initialize processed sample with id
        sample['id'] = index
        
        # Extract media
        media = self._extract_media_from_sample(sample, index)
        
        # Extract question/prompt text
        if 'question' in sample and pd.notna(sample['question']):
            prompt_text = str(sample['question'])
        else:
            raise ValueError(f"No question found in sample {index}.")
        
        # Normalize image placeholders in the text
        prompt = normalize_image_placeholders(prompt_text, len(media))

        # Build choices prompt
        choices, choice_prompt = self._build_choices_prompt(sample)
        prompt += choice_prompt

        # Add hint if available
        hint = sample['hint'] if ('hint' in sample and not pd.isna(sample['hint'])) else None
        if hint:
            prompt += f"\nHint: {hint}"

        # Set final sample fields
        sample['media'] = media
        sample['prompt'] = prompt
        if choices:
            sample["choices"] = choices
        
        # Clean up original fields to reduce memory usage
        if 'image' in sample:
            sample.pop("image")
        
        return sample
