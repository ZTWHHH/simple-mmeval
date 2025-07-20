import sys
import os
import re
import base64
import string
import zipfile
import pandas as pd
from PIL import Image
from io import BytesIO
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from .dataset import Dataset


load_dotenv()

__all__ = ["VLMEvalKitDataset"]

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

MULTIPART_DATASET_CONFIG = {
    "MicroBench": {"filename_pattern": "microbench_part_{}_local.tsv", "start_idx": 1, "end_idx": 14},
    "XLRS-Bench-lite": {"filename_pattern": "XLRS-Bench-lite_part{}_local.tsv", "start_idx": 0, "end_idx": 14}, 
    "OmniEarth-Bench": {"filename_pattern": "OmniEarth-Bench_MCQ_part{}_local.tsv", "start_idx": 0, "end_idx": 14},
    "OmniMedVQA": {"filename_pattern": "omnimedbench_part_{}_local.tsv", "start_idx": 1, "end_idx": 14}
}

CONCAT_DATASET_SETS = {
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
    It supports loading datasets from TSV files in the specified dataset directory, supporting both
    image and video datasets with their specific configurations.
    """
    
    def __init__(
        self, 
        args
    ):
        """Initialize the VLMEvalKit dataset with parallel processing support.
        
        Parameters
        ----------
        args: argparse.Namespace
            Arguments from argparse containing dataset configuration
        """
        # Store args for use in _load_raw_data
        self.dataset_name = args.dataset
        self.dataset_dir = os.getenv('VLMEVALKIT_DATASET_DIR')
        self.parallel_num = args.parallel_per_task
        self.rank = args.rank
        
        # Initialize parent class with parallel processing parameters
        super().__init__(
            self.dataset_name, 
            parallel_num=self.parallel_num, 
            rank=self.rank
        )

    def _resolve_image_path(self, path: str) -> str:
        """Resolve image path to absolute path if needed.
        
        Parameters
        ----------
        path : str
            Image path (relative or absolute)
            
        Returns
        -------
        str
            Resolved absolute path
            
        Raises
        ------
        FileNotFoundError
            If image file is not found
        """
        if os.path.exists(path):
            return path
        
        # Check in dataset-specific images folder
        abs_path = os.path.join(self.dataset_dir, self.dataset_name, 'images', path)
        if os.path.exists(abs_path):
            return abs_path
            
        # Download images if folder doesn't exist
        images_folder = os.path.join(self.dataset_dir, self.dataset_name, 'images')
        if not os.path.exists(images_folder):
            print(f"Images folder not found, downloading {self.dataset_name} images...")
            self.download_image(self.dataset_name)
            
            # Try again after download
            if os.path.exists(abs_path):
                return abs_path
        
        raise FileNotFoundError(f"Image file not found: {abs_path}")

    def download_tsv(self, dataset_name: str) -> None:
        """Download TSV file from Hugging Face for the specified dataset.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset to download
        """
        # Create dataset directory if it doesn't exist
        dataset_path = os.path.join(self.dataset_dir, dataset_name)
        os.makedirs(dataset_path, exist_ok=True)
        
        # Handle multipart datasets
        if dataset_name in MULTIPART_DATASET_CONFIG:
            config = MULTIPART_DATASET_CONFIG[dataset_name]
            pattern = config["filename_pattern"]
            start_idx = config["start_idx"]
            end_idx = config["end_idx"]
            
            for part_num in range(start_idx, end_idx+1):
                filename = pattern.format(part_num)
                try:
                    hf_hub_download(
                        repo_id="mm-eval/VLMEvalKit",
                        filename=filename,
                        local_dir=dataset_path,
                        repo_type="dataset"
                    )
                except Exception as e:
                    print(f"Warning: Failed to download {filename}: {e}")
        else:
            # Download main TSV file
            try:
                hf_hub_download(
                    repo_id="mm-eval/VLMEvalKit",
                    filename=f"{dataset_name}.tsv",
                    local_dir=dataset_path,
                    repo_type="dataset"
                )
            except Exception as e:
                print(f"Warning: Failed to download {dataset_name}.tsv: {e}")
            
            # Try to download local version if exists
            try:
                hf_hub_download(
                    repo_id="mm-eval/VLMEvalKit", 
                    filename=f"{dataset_name}_local.tsv",
                    local_dir=dataset_path,
                    repo_type="dataset"
                )
            except Exception:
                # Local version may not exist, which is fine
                pass

    def download_image(self, dataset_name: str) -> None:
        """Download and extract image ZIP file from Hugging Face for the specified dataset.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset to download images for
        """
        dataset_path = os.path.join(self.dataset_dir, dataset_name)
        os.makedirs(dataset_path, exist_ok=True)
        
        # Download image ZIP file
        zip_path = hf_hub_download(
            repo_id="mm-eval/VLMEvalKit",
            filename=f"{dataset_name}.zip",
            local_dir=dataset_path,
            repo_type="dataset"
        )
        
        # Extract ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        
        # Rename the extracted folder to 'images' (folder name same as zip name)
        old_folder_path = os.path.join(dataset_path, dataset_name)
        new_folder_path = os.path.join(dataset_path, 'images')
        
        if os.path.exists(old_folder_path) and not os.path.exists(new_folder_path):
            os.rename(old_folder_path, new_folder_path)
        
        # Remove the downloaded ZIP file
        os.remove(zip_path)

    def _extract_media_from_sample(self, sample: Dict[str, Any], index: int) -> list:
        """Extract media paths or objects from a sample.
        
        Parameters
        ----------
        sample : Dict[str, Any]
            Sample dictionary
        index : int
            Sample index for error reporting
            
        Returns
        -------
        list
            List of media paths or objects
        """
        media = []
        
        # Check for image-related columns in priority order
        if 'image_path' in sample and pd.notna(sample['image_path']):
            image_path = sample['image_path']
            
            # Handle multiple image paths stored as string representation of list
            if isinstance(image_path, str):
                if image_path.startswith('[') and image_path.endswith(']'):
                    try:
                        import ast
                        parsed_paths = ast.literal_eval(image_path)
                        if isinstance(parsed_paths, list):
                            # Process each path in the list
                            for path in parsed_paths:
                                media.append(self._resolve_image_path(path))
                    except Exception as e:
                        raise ValueError(f"Failed to parse image path in sample {index}: {e}")
                else:
                    # Single path
                    media.append(self._resolve_image_path(image_path))
                        
        elif 'image_url' in sample and pd.notna(sample['image_url']):
            media.append(sample['image_url'])  
        elif 'image' in sample and pd.notna(sample['image']):
            try:
                image_data = base64.b64decode(sample['image'])
                image_object = Image.open(BytesIO(image_data))
            except Exception as e:
                raise ValueError (f"Failed to load image in sample {index}: {e}")
            media.append(image_object)
        else:
            raise ValueError(f"No image found in sample {index}.")
        
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
    
    def _load_single_tsv_file(self, dataset_name: str) -> pd.DataFrame:
        """Load a single TSV file for the given dataset.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset
            
        Returns
        -------
        pd.DataFrame
            Loaded DataFrame
        """
        # Try to find TSV file for this dataset
        tsv_file = os.path.join(self.dataset_dir, dataset_name, f"{dataset_name}.tsv")
        
        # Download TSV if not exists
        if not os.path.exists(tsv_file):
            print(f"TSV file not found, downloading {dataset_name}...")
            self.download_tsv(dataset_name)
        
        # Check for large file and use local version if available
        if os.path.exists(tsv_file) and os.stat(tsv_file).st_size / 2 ** 30 > 1:
            local_tsv_file = os.path.join(self.dataset_dir, dataset_name, f"{dataset_name}_local.tsv")
            if os.path.exists(local_tsv_file):
                tsv_file = local_tsv_file
            else:
                raise FileNotFoundError(f"Large TSV file detected but local version not found: {local_tsv_file}")

        # Load TSV file using pandas
        try:
            dataset = pd.read_csv(tsv_file, sep='\t')
            # Ensure the dataset has required columns
            if 'index' not in dataset.columns:
                dataset['index'] = range(len(dataset))
            return dataset
            
        except Exception as e:
            raise RuntimeError(f"Failed to load TSV file '{tsv_file}': {e}")

    def _load_multipart_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load and concatenate multiple part files for datasets like MicroBench.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset to load
            
        Returns
        -------
        pd.DataFrame
            Concatenated DataFrame from all parts
        """        
        # Get dataset configuration
        config = MULTIPART_DATASET_CONFIG[dataset_name]
        pattern = config["filename_pattern"]
        start_idx = config["start_idx"]
        end_idx = config["end_idx"]

        dataframes = []
        
        # Check if all part files exist, if not download
        missing_file = False
        for part_num in range(start_idx, end_idx+1):
            part_file = os.path.join(self.dataset_dir, dataset_name, pattern.format(part_num))
            if not os.path.exists(part_file):
                missing_file = True
                break
        
        if missing_file:
            print(f"Missing file for {dataset_name}, downloading...")
            self.download_tsv(dataset_name)
        
        for part_num in range(start_idx, end_idx+1):
            tsv_file = os.path.join(self.dataset_dir, dataset_name, pattern.format(part_num))
            
            if not os.path.exists(tsv_file):
                raise FileNotFoundError(f"TSV file not found: {tsv_file}")
            
            df = pd.read_csv(tsv_file, sep='\t')
            dataframes.append(df)
        
        # Concatenate all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Ensure the dataset has required columns
        if 'index' not in combined_df.columns:
            combined_df['index'] = range(len(combined_df))
        
        return combined_df

    def _load_concat_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load and concatenate multiple part files for datasets like MicroBench.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset to load
        
        Returns
        -------
        pd.DataFrame
            Concatenated DataFrame from all parts
        """
        dataset_list = CONCAT_DATASET_SETS[dataset_name]
        dataframes = []
        for sub_dataset_name in dataset_list:
            sub_dataset = self._load_single_tsv_file(sub_dataset_name)
            sub_dataset['sub_dataset'] = [sub_dataset_name] * len(sub_dataset)
            dataframes.append(sub_dataset)

        # Concatenate all dataframes
        return pd.concat(dataframes, ignore_index=True)


    def _load_raw_data(self) -> Any:
        """Load raw data from TSV files in the dataset directory.
        
        Returns
        -------
        Any
            Pandas DataFrame containing the dataset
        """
        # Validate environment
        if not self.dataset_dir:
            raise ValueError("VLMEVALKIT_DATASET_DIR environment variable is not set")
        
        if not os.path.exists(self.dataset_dir):
            raise ValueError(f"Dataset directory does not exist: {self.dataset_dir}")
        
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
        media = self._extract_media_from_sample(raw_item, index)
        
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
        
        # Clean up original fields
        if 'image' in sample:
            sample.pop("image")
        
        return sample
