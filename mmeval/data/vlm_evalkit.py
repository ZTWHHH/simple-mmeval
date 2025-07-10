import sys
import os
import re
from typing import Dict, Any, Optional
from .dataset import Dataset

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
    """Dataset class for loading VLMEvalKit datasets using build_dataset function.
    
    This class provides a bridge between VLMEvalKit's dataset system and the mmeval Dataset interface.
    It supports all VLMEvalKit datasets including image, video, and text datasets with their specific
    configurations and model requirements.
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
        self.parallel_num = args.parallel_per_task
        self.rank = args.rank
        
        # Initialize parent class with parallel processing parameters
        super().__init__(
            self.dataset_name, 
            parallel_num=self.parallel_num, 
            rank=self.rank
        )
    
    def _load_raw_data(self) -> Any:
        """Load raw data using VLMEvalKit's build_dataset function.
        
        Returns
        -------
        Any
            VLMEvalKit dataset object that supports indexing and length
        """
        # Add VLMEvalKit to Python path if not already there
        try:
            vlm_evalkit_path = os.path.join(os.path.dirname(__file__), '..', '..', 'VLMEvalKit')
            if vlm_evalkit_path not in sys.path:
                sys.path.insert(0, vlm_evalkit_path)
            from vlmeval.dataset import build_dataset
        except ImportError as e:
            raise ImportError(f"Failed to import VLMEvalKit: {e}. Please ensure VLMEvalKit is available.")
        
        # Build VLMEvalKit dataset with model parameter if needed
        dataset_kwargs = {}
        
        # Check if this dataset requires a model parameter
        model_required_datasets = {
            'MMLongBench_DOC', 'DUDE', 'DUDE_MINI', 
            'SLIDEVQA', 'SLIDEVQA_MINI'
        }
        
        if self.dataset_name in model_required_datasets:
            # Use a default model that's commonly supported
            dataset_kwargs['model'] = 'GPT4V'
        
        try:
            vlm_dataset = build_dataset(self.dataset_name, **dataset_kwargs)
            if vlm_dataset is None:
                raise ValueError(f"Failed to build VLMEvalKit dataset: {self.dataset_name}")
            return vlm_dataset
        except Exception as e:
            raise RuntimeError(f"Failed to build VLMEvalKit dataset '{self.dataset_name}': {e}")

    def _process_sample(self, index: int) -> Dict[str, Any]:
        """Process a raw VLMEvalKit sample to match format.
        
        Parameters
        ----------
        index : int
            Global index of the sample in the dataset
            
        Returns
        -------
        Dict[str, Any]
            Processed sample with unified format
        """
  
        # Get raw item from VLMEvalKit dataset
        raw_item = self._raw_dataset[index]
        
        # Build VLM prompt using the dataset's method with the complete data row
        vlm_prompt = self._raw_dataset.build_prompt(raw_item)

        # Start with all original data
        sample = dict(raw_item)
        
        # Initialize processed sample with id
        sample['id'] = index
        
        # Extract media paths and build placeholder prompt
        media_path_list = []
        prompt_list = []
        
        if isinstance(vlm_prompt, list):
            # Handle list format prompts with interleaved media and text
            for item in vlm_prompt:
                if isinstance(item, dict):
                    if item.get('type') == 'image':
                        media_path_list.append(item.get('value', ''))
                    elif item.get('type') == 'video':
                        media_path_list.append(item.get('value', ''))
                    elif item.get('type') == 'text':
                        prompt_list.append(item.get('value', ''))
                    else:
                        raise ValueError(f"Unknown item type: {item}")
                else:
                    raise ValueError(f"Item is not a dictionary: {item}")
            
            # Join text parts
            text_prompt = ' '.join(prompt_list).strip()
            
            # Normalize image placeholders in the text
            prompt = normalize_image_placeholders(text_prompt, len(media_path_list))
        
        # Set media and prompt fields
        sample['media'] = media_path_list
        sample['prompt'] = prompt
        
        # Store the original VLM prompt for reference
        sample['vlm_prompt'] = vlm_prompt

        # Remove the original image field since we now have media
        sample.pop("image", None)
        
        return sample
