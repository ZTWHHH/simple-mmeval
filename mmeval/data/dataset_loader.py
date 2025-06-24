from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Tuple, Union

from PIL import Image

from .dataset import Dataset

__all__ = [
    "BaseDatasetLoader",
]


class BaseDatasetLoader(ABC):
    """Abstract base class for all dataset loaders.

    This base class provides:
    1. Dataset creation and management
    2. Unified interface across all dataset types
    3. ID uniqueness enforcement
    4. Image resizing with padding utilities

    Subclasses must implement data loading and processing methods.
    """

    def __init__(self):
        """Initialize the dataset loader."""
        self.dataset: Optional[Dataset] = None
        self._used_ids: Set[str] = set()

    def _ensure_unique_id(self, original_id: str) -> str:
        """Ensure ID uniqueness by adding suffix if needed.
        
        The first occurrence keeps the original ID unchanged.
        Subsequent duplicates get suffixes: _01, _02, etc.
        
        Parameters
        ----------
        original_id : str
            The original ID that might be duplicate
            
        Returns
        -------
        str
            A unique ID, with suffix added if necessary
        """
        if original_id not in self._used_ids:
            self._used_ids.add(original_id)
            return original_id
        
        # Find the next available suffix for duplicates
        counter = 1
        while True:
            unique_id = f"{original_id}_{counter:02d}"
            if unique_id not in self._used_ids:
                self._used_ids.add(unique_id)
                return unique_id
            counter += 1

    def create_dataset(self, **kwargs) -> Dataset:
        """Create and populate the dataset.

        Parameters
        ----------
        **kwargs
            Additional arguments specific to the loader type
            
        Returns
        -------
        Dataset
            The created and populated dataset
        """
        # Create dataset instance
        dataset_name = self._get_dataset_name()
        self.dataset = Dataset(dataset_name)

        # Reset used IDs for this creation
        self._used_ids.clear()

        # Load and process raw data
        raw_data = self._load_raw_data(**kwargs)
        
        for item in raw_data:
            processed = self._process_sample(item)
            # Ensure ID uniqueness
            if 'id' in processed:
                processed['id'] = self._ensure_unique_id(processed['id'])
            self.dataset.add_sample(processed)

        return self.dataset

    def resize_image(
        self, 
        image: Image.Image, 
        size: Union[int, Tuple[int, int]], 
        padding: bool = True,
        fill_color: Union[int, Tuple[int, int, int]] = 0
    ) -> Image.Image:
        """Resize a PIL Image object with optional padding.

        Parameters
        ----------
        image : Image.Image
            The PIL Image object to resize
        size : Union[int, Tuple[int, int]]
            Target size. If int, resize to (size, size). If tuple, resize to (width, height)
        padding : bool, default=True
            If True, maintain aspect ratio and pad to target size.
            If False, directly resize to target size.
        fill_color : Union[int, Tuple[int, int, int]], default=0
            Fill color for padding. Single int for grayscale, tuple for RGB.

        Returns
        -------
        Image.Image
            The resized PIL Image object
        """
        if isinstance(size, int):
            target_size = (size, size)
        else:
            target_size = size

        if not padding:
            # Direct resize to target size
            return image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Resize with padding to maintain aspect ratio
        original_width, original_height = image.size
        target_width, target_height = target_size
        
        # Calculate scale to fit within target size
        scale = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with target size and fill color
        padded_image = Image.new(image.mode, target_size, fill_color)
        
        # Calculate position to center the resized image
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Paste the resized image onto the padded image
        padded_image.paste(resized_image, (x_offset, y_offset))
        
        return padded_image

    def prepare_circular_data(self, dataset: Dataset, **kwargs) -> Any:
        """Prepare dataset for circular evaluation by generating circular variants.
        
        This method transforms the dataset to create circular variants of questions
        and choices that can be used for robust circular evaluation strategies.
        
        Parameters
        ----------
        dataset : Dataset
            The dataset to prepare for circular evaluation
        **kwargs
            Additional arguments for circular data preparation configuration
            
        Returns
        -------
        Any
            Results from circular data preparation (implementation dependent)
        """
        # Placeholder for circular data preparation implementation
        raise NotImplementedError("Circular data preparation not yet implemented")

    # Abstract methods that subclasses must implement
    @abstractmethod
    def _load_raw_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Load raw data from the data source.
        
        Parameters
        ----------
        **kwargs
            Additional arguments specific to the loader type
            
        Returns
        -------
        List[Dict[str, Any]]
            Raw data items
        """

    @abstractmethod
    def _process_sample(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a raw dataset sample into the common schema. Set the target output.
        
        Parameters
        ----------
        item : Dict[str, Any]
            Raw data item
            
        Returns
        -------
        Dict[str, Any]
            Processed data item following the common schema
        """

    @abstractmethod
    def _get_dataset_name(self) -> str:
        """Return the dataset name for this loader.
        
        Returns
        -------
        str
            The name of the dataset
        """
