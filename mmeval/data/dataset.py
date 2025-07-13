from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Tuple, Union

from PIL import Image

__all__ = ["Dataset"]


class Dataset(ABC):
    """A unified dataset class for loading, processing, and storing dataset samples.
    
    This class provides:
    1. Lazy data loading
    2. Memory-efficient processing
    3. Standard iteration helpers (__iter__, __getitem__, __len__)
    4. ID uniqueness enforcement
    5. Image resizing utilities
    6. Parallel processing support
    7. Circular data preparation placeholder
    """
    
    def __init__(
        self, 
        dataset_name: str, 
        parallel_num: int = 1,
        rank: int = 0,
        **kwargs
    ):
        """Initialize the dataset with lazy loading and parallel processing support.
        
        Parameters
        ----------
        dataset_name : str
            Name/identifier for this dataset
        parallel_num : int, default=1
            Total number of parallel workers/processes
        rank : int, default=0
            Current worker rank (0-indexed, must be < parallel_num)
        **kwargs
            Additional arguments for data loading
        """
        self.name = dataset_name
        self.parallel_num = parallel_num
        self.rank = rank
        self._raw_dataset = None
        
        # Validate parallel processing parameters
        self._validate_parallel_params()
        
        # Load raw dataset metadata only (no processing)
        self._raw_dataset = self._load_raw_data(**kwargs)
        
        # Calculate parallel processing indices
        self._setup_parallel_indices()

        # Create a global-to-local index mapping for convenience
        self.data_index = range(len(self._raw_dataset)) if self._raw_dataset is not None else range(0)
    
    def _validate_parallel_params(self):
        """Validate parallel processing parameters."""
        if not isinstance(self.parallel_num, int) or self.parallel_num < 1:
            raise ValueError(f"parallel_num must be a positive integer, got {self.parallel_num}")
        
        if not isinstance(self.rank, int) or self.rank < 0 or self.rank >= self.parallel_num:
            raise ValueError(f"rank must be an integer in range [0, {self.parallel_num}), got {self.rank}")
    
    def _setup_parallel_indices(self):
        """Setup indices for parallel processing based on rank and parallel_num."""
        if self._raw_dataset is None:
            self._parallel_indices = []
            self._parallel_length = 0
            return
        
        total_length = len(self._raw_dataset)
        
        # Calculate chunk size for each worker
        chunk_size = total_length // self.parallel_num
        remainder = total_length % self.parallel_num
        
        # Calculate start and end indices for this rank
        start_idx = self.rank * chunk_size + min(self.rank, remainder)
        end_idx = start_idx + chunk_size + (1 if self.rank < remainder else 0)
        
        # Store indices assigned to this worker
        self._parallel_indices = list(range(start_idx, min(end_idx, total_length)))
        self._parallel_length = len(self._parallel_indices)
    
    def get_global_index(self, local_index: int) -> int:
        """Convert local index (for this worker) to global dataset index.
        
        Parameters
        ----------
        local_index : int
            Local index within this worker's data subset
            
        Returns
        -------
        int
            Global index in the original dataset
        """
        if local_index < 0:
            local_index = self._parallel_length + local_index
        
        if local_index < 0 or local_index >= self._parallel_length:
            raise IndexError(f"Local index {local_index} out of range [0, {self._parallel_length}]")
        
        return self._parallel_indices[local_index]
    
    def get_parallel_info(self) -> Dict[str, Any]:
        """Get information about parallel processing setup.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing parallel processing information
        """
        return {
            "parallel_num": self.parallel_num,
            "rank": self.rank,
            "local_length": self._parallel_length,
            "global_length": len(self._raw_dataset) if self._raw_dataset else 0,
            "start_index": self._parallel_indices[0] if self._parallel_indices else None,
            "end_index": self._parallel_indices[-1] if self._parallel_indices else None
        }

    def resize_image(
        self, 
        image: Image.Image, 
        size: Union[int, Tuple[int, int]], 
        padding: bool = True,
        fill_color: str = "black"
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
        fill_color : str, default="black"
            Fill color for padding. Common colors: "black", "white", "red", "green", "blue", "gray"

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

    def prepare_circular_data(self, **kwargs) -> Any:
        """Prepare dataset for circular evaluation by generating circular variants.
        
        Parameters
        ----------
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
    def _load_raw_data(self, **kwargs) -> Any:
        """Load raw data from the data source.
        
        Returns a dataset object that supports indexing and length,
        but doesn't necessarily load all data into memory at once.
        
        Parameters
        ----------
        **kwargs
            Additional arguments specific to the dataset type
            
        Returns
        -------
        Any
            Dataset object that supports indexing (dataset[i]) and len()
        """

    @abstractmethod
    def _process_sample(self, index: int) -> Dict[str, Any]:
        """Process sample by index - to be implemented by subclasses.
        
        Subclasses should:
        1. Get raw item: raw_item = self._raw_dataset[index]
        2. Process the raw item into common schema
        
        Parameters
        ----------
        index : int
            Global sample index (not local to this worker)
            
        Returns
        -------
        Dict[str, Any]
            Processed data item following the common schema
        """

    # Iteration helpers with lazy loading and parallel processing
    def __iter__(self):
        """Iterate over dataset samples assigned to this worker with lazy loading."""
        for local_index in range(self._parallel_length):
            global_index = self.get_global_index(local_index)
            yield self._process_sample(global_index)

    def __getitem__(self, index):
        """Get sample by local index with lazy loading.
        
        Parameters
        ----------
        index : int
            Local index within this worker's data subset
            
        Returns
        -------
        Dict[str, Any]
            Processed sample
        """
        global_index = self.get_global_index(index)
        return self._process_sample(global_index)

    def __len__(self):
        """Get number of samples assigned to this worker."""
        return self._parallel_length
    
    def get_global_length(self):
        """Get total number of samples in the original dataset."""
        return len(self._raw_dataset) if self._raw_dataset else 0
    
    def __repr__(self):
        if self.parallel_num > 1:
            return f"Dataset(name='{self.name}', rank={self.rank}/{self.parallel_num}, local_samples={len(self)}, global_samples={self.get_global_length()})"
        else:
            return f"Dataset(name='{self.name}', samples={len(self)})"
    
    def __str__(self):
        if self.parallel_num > 1:
            return f"{self.name} dataset - rank {self.rank}/{self.parallel_num} with {len(self)}/{self.get_global_length()} samples"
        else:
            return f"{self.name} dataset with {len(self)} samples" 