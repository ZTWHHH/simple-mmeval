import os
import requests

def get_hf_tsv_url(dataset_name: str) -> str:
    """Get Hugging Face URL for a dataset.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    
    Returns
    -------
    str
        Hugging Face URL for the dataset
    """
    return f"https://huggingface.co/datasets/mm-eval/VLMEvalKit/resolve/main/{dataset_name}.tsv"


def download_tsv(file_url: str, dataset_name: str, dataset_dir: str) -> str:
    """Download TSV file from URL to dataset directory with specified name.
    
    Parameters
    ----------
    file_url : str
        URL to download TSV file from
    dataset_name : str
        Name of the dataset (will be saved as dataset_name.tsv)
    dataset_dir : str
        Directory to save the downloaded file
        
    Raises
    ------
    RuntimeError
        If download fails
    """
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create filename based on dataset name
    filename = f"{dataset_name}.tsv"
    file_path = os.path.join(dataset_dir, filename)
    
    try:
        response = requests.get(file_url, stream=True)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
    except Exception as e:
        raise RuntimeError(f"Failed to download from URL {file_url}: {e}") 