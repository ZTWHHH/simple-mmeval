import os
import tempfile
import requests
from filelock import FileLock


def download_tsv(file_url: str, save_path: str) -> None:
    """Download TSV file from URL to save_path atomically.
    
    Parameters
    ----------
    file_url : str
        URL to download TSV file from
    save_path : str
        Path to save the downloaded file
    """
    file_dir = os.path.dirname(save_path)
    os.makedirs(file_dir, exist_ok=True)
    
    lock_path = save_path + ".lock"
    with FileLock(lock_path):
        if not os.path.exists(save_path):
            response = requests.get(file_url, stream=True)
            response.raise_for_status()
            
            fd, tmp_path = tempfile.mkstemp(dir=file_dir, suffix=".tsv.tmp")
            try:
                with os.fdopen(fd, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=100000):
                        f.write(chunk)
                os.replace(tmp_path, save_path)
            except BaseException:
                os.unlink(tmp_path)
                raise
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass


def write_tsv(df, save_path: str, **csv_kwargs) -> None:
    """Write a DataFrame to TSV atomically (temp file + os.replace).
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to write
    save_path : str
        Final destination path
    **csv_kwargs
        Forwarded to ``df.to_csv()``
    """
    file_dir = os.path.dirname(save_path)
    os.makedirs(file_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=file_dir, suffix=".tsv.tmp")
    try:
        with os.fdopen(fd, 'w') as f:
            df.to_csv(f, **csv_kwargs)
        os.replace(tmp_path, save_path)
    except BaseException:
        os.unlink(tmp_path)
        raise
