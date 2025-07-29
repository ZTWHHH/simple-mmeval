import os
from .utils import download_tsv, get_hf_tsv_url
from .local import LocalJSONDataset
from .vlm_evalkit import VLMEvalKitDataset, VLMEVALKIT_DATASET_LIST, VLMEVALKIT_MULTIPART_DATASET_CONFIG, VLMEVALKIT_CONCAT_DATASET_SETS


def load_dataset(args):
    """Load dataset based on the dataset specification.
    
    Supported formats:
    - local@json: Load from local JSON file
    - evalkit@dataset_name: Load VLMEvalKit dataset
    - evalkit@url: Load from remote TSV URL
    - dataset_name: Legacy support for VLMEvalKit datasets
    """
    dataset_dir = os.getenv('DATASET_DIR')

    if args.dataset == "local@json":
        return LocalJSONDataset(args)

    elif args.dataset.startswith("evalkit@http"):
        file_url = args.dataset[8:]  # Remove "evalkit@" prefix
        dataset_name = file_url.split('/')[-1].replace('.tsv', '')
        if not os.path.exists(os.path.join(dataset_dir, f"{dataset_name}.tsv")):  
            download_tsv(file_url, dataset_name, dataset_dir)
            print(f"Downloaded dataset {dataset_name} to {dataset_dir}")
        args.dataset = dataset_name
        return VLMEvalKitDataset(args)

    elif args.dataset.startswith("evalkit@"):
        dataset_name = args.dataset[8:]  # Remove "evalkit@" prefix
        if dataset_name not in VLMEVALKIT_DATASET_LIST:
            raise ValueError(f"Unsupported VLMEvalKit dataset: {args.dataset}")

        if dataset_name in VLMEVALKIT_MULTIPART_DATASET_CONFIG:
            config = VLMEVALKIT_MULTIPART_DATASET_CONFIG[dataset_name]
            pattern = config["filename_pattern"]
            start_idx = config["start_idx"]
            end_idx = config["end_idx"]

            for part_idx in range(start_idx, end_idx+1):
                tsv_file = os.path.join(dataset_dir, pattern.format(part_idx))
                if not os.path.exists(tsv_file):
                    file_url = get_hf_tsv_url(pattern.format(part_idx))
                    download_tsv(file_url, pattern.format(part_idx), dataset_dir)
                    print(f"Downloaded dataset {pattern.format(part_idx)} to {dataset_dir}")

        elif dataset_name in VLMEVALKIT_CONCAT_DATASET_SETS:
            dataset_list = VLMEVALKIT_CONCAT_DATASET_SETS[dataset_name]
            for sub_dataset_name in dataset_list:
                if not os.path.exists(os.path.join(dataset_dir, f"{sub_dataset_name}.tsv")):
                    file_url = get_hf_tsv_url(sub_dataset_name)
                    download_tsv(file_url, sub_dataset_name, dataset_dir)
                    print(f"Downloaded dataset {sub_dataset_name} to {dataset_dir}")

        else:
            file_url = get_hf_tsv_url(dataset_name)
            if not os.path.exists(os.path.join(dataset_dir, f"{dataset_name}.tsv")):
                download_tsv(file_url, dataset_name, dataset_dir)
                print(f"Downloaded dataset {dataset_name} to {dataset_dir}") 
        
        args.dataset = dataset_name
        return VLMEvalKitDataset(args)

    else:
        raise ValueError(f"Unsupported dataset specification: {args.dataset}") 