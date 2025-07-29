import os
import requests
import pandas as pd
from typing import Dict, Any
from dotenv import load_dotenv
from mmeval.data.tsv import TSVDataset

load_dotenv(dotenv_path=".env", override=True)

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
   'GOBench',
   'GQA_TestDev_Balanced',
   'HRBench4K',
   'HRBench8K',
   'InfoVQA_TEST',
   'InfoVQA_VAL',
   'LEGO',
   'LLaVABench',
   'LogicVista',
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

class VLMEvalKitDataset():
    """Dataset class for loading VLMEvalKit datasets.
    
    This class provides a bridge between VLMEvalKit's dataset files and the mmeval Dataset interface.
    """
    
    def __init__(self, args):
        """Initialize the VLMEvalKit dataset with parallel processing support.
        
        Parameters
        ----------
        args: argparse.Namespace
            Arguments from argparse containing dataset configuration
        """
        self.dataset_dir = os.getenv('DATASET_DIR')
        self.dataset_url = None

        if args.dataset.startswith("evalkit@"):
            args.dataset = args.dataset.split("@")[-1]
        elif args.dataset.startswith("http"):
            self.dataset_url = args.dataset
            args.dataset = args.dataset.split('/')[-1].replace('.tsv', '')
        elif os.path.exists(os.path.join(self.dataset_dir, args.dataset)):
            args.dataset = args.dataset.split('/')[-1].replace('.tsv', '')

        self.dataset_name = args.dataset
        self.args = args
    
    def _get_hf_tsv_url(self,dataset_name: str) -> str:
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

    def _download_tsv(self, file_url: str, dataset_name: str, dataset_dir: str) -> str:
        """Download TSV file from URL to dataset directory with specified name.
        
        Parameters
        ----------
        file_url : str
            URL to download TSV file from
        dataset_name : str
            Name of the dataset (will be saved as dataset_name.tsv)
        dataset_dir : str
            Directory to save the downloaded file
        """
        file_path = os.path.join(dataset_dir, f"{dataset_name}.tsv")

        response = requests.get(file_url, stream=True)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def _load_single_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load single dataset file for VLMEvalKit specific datasets.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset to load

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame from the single dataset file
        """
        file_path = os.path.join(self.dataset_dir, f"{dataset_name}.tsv")

        if not os.path.exists(file_path):
            self._download_tsv(self._get_hf_tsv_url(dataset_name), dataset_name, self.dataset_dir)

    def _load_multipart_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load multipart dataset files for VLMEvalKit specific datasets.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset to load
            
        Returns
        -------
        pd.DataFrame
            Loaded and merged DataFrame from all parts, saved as dataset_name.tsv
        """
        file_path = os.path.join(self.dataset_dir, f"{dataset_name}.tsv")

        if not os.path.exists(file_path):
            config = VLMEVALKIT_MULTIPART_DATASET_CONFIG[dataset_name]
            pattern = config["filename_pattern"]
            
            # Load and merge all parts into DataFrames
            dataframes = []
            for part_idx in range(config["start_idx"], config["end_idx"]+1):
                file_path = os.path.join(self.dataset_dir, f"{pattern.format(part_idx)}.tsv")
                if not os.path.exists(file_path):
                    self._download_tsv(self._get_hf_tsv_url(pattern.format(part_idx)), pattern.format(part_idx), self.dataset_dir)
                dataset = pd.read_csv(file_path, sep='\t')
                dataframes.append(dataset)
            
            # Concatenate all dataframes
            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df.to_csv(file_path, sep='\t', index=False, chunksize=100000)

    def _load_concat_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load multiple datasets for VLMEvalKit composite datasets.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset to load
        
        Returns
        -------
        pd.DataFrame
            Concatenated DataFrame from all parts, saved as dataset_name.tsv
        """
        file_path = os.path.join(self.dataset_dir, f"{dataset_name}.tsv")

        if not os.path.exists(file_path):
            dataset_list = VLMEVALKIT_CONCAT_DATASET_SETS[dataset_name]
            dataframes = []
            for sub_dataset_name in dataset_list:
                sub_file_path = os.path.join(self.dataset_dir, f"{sub_dataset_name}.tsv")
                if not os.path.exists(sub_file_path):
                    self._download_tsv(self._get_hf_tsv_url(sub_dataset_name), sub_dataset_name, self.dataset_dir)
                sub_dataset = pd.read_csv(sub_file_path, sep='\t')
                sub_dataset['sub_dataset'] = [sub_dataset_name] * len(sub_dataset)
                dataframes.append(sub_dataset)

            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df.to_csv(file_path, sep='\t', index=False, chunksize=100000)

    def load_dataset(self) -> Any:
        """Load raw data from TSV files.
        
        Returns
        -------
        Any
            Pandas DataFrame containing the dataset
        """
        # Validate environment
        os.makedirs(self.dataset_dir, exist_ok=True)

        if self.dataset_url:
            self._download_tsv(self.dataset_url, self.dataset_name, self.dataset_dir)
        if self.dataset_name in ["MicroBench", "XLRS-Bench-lite", "OmniEarth-Bench", "OmniMedVQA"]:
            self._load_multipart_dataset(self.dataset_name)
        elif self.dataset_name in ["MMMB", "MTL_MMBench_DEV"]:
            self._load_concat_dataset(self.dataset_name)
        else:
            if self.dataset_name in VLMEVALKIT_DATASET_LIST:
                self._load_single_dataset(self.dataset_name)
            else:
                raise ValueError(f"Dataset {self.dataset_name} is not supported.")
        
        # Handle single file datasets
        return TSVDataset(self.args)


