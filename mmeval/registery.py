import os

env_dir = os.getenv('ENV_DIR') or ""

series_mapping = {
    "qwenvl2d5": ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct"],
    "gemma3": ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"],
    "llava": ["llava-1.5-7b-hf", "llava-1.5-13b-hf", "bakLlava-v1-hf"],
    "llava_next": ["llava-v1.6-mistral-7b-hf", "llava-v1.6-vicuna-7b-hf", "llava-v1.6-vicuna-13b-hf", "llava-v1.6-34b-hf", "llama3-llava-next-8b-hf", "llava-next-72b-hf", "llava-next-110b-hf"],
    "glm_4v": ["glm-4v-9b"]
}

series_infer_env_mapping = {
    "qwenvl2d5": {
        "env": os.path.join(env_dir, "vllm"),
        "infer_file": "qwenvl2d5.py",
    }, 
    "gemma3": {
        "env": os.path.join(env_dir, "gemma3"),
        "infer_file": "gemma3.py",
    },
    "llava": {
        "env": os.path.join(env_dir, "llava"),
        "infer_file": "llava.py",
    },
    "llava_next": {
        "env": os.path.join(env_dir, "llava_next"),
        "infer_file": "llava_next.py",
    },
    "glm_4v": {
        "env": os.path.join(env_dir, "glm_4v"),
        "infer_file": "glm_4v.py",
    }
}