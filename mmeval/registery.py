import os

env_dir = os.getenv('ENV_DIR') or ""

series_mapping = {
    "qwenvl2d5": ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct"],
    "gemma3": ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"],
    "llava_ov": ["llava-onevision-qwen2-0.5b-si-hf", 
                 "llava-onevision-qwen2-0.5b-ov-hf", 
                 "llava-onevision-qwen2-7b-si-hf", 
                 "llava-onevision-qwen2-7b-ov-hf", 
                 "llava-onevision-qwen2-72b-ov-hf", 
                 "llava-onevision-qwen2-72b-si-hf", 
                 "llava-onevision-qwen2-72b-ov-chat-hf", 
                 "llava-onevision-qwen2-7b-ov-chat-hf"],
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
    "llava_ov": {
        "env": os.path.join(env_dir, "llava_ov"),
        "infer_file": "llava_ov.py",
    }
}