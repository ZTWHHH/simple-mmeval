series_mapping = {
    "qwenvl2d5": ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct"],
    "xinyuanvl": ["Xinyuan-VL-2B"]
}

series_infer_env_mapping = {
    "qwenvl2d5": {
        "env": "vllm",
        "infer_file": "qwenvl2d5.py",
    },
    "xinyuanvl": {
        "env": "/root/yuexuanliu/simple-mmeval/envs/Xinyuan-VL-2B",
        "infer_file": "xinyuanvl.py",
    }
}