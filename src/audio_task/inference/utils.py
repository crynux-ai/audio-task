from __future__ import annotations

import os
import platform
from typing import Any, Dict

import torch

from audio_task.config import Config, get_config


def get_accelerator():
    if platform.system() == "Darwin":
        try:
            import torch.mps

            return "mps"
        except ImportError:
            pass

    try:
        import torch.cuda

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    return "cpu"


def load_model_kwargs(config: Config | None = None) -> Dict[str, Any]:
    """
    generate model kwargs from config.
    config may contains:
        - cache_dir
        - proxies
    """
    if config is None:
        config = get_config()

    res = {}
    if config.data_dir is not None:
        res["cache_dir"] = config.data_dir.models.huggingface
    if config.proxy is not None and config.proxy.host != "":
        if "://" in config.proxy.host:
            scheme, host = config.proxy.host.split("://", 2)
        else:
            scheme, host = "", config.proxy.host
        
        proxy_str = ""
        if scheme != "":
            proxy_str += f"{scheme}://"
        if config.proxy.username != "":
            proxy_str += f"{config.proxy.username}:{config.proxy.password}@"
        proxy_str += f"{host}:{config.proxy.port}"

        res["proxies"] = {"http": proxy_str, "https": proxy_str}

    return res


def use_deterministic_mode():
    r"""
    use deterministic mode
    """
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
