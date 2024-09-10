from __future__ import annotations

import logging
from typing import Literal, cast

import torch
from transformers import Pipeline, TextToAudioPipeline, pipeline, set_seed

from audio_task import models
from audio_task.cache import ModelCache
from audio_task.config import Config

from .errors import wrap_error
from .key import generate_model_key
from .utils import load_model_kwargs, use_deterministic_mode

_logger = logging.getLogger(__name__)


@wrap_error
def run_task(
    args: models.AudioTaskArgs | None = None,
    *,
    model: str | None = None,
    prompt: str | None = None,
    duration: float = 30,
    generation_config: models.AudioGenerationConfig | None = None,
    seed: int = 0,
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
    quantize_bits: Literal[4, 8] | None = None,
    config: Config | None = None,
    model_cache: ModelCache[Pipeline] | None = None,
):
    if args is None:
        args = models.AudioTaskArgs.model_validate(
            {
                "model": model,
                "prompt": prompt,
                "duration": duration,
                "generation_config": generation_config,
                "seed": seed,
                "dtype": dtype,
                "quantize_bits": quantize_bits,
            }
        )

    _logger.info("Task starts")
    _logger.debug(f"task args: {args}")

    use_deterministic_mode()

    set_seed(args.seed)

    def load_model():
        _logger.info("Start loading pipeline")

        torch_dtype = None
        if args.dtype == "float16":
            torch_dtype = torch.float16
        elif args.dtype == "float32":
            torch_dtype = torch.float32
        elif args.dtype == "bfloat16":
            torch_dtype = torch.bfloat16

        model_kwargs = load_model_kwargs(config=config)

        if args.quantize_bits is None:
            model_kwargs["torch_dtype"] = torch_dtype
        if args.quantize_bits == 4:
            model_kwargs["load_in_4bit"] = True
        elif args.quantize_bits == 8:
            model_kwargs["load_in_8bit"] = True

        _logger.debug(f"model kwargs: {model_kwargs}")

        pipe = pipeline(
            "text-to-audio",
            model=args.model,
            trust_remote_code=True,
            use_fast=False,
            model_kwargs=dict(
                offload_folder="offload", offload_state_dict=True, **model_kwargs
            ),
        )
        _logger.info("Loading pipeline completes")
        return pipe

    key = generate_model_key(args)

    if model_cache is not None:
        pipe = model_cache.load(key, load_model)
    else:
        pipe = load_model()

    pipe = cast(TextToAudioPipeline, pipe)
    sr = pipe.sampling_rate
    assert sr is not None

    max_length = int(args.duration * sr)

    generate_kwargs = {
        "max_length": max_length,
        "do_sample": True,
        "top_k": 250,
        "top_p": 0.0,
        "temperature": 1.0,
        "guidance_scale": 3,
        "extend_stride": 18,
    }

    if args.generation_config is not None:
        for k, v in args.generation_config.items():
            if v is not None:
                generate_kwargs[k] = v

    output = pipe(args.prompt, forward_params=generate_kwargs)
    assert isinstance(output, dict)

    audio = output["audio"]
    sr = output["sampling_rate"]

    return audio, sr
