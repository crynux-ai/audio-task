from __future__ import annotations

import logging
import random
from typing import Literal, cast

import torch
from transformers import Pipeline, TextToAudioPipeline, pipeline, set_seed

from audio_task import models
from audio_task.cache import ModelCache
from audio_task.config import Config

from .errors import wrap_error
from .key import generate_model_key
from .utils import load_model_kwargs, use_deterministic_mode, get_accelerator

_logger = logging.getLogger(__name__)


@wrap_error
@torch.no_grad
def run_task(
    args: models.AudioTaskArgs | None = None,
    *,
    model: str | None = None,
    prompt: str | None = None,
    generation_config: models.AudioGenerationConfig | None = None,
    seed: int = -1,
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
    quantize_bits: Literal[4, 8] | None = None,
    config: Config | None = None,
    model_cache: ModelCache[Pipeline] | None = None,
):
    if seed == -1:
        seed = random.randint(1, 2 ** 32 - 1)

    if args is None:
        args = models.AudioTaskArgs.model_validate(
            {
                "model": model,
                "prompt": prompt,
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

        model_kwargs["torch_dtype"] = torch_dtype
        if args.quantize_bits == 4:
            model_kwargs["load_in_4bit"] = True
        elif args.quantize_bits == 8:
            model_kwargs["load_in_8bit"] = True

        _logger.debug(f"model kwargs: {model_kwargs}")

        device = None
        if args.quantize_bits is None:
            device = get_accelerator()

        pipe = pipeline(
            "text-to-audio",
            model=args.model,
            device=device,
            framework="pt",
            use_fast=False,
            trust_remote_code=True,
            model_kwargs=dict(
                offload_folder="offload",
                offload_state_dict=True,
                **model_kwargs
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
    generate_kwargs = {}

    if args.generation_config is not None:
        for k, v in args.generation_config.items():
            if v is not None:
                generate_kwargs[k] = v

    if pipe.model.can_generate():
        output = pipe(args.prompt, generate_kwargs=generate_kwargs)
    else:
        output = pipe(args.prompt)
    assert isinstance(output, dict)

    audio = output["audio"]
    sr = output["sampling_rate"]

    assert audio.ndim == 3
    assert audio.shape[0] == 1
    audio = audio[0].transpose(1, 0)

    return audio, sr
