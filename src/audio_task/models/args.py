from typing import Literal, Optional

from annotated_types import MinLen
from pydantic import BaseModel
from typing_extensions import Annotated, TypedDict

NonEmptyString = Annotated[str, MinLen(1)]


class AudioGenerationConfig(TypedDict, total=False):
    do_sample: bool
    top_k: int
    top_p: float
    temperature: float
    guidance_scale: float


class AudioTaskArgs(BaseModel):
    model: NonEmptyString
    prompt: str

    duration: float = 30
    generation_config: Optional[AudioGenerationConfig] = None

    seed: int = 0
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto"
    quantize_bits: Optional[Literal[4, 8]] = None
