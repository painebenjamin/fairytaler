from typing import Union, Tuple, Sequence
from numpy.typing import ArrayLike
from torch import Tensor

from .util import SeedType # re-exported

__all__ = [
    "SingleAudioType",
    "AudioType",
    "AudioResultType",
    "SeedType",
]
SingleAudioType = Union[
    str, bytes, bytearray, ArrayLike, Tensor, Sequence[Tuple[Union[int, float], ...]]
]
AudioType = Union[
    SingleAudioType,
    Sequence[SingleAudioType],
]
AudioResultType = Union[
    SingleAudioType,
    Sequence[SingleAudioType],
    Sequence[Tuple[SingleAudioType, ...]],
]
SeedType = SeedType # Silence importchecker
