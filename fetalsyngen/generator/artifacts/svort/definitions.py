from typing import (
    Dict,
    List,
    Any,
    Union,
)
from os import PathLike
import torch

PathType = Union[str, PathLike[str]]
DeviceType = Union[torch.device, str, None]
