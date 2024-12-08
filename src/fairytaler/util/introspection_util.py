from __future__ import annotations

from typing import Any, Dict, Type, List, TypeGuard, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch

__all__ = [
    "has_type_name",
    "is_numpy_array",
    "is_torch_tensor",
]

MRO_TYPE_NAME_CACHE: Dict[Type[Any], List[str]] = {}
def has_type_name(maybe_type: Any, type_name: str) -> bool:
    """
    Check if a type has a specific type name.
    This allows us to check if a type is a subclass of a
    specific type without having to import the type.
    """
    global MRO_TYPE_NAME_CACHE
    if not isinstance(maybe_type, type):
        maybe_type = type(maybe_type)
    if maybe_type not in MRO_TYPE_NAME_CACHE:
        MRO_TYPE_NAME_CACHE[maybe_type] = [
            mro_type.__name__.lower() for mro_type in maybe_type.mro()
        ]
    return type_name.lower() in MRO_TYPE_NAME_CACHE[maybe_type]

def is_numpy_array(maybe_array: Any) -> TypeGuard[np.ndarray[Any,Any]]:
    """
    Check if an object is a numpy array.
    """
    return has_type_name(maybe_array, "ndarray")

def is_torch_tensor(maybe_tensor: Any) -> TypeGuard[torch.Tensor]:
    """
    Check if an object is a torch tensor.
    """
    return has_type_name(maybe_tensor, "tensor")
