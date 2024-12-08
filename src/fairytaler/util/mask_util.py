from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = [
    "mask_from_seq_lengths",
    "mask_from_start_end_indices",
    "mask_from_frac_lengths",
]

def mask_from_seq_lengths(
    t: torch.Tensor,
    length: Optional[int]=None
) -> torch.Tensor:
    """
    Convert a seq_lengths tensor to a mask tensor

    :param t: A tensor of seq_lengths
    :param length: The length of the mask
    """
    import torch

    if length is None:
        length = t.max().item() # type: ignore[assignment]

    mask = torch.arange(length, device=t.device, dtype=t.dtype) # type: ignore[arg-type]
    return mask[None, :] < t[:, None]

def mask_from_start_end_indices(
    seq_length: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    """
    Convert start and end indices to a mask

    :param seq_length: The length of the sequence
    :param start: The start indices
    :param end: The end indices
    """
    import torch
    seq = torch.arange(seq_length.max().item(), device=seq_length.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask

def mask_from_frac_lengths(
    seq_length: torch.Tensor,
    frac_lengths: torch.Tensor,
    generator: Optional[torch.Generator]=None
) -> torch.Tensor:
    """
    Convert fractional lengths to a mask

    :param seq_length: The length of the sequence
    :param frac_lengths: The fractional lengths
    :param generator: The random number generator, optional
    """
    import torch
    lengths = (frac_lengths * seq_length).long()
    max_start = seq_length - lengths

    rand = torch.randn(
        *frac_lengths.shape,
        device=frac_lengths.device,
        dtype=frac_lengths.dtype,
        generator=generator
    )
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths
    
    return mask_from_start_end_indices(seq_length, start, end)
