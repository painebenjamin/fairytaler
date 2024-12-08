import torch

__all__ = ["Module"]

class Module(torch.nn.Module):
    """
    A simple wrapper class that extends torch.nn.Module.
    """
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("dummy", torch.tensor(0.0), persistent=False)

    @property
    def device(self) -> torch.device:
        """
        Returns the device of the first parameter.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            return self.dummy.device # type: ignore[no-any-return]

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the first parameter.
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return self.dummy.dtype # type: ignore[no-any-return]

    @property
    def num_params(self) -> int:
        """
        Returns the number of parameters in the module.
        """
        return sum(p.numel() for p in self.parameters())

    @property
    def is_half(self) -> bool:
        """
        Returns whether the module is using half precision.
        """
        return self.dtype in [torch.float16, torch.bfloat16]
