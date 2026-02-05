"""
DC Tensor Operations

Module wrappers for functional tensor operations that don't have PyTorch
module equivalents. These are used by the functional replacer to replace
functional calls with patchable modules.

Forward format: [4*batch] = [pos; neg; 0; 0]
Backward format: [4*batch] = [delta_pp; delta_np; delta_pn; delta_nn]
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional, List, Union


class DCSplit(nn.Module):
    """
    Split tensor along a dimension for DC decomposition.

    Split is a linear operation - pos and neg are split identically.

    Args:
        split_size: Size of each split or list of sizes
        dim: Dimension to split along
    """
    
    def __init__(self, split_size: Union[int, List[int]], dim: int = 0):
        super().__init__()
        self.split_size = split_size
        self.dim = dim

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        return torch.split(x, self.split_size, dim=self.dim)

    def extra_repr(self) -> str:
        return f'split_size={self.split_size}, dim={self.dim}'


class DCChunk(nn.Module):
    """
    Chunk tensor into equal parts for DC decomposition.

    Chunk is a linear operation - pos and neg are chunked identically.

    Args:
        chunks: Number of chunks
        dim: Dimension to chunk along
    """
    
    def __init__(self, chunks: int, dim: int = 0):
        super().__init__()
        self.chunks = chunks
        self.dim = dim

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        return torch.chunk(x, self.chunks, dim=self.dim)

    def extra_repr(self) -> str:
        return f'chunks={self.chunks}, dim={self.dim}'


class DCCat(nn.Module):
    """
    Concatenate tensors for DC decomposition.

    Concatenation is a linear operation - pos and neg are concatenated identically.

    Args:
        dim: Dimension to concatenate along
    """
    
    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, tensors: List[Tensor]) -> Tensor:
        return torch.cat(tensors, dim=self.dim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class DCSlice(nn.Module):
    """
    Slice tensor for DC decomposition.

    Slicing is a linear operation - pos and neg are sliced identically.

    Args:
        dim: Dimension to slice
        start: Start index
        end: End index (exclusive)
    """
    
    def __init__(self, dim: int, start: Optional[int] = None, end: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.start = start
        self.end = end

    def forward(self, x: Tensor) -> Tensor:
        slices = [slice(None)] * x.dim()
        slices[self.dim] = slice(self.start, self.end)
        return x[tuple(slices)]

    def extra_repr(self) -> str:
        return f'dim={self.dim}, start={self.start}, end={self.end}'


