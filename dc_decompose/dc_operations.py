"""
DC Operation Modules

These modules wrap common tensor operations to make them compatible with
hook-based DC decomposition. Each operation is implemented as a module
that can be hooked by HookDecomposer.

For most operations, pos and neg streams are processed identically since
the operations are linear (reshape, permute, transpose, scalar mul/div).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, List, Union


class DCReshape(nn.Module):
    """
    Reshape/view module for DC decomposition.

    Reshape is a linear operation - both pos and neg streams
    are reshaped identically.

    Args:
        target_shape: Target shape (can include -1 for inference)
    """
    _dc_is_reshape = True

    def __init__(self, *target_shape):
        super().__init__()
        if len(target_shape) == 1 and isinstance(target_shape[0], (list, tuple)):
            self.target_shape = tuple(target_shape[0])
        else:
            self.target_shape = target_shape

    def forward(self, x: Tensor) -> Tensor:
        return x.view(*self.target_shape)

    def extra_repr(self) -> str:
        return f'target_shape={self.target_shape}'


class DCDynamicReshape(nn.Module):
    """
    Dynamic reshape module where shape is computed at runtime.

    Use set_shape() before forward pass to set the target shape.
    """
    _dc_is_reshape = True

    def __init__(self):
        super().__init__()
        self._target_shape: Optional[Tuple[int, ...]] = None

    def set_shape(self, *shape):
        """Set target shape for next forward pass."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            self._target_shape = tuple(shape[0])
        else:
            self._target_shape = shape

    def forward(self, x: Tensor) -> Tensor:
        if self._target_shape is None:
            raise RuntimeError("Target shape not set. Call set_shape() first.")
        return x.view(*self._target_shape)


class DCPermute(nn.Module):
    """
    Permute dimensions module for DC decomposition.

    Permute is a linear operation - both pos and neg streams
    are permuted identically.

    Args:
        dims: Permutation of dimensions
    """
    _dc_is_permute = True

    def __init__(self, *dims):
        super().__init__()
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            self.dims = tuple(dims[0])
        else:
            self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(*self.dims)

    def extra_repr(self) -> str:
        return f'dims={self.dims}'


class DCTranspose(nn.Module):
    """
    Transpose dimensions module for DC decomposition.

    Transpose is a linear operation - both pos and neg streams
    are transposed identically.

    Args:
        dim0: First dimension to transpose
        dim1: Second dimension to transpose
    """
    _dc_is_transpose = True

    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(self.dim0, self.dim1)

    def extra_repr(self) -> str:
        return f'dim0={self.dim0}, dim1={self.dim1}'


class DCContiguous(nn.Module):
    """
    Make tensor contiguous for DC decomposition.

    This is a no-op mathematically but ensures memory layout.
    """
    _dc_is_contiguous = True

    def forward(self, x: Tensor) -> Tensor:
        return x.contiguous()


class DCScalarMul(nn.Module):
    """
    Scalar multiplication for DC decomposition.

    For positive scalar s: pos_out = s * pos_in, neg_out = s * neg_in
    For negative scalar s: pos_out = |s| * neg_in, neg_out = |s| * pos_in

    Args:
        scalar: Scalar value to multiply by
    """
    _dc_is_scalar_mul = True

    def __init__(self, scalar: float):
        super().__init__()
        self.register_buffer('scalar', torch.tensor(scalar))
        self.register_buffer('is_negative', torch.tensor(scalar < 0))
        self.register_buffer('abs_scalar', torch.tensor(abs(scalar)))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scalar

    def extra_repr(self) -> str:
        return f'scalar={self.scalar.item():.6f}'


class DCScalarDiv(nn.Module):
    """
    Scalar division for DC decomposition.

    Division by scalar s is multiplication by 1/s.
    For positive s: pos_out = pos_in / s, neg_out = neg_in / s
    For negative s: pos_out = neg_in / |s|, neg_out = pos_in / |s|

    Args:
        scalar: Scalar value to divide by
    """
    _dc_is_scalar_div = True

    def __init__(self, scalar: float):
        super().__init__()
        self.register_buffer('scalar', torch.tensor(scalar))
        self.register_buffer('is_negative', torch.tensor(scalar < 0))
        self.register_buffer('abs_scalar', torch.tensor(abs(scalar)))

    def forward(self, x: Tensor) -> Tensor:
        return x / self.scalar

    def extra_repr(self) -> str:
        return f'scalar={self.scalar.item():.6f}'


class DCAdd(nn.Module):
    """
    Element-wise addition for DC decomposition.

    (a_pos - a_neg) + (b_pos - b_neg) = (a_pos + b_pos) - (a_neg + b_neg)

    This module stores the second operand's decomposition for the hook.
    """
    _dc_is_add = True

    def __init__(self):
        super().__init__()
        # Storage for second operand's DC decomposition
        self._dc_operand_pos: Optional[Tensor] = None
        self._dc_operand_neg: Optional[Tensor] = None

    def set_operand(self, b: Tensor, b_pos: Optional[Tensor] = None, b_neg: Optional[Tensor] = None):
        """Set the second operand for addition."""
        if b_pos is not None and b_neg is not None:
            self._dc_operand_pos = b_pos
            self._dc_operand_neg = b_neg
        else:
            self._dc_operand_pos = F.relu(b)
            self._dc_operand_neg = F.relu(-b)

    def set_operand_decomposed(self, b_pos: Tensor, b_neg: Tensor):
        """Set the second operand with pre-decomposed components."""
        self._dc_operand_pos = b_pos
        self._dc_operand_neg = b_neg

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return a + b


class DCSplit(nn.Module):
    """
    Split tensor along a dimension for DC decomposition.

    Split is a linear operation - pos and neg are split identically.

    Args:
        split_size: Size of each split or list of sizes
        dim: Dimension to split along
    """
    _dc_is_split = True

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
    _dc_is_chunk = True

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
    _dc_is_cat = True

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
    _dc_is_slice = True

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


class DCDropout(nn.Module):
    """
    Dropout for DC decomposition.

    In eval mode, dropout is identity.
    In train mode, the same mask is applied to both pos and neg.

    Args:
        p: Dropout probability
    """
    _dc_is_dropout = True

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.dropout = nn.Dropout(p)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x)

    def extra_repr(self) -> str:
        return f'p={self.p}'


class DCIdentity(nn.Module):
    """
    Identity module for DC decomposition.

    Useful as a placeholder or for skip connections.
    """
    _dc_is_identity = True

    def forward(self, x: Tensor) -> Tensor:
        return x


class DCEmbedding(nn.Module):
    """
    Embedding lookup for DC decomposition.

    The embedding weights are decomposed: W = W_pos - W_neg
    Output: pos = W_pos[indices], neg = W_neg[indices]

    Args:
        num_embeddings: Size of the dictionary
        embedding_dim: Dimension of embeddings
        padding_idx: If given, pads output with zeros at this index
    """
    _dc_is_embedding = True

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)

    def extra_repr(self) -> str:
        return f'{self.num_embeddings}, {self.embedding_dim}'


class DCGather(nn.Module):
    """
    Gather operation for DC decomposition.

    Gather is a linear operation - pos and neg are gathered identically.

    Args:
        dim: Dimension to gather along
    """
    _dc_is_gather = True

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor, index: Tensor) -> Tensor:
        return torch.gather(x, self.dim, index)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class DCMean(nn.Module):
    """
    Mean reduction for DC decomposition.

    Mean is a linear operation: mean(pos - neg) = mean(pos) - mean(neg)

    Args:
        dim: Dimension(s) to reduce
        keepdim: Whether to keep reduced dimensions
    """
    _dc_is_mean = True

    def __init__(self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: Tensor) -> Tensor:
        if self.dim is None:
            return x.mean()
        return x.mean(dim=self.dim, keepdim=self.keepdim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}, keepdim={self.keepdim}'


class DCSum(nn.Module):
    """
    Sum reduction for DC decomposition.

    Sum is a linear operation: sum(pos - neg) = sum(pos) - sum(neg)

    Args:
        dim: Dimension(s) to reduce
        keepdim: Whether to keep reduced dimensions
    """
    _dc_is_sum = True

    def __init__(self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: Tensor) -> Tensor:
        if self.dim is None:
            return x.sum()
        return x.sum(dim=self.dim, keepdim=self.keepdim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}, keepdim={self.keepdim}'
