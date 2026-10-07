# Origin: cc (Claude Code, 2026-10-02) - reads PyTorch's zip serialization format directly.
# Purpose: load a torch .ckpt state_dict into numpy arrays WITHOUT importing torch, and reshape it
#          into the ppjax parameter pytree. Keeps the JAX port free of a torch dependency.

from __future__ import annotations

import io
import pickle
import zipfile
from typing import Any, Dict

import numpy as np

# torch storage class name -> numpy dtype
_STORAGE_DTYPE = {
    "FloatStorage": np.dtype("<f4"),
    "DoubleStorage": np.dtype("<f8"),
    "HalfStorage": np.dtype("<f2"),
    "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"),
    "ShortStorage": np.dtype("<i2"),
    "CharStorage": np.dtype("<i1"),
    "ByteStorage": np.dtype("<u1"),
    "BoolStorage": np.dtype("?"),
    "BFloat16Storage": np.dtype("<u2"),  # raw; converted below
}


class _Storage:
    __slots__ = ("key", "dtype", "numel")

    def __init__(self, key: str, dtype: np.dtype, numel: int) -> None:
        self.key, self.dtype, self.numel = key, dtype, numel


class _Stub:
    """Instance stand-in for a global the pickle references but ppjax never needs."""

    def __init__(self, module, name, args=(), kwargs=None):
        self._mod, self._name = module, name
        self._args, self._kwargs = args, kwargs or {}

    def __setstate__(self, state):
        self._state = state

    def __repr__(self):
        return f"<stub {self._mod}.{self._name}>"


def _stub_class(module: str, name: str):
    """A real ``type`` (so NEWOBJ/REDUCE work) that swallows any construction."""

    def __new__(cls, *a, **k):
        obj = object.__new__(cls)
        _Stub.__init__(obj, module, name, a, k)
        return obj

    return type(name, (_Stub,), {"__new__": __new__, "__module__": module})


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False,
                       backward_hooks=None, metadata=None):
    return ("tensor", storage, int(storage_offset), tuple(size), tuple(stride))


def _rebuild_parameter(data, requires_grad=True, backward_hooks=None):
    return data


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "collections":
            import collections
            return getattr(collections, name)
        if module.startswith("torch") and name.endswith("Storage"):
            return _stub_class(module, name)
        if module == "numpy" or module.startswith("numpy."):
            return super().find_class(module, name)
        if module in ("builtins", "__builtin__") and name in (
                "set", "frozenset", "list", "dict", "tuple", "int", "float", "str", "bool"):
            return super().find_class(module, name)
        return _stub_class(module, name)

    def persistent_load(self, pid):
        # ('storage', <StorageClass>, key, location, numel)
        tag = pid[0]
        if tag != "storage":
            raise NotImplementedError(f"unsupported persistent id {tag!r}")
        storage_cls, key, _location, numel = pid[1], pid[2], pid[3], pid[4]
        name = getattr(storage_cls, "__name__", str(storage_cls))
        dtype = _STORAGE_DTYPE.get(name)
        if dtype is None:
            raise NotImplementedError(f"unsupported storage type {name!r}")
        return _Storage(str(key), dtype, int(numel))


def _materialise(obj: Any, zf: zipfile.ZipFile, prefix: str) -> Any:
    """Walk the unpickled tree, turning ('tensor', ...) placeholders into numpy arrays."""
    if isinstance(obj, tuple) and len(obj) == 5 and obj[0] == "tensor":
        _, storage, offset, size, stride = obj
        raw = zf.read(f"{prefix}data/{storage.key}")
        flat = np.frombuffer(raw, dtype=storage.dtype, count=storage.numel)
        if not size:
            return np.array(flat[offset], dtype=storage.dtype)
        arr = np.lib.stride_tricks.as_strided(
            flat[offset:], shape=size,
            strides=tuple(s * storage.dtype.itemsize for s in stride))
        return np.array(arr)  # copy: the strided view aliases a read-only buffer
    if isinstance(obj, dict):
        return {k: _materialise(v, zf, prefix) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_materialise(v, zf, prefix) for v in obj)
    return obj


def load_torch_checkpoint(path: str) -> Dict[str, Any]:
    """Read a torch ``.ckpt`` / ``.pt`` (zip format) into plain python + numpy, no torch import."""
    with zipfile.ZipFile(path) as zf:
        pkl_name = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        prefix = pkl_name[: -len("data.pkl")]
        obj = _Unpickler(io.BytesIO(zf.read(pkl_name))).load()
        return _materialise(obj, zf, prefix)


def load_state_dict(path: str, key: str = "model") -> Dict[str, np.ndarray]:
    """The flat ``name -> array`` state dict out of a PottsMPNN lightning checkpoint."""
    ckpt = load_torch_checkpoint(path)
    if isinstance(ckpt, dict) and key in ckpt:
        sd = ckpt[key]
    elif isinstance(ckpt, dict) and all(isinstance(v, np.ndarray) for v in ckpt.values()):
        sd = ckpt
    else:
        raise KeyError(f"checkpoint has no {key!r} entry; keys={list(ckpt)[:8]}")
    return {str(k): np.asarray(v) for k, v in sd.items()}
