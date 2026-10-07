# Origin: cc (Claude Code, 2026-10-02) - re-implemented from PyTorch's CPU RNG semantics
#         (aten/src/ATen/core/MT19937RNGEngine.h, DistributionsHelper.h, Distributions.cpp,
#         aten/src/ATen/native/Multinomial.cpp). No code copied; behaviour matched and
#         validated against torch by tests/test_PPJAX_rng_cc.py.
# Purpose: bit-exact re-implementation of the torch CPU RNG stream (manual_seed / rand /
#          exponential_ / multinomial / randint / randperm) so the JAX port generates the
#          SAME sequences as the torch original for the same seed, without importing torch.

from __future__ import annotations

from typing import Optional

import numpy as np

_N = 624
_M = 397
_MATRIX_A = np.uint32(0x9908B0DF)
_UPPER_MASK = np.uint32(0x80000000)
_LOWER_MASK = np.uint32(0x7FFFFFFF)


def _normal_fill_16(data: np.ndarray, mean: np.float32, std: np.float32) -> np.ndarray:
    """``normal_fill_16``: Box-Muller over a 16-wide block, pairing slot j with slot j+8."""
    out = np.empty(16, dtype=np.float32)
    u1 = (np.float32(1.0) - data[:8]).astype(np.float32)
    u2 = data[8:16].astype(np.float32)
    radius = np.sqrt((np.float32(-2.0) * np.log(u1, dtype=np.float32)).astype(np.float32),
                     dtype=np.float32)
    theta = (2.0 * np.pi * u2.astype(np.float64)).astype(np.float32)
    out[:8] = (radius * np.cos(theta, dtype=np.float32) * std + mean).astype(np.float32)
    out[8:16] = (radius * np.sin(theta, dtype=np.float32) * std + mean).astype(np.float32)
    return out


class MT19937:
    """PyTorch's ``at::mt19937_engine``: standard MT19937 seeded with ``seed & 0xffffffff``."""

    __slots__ = ("_state", "_next")

    def __init__(self, seed: int = 5489) -> None:
        self.seed(seed)

    def seed(self, seed: int) -> None:
        s = int(seed) & 0xFFFFFFFF
        state = np.empty(_N, dtype=np.uint32)
        state[0] = np.uint32(s)
        prev = s
        for j in range(1, _N):
            prev = (1812433253 * (prev ^ (prev >> 30)) + j) & 0xFFFFFFFF
            state[j] = np.uint32(prev)
        self._state = state
        self._next = _N  # force a twist on the first draw (torch starts with left_ == 1)

    def _twist(self) -> None:
        s = self._state
        y = (s & _UPPER_MASK) | (np.roll(s, -1) & _LOWER_MASK)
        mag = np.where(np.roll(s, -1) & np.uint32(1), _MATRIX_A, np.uint32(0)).astype(np.uint32)
        nxt = np.roll(s, -_M) ^ (y >> np.uint32(1)) ^ mag
        # np.roll gives the right answer only for a simultaneous update; MT19937 updates in
        # place, so redo the tail sequentially where the wrapped reads see new values.
        out = nxt.copy()
        for k in range(_N - _M, _N):
            y_k = np.uint32((s[k] & _UPPER_MASK) | (s[(k + 1) % _N] & _LOWER_MASK))
            if k + 1 == _N:
                y_k = np.uint32((s[k] & _UPPER_MASK) | (out[0] & _LOWER_MASK))
            out[k] = out[(k + _M) % _N] ^ (y_k >> np.uint32(1)) ^ (
                _MATRIX_A if (y_k & np.uint32(1)) else np.uint32(0))
        self._state = out
        self._next = 0

    def random_block(self, n: int) -> np.ndarray:
        """``n`` tempered uint32 draws, in stream order."""
        out = np.empty(n, dtype=np.uint32)
        filled = 0
        while filled < n:
            if self._next >= _N:
                self._twist()
            take = min(_N - self._next, n - filled)
            out[filled:filled + take] = self._state[self._next:self._next + take]
            self._next += take
            filled += take
        y = out
        y = y ^ (y >> np.uint32(11))
        y = y ^ ((y << np.uint32(7)) & np.uint32(0x9D2C5680))
        y = y ^ ((y << np.uint32(15)) & np.uint32(0xEFC60000))
        y = y ^ (y >> np.uint32(18))
        return y

    def random(self) -> int:
        return int(self.random_block(1)[0])


class TorchCPUGenerator:
    """The subset of ``torch.Generator`` (CPU) the design engine consumes.

    ``torch.manual_seed(k)`` seeds the default CPU generator; every draw below consumes the
    same number of uint32 words, in the same order, as the corresponding torch call, so a
    sequence of calls reproduces torch's stream exactly.
    """

    def __init__(self, seed: int = 0) -> None:
        self._engine = MT19937(seed)
        self._normal_cache: Optional[float] = None

    def manual_seed(self, seed: int) -> "TorchCPUGenerator":
        self._engine.seed(seed)
        # CPUGeneratorImpl::set_current_seed also drops the cached normal partner.
        self._normal_cache = None
        return self

    # --- uniform -------------------------------------------------------------
    def uniform_float32(self, n: int) -> np.ndarray:
        """``torch.rand(n, dtype=torch.float32)``: one uint32 per sample, low 24 bits / 2**24."""
        w = self._engine.random_block(n).astype(np.uint64)
        return ((w & np.uint64((1 << 24) - 1)).astype(np.float64) * (2.0 ** -24)).astype(np.float32)

    def uniform_float64(self, n: int) -> np.ndarray:
        """``torch.rand(n, dtype=torch.float64)``: two uint32 per sample (hi<<32 | lo), 53 bits."""
        w = self._engine.random_block(2 * n).astype(np.uint64)
        hi, lo = w[0::2], w[1::2]
        r64 = (hi << np.uint64(32)) | lo
        return (r64 & np.uint64((1 << 53) - 1)).astype(np.float64) * (2.0 ** -53)

    # --- derived distributions ----------------------------------------------
    def exponential_float32(self, n: int, lambd: float = 1.0) -> np.ndarray:
        """``Tensor.exponential_(lambd)`` into a float32 tensor.

        torch's CPU kernel instantiates ``exponential_distribution<double>`` whatever the tensor
        dtype (``aten/src/ATen/native/cpu/DistributionTemplates.h``), so each sample consumes TWO
        uint32 words (the float64 uniform) and is evaluated in double before the cast to float32.
        The CPU transformation is ``-log1p(-u) / lambda`` (NOT ``-log(u)``, which is the CUDA
        path) - measured with scripts/util/PPJAX_probe_torch_exponential_cc.py on torch 2.14.1.
        """
        u = self.uniform_float64(n)
        return (-1.0 / float(lambd) * np.log1p(-u)).astype(np.float32)

    def multinomial1(self, probs: np.ndarray) -> int:
        """``int(torch.multinomial(probs, 1))`` for a 1-D float32 ``probs``.

        ``n_sample == 1`` always takes torch's without-replacement fast path
        (``native/Multinomial.cpp``): draw one Exponential(1) per category into a tensor of the
        input's dtype, then ``argmax(probs / q)`` — the exponential-race / Gumbel trick. It
        consumes exactly ``2 * len(probs)`` uint32 words.
        """
        p = np.asarray(probs, dtype=np.float32).reshape(-1)
        q = self.exponential_float32(p.shape[0], 1.0)
        return int(np.argmax((p / q).astype(np.float32)))

    def normal_float32(self, n: int, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
        """``torch.randn(n, dtype=torch.float32)``.

        torch's CPU kernel has two paths (``DistributionTemplates.h``):
          * n >= 16 and contiguous float -> ``normal_fill``: draw n float32 uniforms, then
            Box-Muller them in blocks of 16 pairing j with j+8; when ``n % 16 != 0`` it draws
            16 MORE uniforms and recomputes the final 16 values. Consumes n (+16) words.
          * otherwise -> the scalar ``normal_distribution<double>`` with a generator-level
            cache of the sine partner (2 float64 uniforms = 4 words per PAIR of samples).
        """
        mean32, std32 = np.float32(mean), np.float32(std)
        if n >= 16:
            data = self.uniform_float32(n).astype(np.float32)
            for i in range(0, n - 15, 16):
                data[i:i + 16] = _normal_fill_16(data[i:i + 16], mean32, std32)
            if n % 16 != 0:
                data[n - 16:] = _normal_fill_16(
                    self.uniform_float32(16).astype(np.float32), mean32, std32)
            return data
        out = np.empty(n, dtype=np.float32)
        for k in range(n):
            if self._normal_cache is not None:
                out[k] = np.float32(self._normal_cache * std + mean)
                self._normal_cache = None
                continue
            u = self.uniform_float64(2)
            r = np.sqrt(-2.0 * np.log1p(-u[1]))
            theta = 2.0 * np.pi * u[0]
            self._normal_cache = float(r * np.sin(theta))
            out[k] = np.float32(r * np.cos(theta) * std + mean)
        return out

    def multinomial_rows(self, probs: np.ndarray) -> np.ndarray:
        """``torch.multinomial(probs2d, 1).squeeze(-1)`` for a 2-D float32 ``probs`` [N, V].

        torch fills ONE ``[N, V]`` exponential tensor row-major before the row-wise argmax, so the
        stream consumption is ``2 * N * V`` words - not N independent 1-D draws.
        """
        p = np.asarray(probs, dtype=np.float32)
        n, v = p.shape
        q = self.exponential_float32(n * v, 1.0).reshape(n, v)
        return np.argmax((p / q).astype(np.float32), axis=-1).astype(np.int64)

    def randint(self, high: int, n: int = 1, low: int = 0) -> np.ndarray:
        """``torch.randint(low, high, (n,))``.

        ``uniform_int_from_to_distribution`` only reaches for ``random64()`` when the range is
        >= 2**32; below that it consumes ONE uint32 per sample and returns ``w % range + low``.
        """
        span = int(high) - int(low)
        if span >= (1 << 32):
            w = self._engine.random_block(2 * n).astype(np.uint64)
            r = ((w[0::2] << np.uint64(32)) | w[1::2])
        else:
            r = self._engine.random_block(n).astype(np.uint64)
        return (r % np.uint64(span)).astype(np.int64) + np.int64(low)

    def randperm(self, n: int) -> np.ndarray:
        """``torch.randperm(n)``: identity, then ``randperm_cpu``'s Fisher-Yates sweep
        ``i = 0..n-2`` with ``z = random() % (n - i)`` (ONE uint32 per step) and a swap of
        ``i`` with ``i + z``."""
        out = np.arange(n, dtype=np.int64)
        if n > 1:
            w = self._engine.random_block(n - 1).astype(np.uint64)
            for i in range(n - 1):
                j = i + int(w[i] % np.uint64(n - i))
                out[i], out[j] = out[j], out[i]
        return out


# A module-level default generator, mirroring torch's global CPU generator so that the ported
# engine can call manual_seed() at exactly the points the original does.
_DEFAULT = TorchCPUGenerator(0)


def default_generator() -> TorchCPUGenerator:
    return _DEFAULT


def manual_seed(seed: int) -> TorchCPUGenerator:
    """``torch.manual_seed`` on the module-level default generator."""
    return _DEFAULT.manual_seed(seed)
