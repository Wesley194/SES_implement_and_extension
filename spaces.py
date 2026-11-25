"""Wavelet-based low-frequency and multi-resolution search spaces."""
import numpy as np
import pywt


class LowFreqSpace:
    """Splits a noise tensor (C, H, W) into searchable low-freq u and frozen high-freq."""

    def __init__(self, shape, level=4, wavelet="haar", rng=None):
        C, H, W = shape
        if level > 0 and (H % 2**level or W % 2**level):
            raise ValueError(f"H and W must be divisible by 2**level={2**level}")
        rng = rng or np.random.default_rng()
        self.wavelet, self.level = wavelet, level
        x_ref = rng.standard_normal(shape)
        coeffs = pywt.wavedec2(
            x_ref, wavelet, mode="periodization", level=level, axes=(-2, -1)
        )
        self.u_ref = coeffs[0]        # c_LL^(J)
        self.fixed_high = coeffs[1:]  # c_H^fixed (frozen)
        self.u_shape = coeffs[0].shape
        self.dim = coeffs[0].size

    def to_noise(self, u):
        """x0(u) = IDWT(u (+) c_H^fixed)"""
        u = np.asarray(u).reshape(self.u_shape)
        return pywt.waverec2(
            [u] + self.fixed_high, self.wavelet, mode="periodization", axes=(-2, -1)
        )


class WaveletSpace:
    """Multi-resolution parameterisation allowing coarse-to-fine sub-band unfreezing."""

    def __init__(self, shape, max_level=6, wavelet="haar", rng=None):
        C, H, W = shape
        if H % 2**max_level or W % 2**max_level:
            raise ValueError(f"H and W must be divisible by 2**{max_level}")
        rng = rng or np.random.default_rng()
        self.wavelet, self.J0 = wavelet, max_level
        c = pywt.wavedec2(
            rng.standard_normal(shape),
            wavelet,
            mode="periodization",
            level=max_level,
            axes=(-2, -1),
        )
        self.ll = c[0]
        self.details = c[1:]  # coarse -> fine: details[k] = (cH, cV, cD) at level J0-k

    def level_to_k(self, level):
        return self.J0 - level

    def dim(self, k):
        return self.ll.size + sum(a.size for d in self.details[:k] for a in d)

    def ref_vec(self, k):
        parts = [self.ll.ravel()] + [a.ravel() for d in self.details[:k] for a in d]
        return np.concatenate(parts)

    def to_noise(self, vec, k):
        i = self.ll.size
        ll = vec[:i].reshape(self.ll.shape)
        det = []
        for j, d in enumerate(self.details):
            if j < k:
                tri = []
                for a in d:
                    tri.append(vec[i : i + a.size].reshape(a.shape))
                    i += a.size
                det.append(tuple(tri))
            else:
                det.append(d)  # still frozen at reference value
        return pywt.waverec2(
            [ll] + det, self.wavelet, mode="periodization", axes=(-2, -1)
        )