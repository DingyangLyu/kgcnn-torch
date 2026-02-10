"""Polynomial and special function layers for graph neural networks.

Provides layers that compute spherical Bessel functions, Legendre polynomials,
spherical harmonics, and associated Legendre polynomials. These are used in
angular and radial basis expansions for geometric GNNs (e.g., DimeNet, SphereNet).

All computations use explicit closed-form expressions with precomputed prefactors,
avoiding recursion at runtime for efficiency.
"""
import math
import numpy as np
import scipy as sp
import scipy.special
import torch
import torch.nn as nn
from scipy.optimize import brentq


def spherical_bessel_jn(r, n):
    r"""Compute spherical Bessel function :math:`j_n(r)` via scipy.

    Args:
        r (np.ndarray): Argument.
        n (np.ndarray, int): Order.

    Returns:
        np.ndarray: Values of the spherical Bessel function.
    """
    return np.sqrt(np.pi / (2 * r)) * sp.special.jv(n + 0.5, r)


def spherical_bessel_jn_zeros(n, k):
    r"""Compute the first k zeros of spherical Bessel functions :math:`j_n(r)` up to order n.

    Taken from the DimeNet implementation at https://github.com/klicperajo/dimenet.

    Args:
        n: Maximum order (excluded).
        k: Number of zero crossings.

    Returns:
        np.ndarray: Zero crossings of shape (n, k).
    """
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(spherical_bessel_jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]
    return zerosj


def spherical_bessel_jn_normalization_prefactor(n, k):
    r"""Compute normalization prefactor for spherical Bessel functions.

    Taken from the DimeNet implementation at https://github.com/klicperajo/dimenet.

    Args:
        n: Maximum order (excluded).
        k: Maximum frequency (excluded).

    Returns:
        np.ndarray: Normalization prefactors of shape (n, k).
    """
    zeros = spherical_bessel_jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * spherical_bessel_jn(zeros[order, i], order + 1) ** 2]
        normalizer_tmp = 1 / np.array(normalizer_tmp) ** 0.5
        normalizer += [normalizer_tmp]
    return np.array(normalizer)


def torch_spherical_bessel_jn_explicit(x, n=0):
    r"""Compute spherical Bessel function :math:`j_n(x)` explicitly using closed form.

    Uses the explicit expression from https://dlmf.nist.gov/10.49.

    Args:
        x (Tensor): Input values.
        n (int): Non-negative integer order.

    Returns:
        Tensor: Spherical Bessel function values.
    """
    sin_x = torch.sin(x - n * math.pi / 2)
    cos_x = torch.cos(x - n * math.pi / 2)
    sum_sin = torch.zeros_like(x)
    sum_cos = torch.zeros_like(x)
    for k in range(int(np.floor(n / 2)) + 1):
        if 2 * k < n + 1:
            prefactor_sin = float(
                sp.special.factorial(n + 2 * k) / np.power(2, 2 * k) /
                sp.special.factorial(2 * k) / sp.special.factorial(n - 2 * k) *
                np.power(-1, k))
            sum_sin = sum_sin + prefactor_sin * torch.pow(x, -(2 * k + 1))
    for k in range(int(np.floor((n - 1) / 2)) + 1):
        if 2 * k + 1 < n + 1:
            prefactor_cos = float(
                sp.special.factorial(n + 2 * k + 1) / np.power(2, 2 * k + 1) /
                sp.special.factorial(2 * k + 1) / sp.special.factorial(n - 2 * k - 1) *
                np.power(-1, k))
            sum_cos = sum_cos + prefactor_cos * torch.pow(x, -(2 * k + 2))
    return sum_sin * sin_x + sum_cos * cos_x


def torch_spherical_bessel_jn(x, n=0):
    r"""Compute spherical Bessel function :math:`j_n(x)` via recursion.

    Uses the recursive rule from https://dlmf.nist.gov/10.51:
    :math:`j_{n+1}(z)=((2n+1)/z)j_{n}(z)-j_{n-1}(z)`

    Args:
        x (Tensor): Input values.
        n (int): Non-negative integer order.

    Returns:
        Tensor: Spherical Bessel function values.
    """
    if n < 0:
        raise ValueError("Order parameter must be >= 0.")
    x = x.clamp(min=1e-8)  # Avoid division by zero at x→0
    if n == 0:
        return torch.sin(x) / x
    elif n == 1:
        return torch.sin(x) / x.pow(2) - torch.cos(x) / x
    else:
        j_n = torch.sin(x) / x
        j_nn = torch.sin(x) / x.pow(2) - torch.cos(x) / x
        for i in range(1, n):
            temp = j_nn
            j_nn = (2 * i + 1) / x * j_nn - j_n
            j_n = temp
        return j_nn


def torch_legendre_polynomial_pn(x, n=0):
    r"""Compute non-associated Legendre polynomial :math:`P_n(x)` via explicit formula.

    :math:`P_n(x)=\sum_{k=0}^{\lfloor n/2\rfloor} (-1)^k
    \frac{(2n - 2k)!}{(n-k)!(n-2k)!k!2^n} x^{n-2k}`

    Args:
        x (Tensor): Input values.
        n (int): Non-negative integer order.

    Returns:
        Tensor: Legendre polynomial values.
    """
    out_sum = torch.zeros_like(x)
    prefactors = [
        float((-1) ** k * sp.special.factorial(2 * n - 2 * k) /
              sp.special.factorial(n - k) / sp.special.factorial(n - 2 * k) /
              sp.special.factorial(k) / 2 ** n)
        for k in range(0, int(np.floor(n / 2)) + 1)
    ]
    powers = [float(n - 2 * k) for k in range(0, int(np.floor(n / 2)) + 1)]
    for i in range(len(powers)):
        out_sum = out_sum + prefactors[i] * torch.pow(x, powers[i])
    return out_sum


def torch_spherical_harmonics_yl(theta, l=0):
    r"""Compute spherical harmonics :math:`Y_l(\cos\theta)` for m=0.

    Uses the simplified formula from https://en.wikipedia.org/wiki/Spherical_harmonics
    with m=0.

    Args:
        theta (Tensor): Input angle values.
        l (int): Non-negative integer order.

    Returns:
        Tensor: Spherical harmonics values.
    """
    x = torch.cos(theta)
    out_sum = torch.zeros_like(x)
    prefactors = [
        float((-1) ** k * sp.special.factorial(2 * l - 2 * k) /
              sp.special.factorial(l - k) / sp.special.factorial(l - 2 * k) /
              sp.special.factorial(k) / 2 ** l)
        for k in range(0, int(np.floor(l / 2)) + 1)
    ]
    powers = [float(l - 2 * k) for k in range(0, int(np.floor(l / 2)) + 1)]
    for i in range(len(powers)):
        out_sum = out_sum + prefactors[i] * torch.pow(x, powers[i])
    out_sum = out_sum * float(np.sqrt((2 * l + 1) / 4 / np.pi))
    return out_sum


def torch_associated_legendre_polynomial(x, l=0, m=0):
    r"""Compute associated Legendre polynomial :math:`P_l^m(x)` via explicit formula.

    :math:`P_{l}^{m}(x)=(-1)^{m} 2^{l} (1-x^{2})^{m/2}
    \sum_{k=m}^{l}\frac{k!}{(k-m)!} x^{k-m} \binom{l}{k}\binom{(l+k-1)/2}{l}`

    Args:
        x (Tensor): Input values.
        l (int): Non-negative integer for l.
        m (int): Integer for m with |m| <= l.

    Returns:
        Tensor: Associated Legendre polynomial values.
    """
    if np.abs(m) > l:
        raise ValueError("Legendre polynomial must have -l <= m <= l")
    if l < 0:
        raise ValueError("Legendre polynomial must have l >= 0")
    if m < 0:
        m = -m
        neg_m = float(np.power(-1, m) * sp.special.factorial(l - m) / sp.special.factorial(l + m))
    else:
        neg_m = 1.0

    x_prefactor = torch.pow(1 - x.pow(2), m / 2) * float(np.power(-1, m) * np.power(2, l))
    sum_out = torch.zeros_like(x)
    for k in range(m, l + 1):
        sum_out = sum_out + torch.pow(x, k - m) * float(
            sp.special.factorial(k) / sp.special.factorial(k - m) *
            sp.special.binom(l, k) * sp.special.binom((l + k - 1) / 2, l))
    return sum_out * x_prefactor * neg_m


class SphericalBesselJnExplicit(nn.Module):
    r"""Compute spherical Bessel function :math:`j_n(x)` for fixed order n.

    Uses the explicit expression from https://dlmf.nist.gov/10.49 with precomputed
    prefactors. Supports a fused mode where all terms are computed in a single
    batched operation.

    :math:`\mathsf{j}_{n}(z)=\sin(z-\frac{n\pi}{2})
    \sum_{k} a_{2k} z^{-(2k+1)} + \cos(z-\frac{n\pi}{2})
    \sum_{k} a_{2k+1} z^{-(2k+2)}`
    """

    def __init__(self, n: int = 0, fused: bool = False):
        """Initialize with fixed order n.

        Args:
            n: Non-negative integer for the Bessel order.
            fused: Whether to compute polynomial in a fused tensor representation.
        """
        super().__init__()
        self.n = n
        self.fused = fused

        pre_factor_sin = []
        powers_sin = []
        pre_factor_cos = []
        powers_cos = []

        for k in range(int(np.floor(n / 2)) + 1):
            if 2 * k < n + 1:
                fac_sin = float(
                    sp.special.factorial(n + 2 * k) / np.power(2, 2 * k) /
                    sp.special.factorial(2 * k) / sp.special.factorial(n - 2 * k) *
                    np.power(-1, k))
                pow_sin = -(2 * k + 1)
                pre_factor_sin.append(fac_sin)
                powers_sin.append(pow_sin)

        for k in range(int(np.floor((n - 1) / 2)) + 1):
            if 2 * k + 1 < n + 1:
                fac_cos = float(
                    sp.special.factorial(n + 2 * k + 1) / np.power(2, 2 * k + 1) /
                    sp.special.factorial(2 * k + 1) / sp.special.factorial(n - 2 * k - 1) *
                    np.power(-1, k))
                pow_cos = -(2 * k + 2)
                pre_factor_cos.append(fac_cos)
                powers_cos.append(pow_cos)

        if self.fused:
            self.register_buffer("_pre_factor_sin", torch.tensor(pre_factor_sin))
            self.register_buffer("_powers_sin", torch.tensor(powers_sin, dtype=torch.float32))
            self.register_buffer("_pre_factor_cos", torch.tensor(pre_factor_cos))
            self.register_buffer("_powers_cos", torch.tensor(powers_cos, dtype=torch.float32))
        else:
            self._pre_factor_sin = pre_factor_sin
            self._powers_sin = powers_sin
            self._pre_factor_cos = pre_factor_cos
            self._powers_cos = powers_cos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spherical Bessel function.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape with :math:`j_n(x)` values.
        """
        n = self.n
        sin_x = torch.sin(x - n * math.pi / 2)
        cos_x = torch.cos(x - n * math.pi / 2)

        if not self.fused:
            sum_sin = torch.zeros_like(x)
            sum_cos = torch.zeros_like(x)
            for a, r in zip(self._pre_factor_sin, self._powers_sin):
                sum_sin = sum_sin + a * torch.pow(x, r)
            for b, s in zip(self._pre_factor_cos, self._powers_cos):
                sum_cos = sum_cos + b * torch.pow(x, s)
        else:
            sum_sin = torch.sum(
                self._pre_factor_sin * torch.pow(x.unsqueeze(-1), self._powers_sin),
                dim=-1)
            sum_cos = torch.sum(
                self._pre_factor_cos * torch.pow(x.unsqueeze(-1), self._powers_cos),
                dim=-1)

        return sum_sin * sin_x + sum_cos * cos_x


class LegendrePolynomialPn(nn.Module):
    r"""Compute non-associated Legendre polynomial :math:`P_n(x)` for fixed order n.

    Uses the explicit formula with precomputed prefactors:

    :math:`P_n(x)=\sum_{k=0}^{\lfloor n/2\rfloor} (-1)^k
    \frac{(2n-2k)!}{(n-k)!(n-2k)!k!2^n} x^{n-2k}`
    """

    def __init__(self, n: int = 0, fused: bool = False):
        """Initialize with fixed order n.

        Args:
            n: Non-negative integer for the polynomial order.
            fused: Whether to compute polynomial in a fused tensor representation.
        """
        super().__init__()
        self.n = n
        self.fused = fused

        pre_factors = [
            float((-1) ** k * sp.special.factorial(2 * n - 2 * k) /
                  sp.special.factorial(n - k) / sp.special.factorial(n - 2 * k) /
                  sp.special.factorial(k) / 2 ** n)
            for k in range(0, int(np.floor(n / 2)) + 1)
        ]
        powers = [float(n - 2 * k) for k in range(0, int(np.floor(n / 2)) + 1)]

        if self.fused:
            self.register_buffer("_pre_factors", torch.tensor(pre_factors))
            self.register_buffer("_powers", torch.tensor(powers, dtype=torch.float32))
        else:
            self._pre_factors = pre_factors
            self._powers = powers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Legendre polynomial.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape with :math:`P_n(x)` values.
        """
        if not self.fused:
            out_sum = torch.zeros_like(x)
            for a, r in zip(self._pre_factors, self._powers):
                out_sum = out_sum + a * torch.pow(x, r)
        else:
            out_sum = torch.sum(
                self._pre_factors * torch.pow(x.unsqueeze(-1), self._powers),
                dim=-1)
        return out_sum


class SphericalHarmonicsYl(nn.Module):
    r"""Compute spherical harmonics :math:`Y_l(\cos\theta)` for m=0 and fixed order l.

    Uses the simplified formula from https://en.wikipedia.org/wiki/Spherical_harmonics with
    m=0, where the associated Legendre polynomial reduces to the ordinary Legendre polynomial:

    :math:`Y_l^0(\theta, \phi) = \sqrt{\frac{2l+1}{4\pi}} P_l(\cos\theta)`
    """

    def __init__(self, l: int = 0, fused: bool = False):
        """Initialize with fixed order l.

        Args:
            l: Non-negative integer for the spherical harmonics order.
            fused: Whether to compute polynomial in a fused tensor representation.
        """
        super().__init__()
        self.l = l
        self.fused = fused

        pre_factors = [
            float((-1) ** k * sp.special.factorial(2 * l - 2 * k) /
                  sp.special.factorial(l - k) / sp.special.factorial(l - 2 * k) /
                  sp.special.factorial(k) / 2 ** l)
            for k in range(0, int(np.floor(l / 2)) + 1)
        ]
        powers = [float(l - 2 * k) for k in range(0, int(np.floor(l / 2)) + 1)]
        self._scale = float(np.sqrt((2 * l + 1) / 4 / np.pi))

        if self.fused:
            self.register_buffer("_pre_factors", torch.tensor(pre_factors))
            self.register_buffer("_powers", torch.tensor(powers, dtype=torch.float32))
        else:
            self._pre_factors = pre_factors
            self._powers = powers

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        r"""Compute spherical harmonics.

        Args:
            theta: Input angle tensor of arbitrary shape.

        Returns:
            Tensor of same shape with :math:`Y_l(\cos\theta)` values.
        """
        x = torch.cos(theta)
        if not self.fused:
            out_sum = torch.zeros_like(x)
            for a, r in zip(self._pre_factors, self._powers):
                out_sum = out_sum + a * torch.pow(x, r)
        else:
            out_sum = torch.sum(
                self._pre_factors * torch.pow(x.unsqueeze(-1), self._powers),
                dim=-1)
        return out_sum * self._scale


class AssociatedLegendrePolynomialPlm(nn.Module):
    r"""Compute associated Legendre polynomial :math:`P_l^m(x)` for fixed l and m.

    Uses the closed-form expression from
    https://en.wikipedia.org/wiki/Associated_Legendre_polynomials:

    :math:`P_{l}^{m}(x)=(-1)^{m} 2^{l} (1-x^{2})^{m/2}
    \sum_{k=m}^{l}\frac{k!}{(k-m)!} x^{k-m} \binom{l}{k}\binom{(l+k-1)/2}{l}`
    """

    def __init__(self, l: int = 0, m: int = 0, fused: bool = False):
        """Initialize with fixed l, m.

        Args:
            l: Non-negative integer for l.
            m: Integer for m with |m| <= l.
            fused: Whether to compute polynomial in a fused tensor representation.
        """
        super().__init__()
        self.l = l
        self.m = m
        self.fused = fused

        if np.abs(m) > l:
            raise ValueError("Legendre polynomial must have -l <= m <= l")
        if l < 0:
            raise ValueError("Legendre polynomial must have l >= 0")

        m_abs = abs(m)
        if m < 0:
            neg_m = float(np.power(-1, m_abs) * sp.special.factorial(l - m_abs) /
                          sp.special.factorial(l + m_abs))
        else:
            neg_m = 1.0

        self._m_abs = m_abs
        self._neg_m = neg_m
        self._x_pre_factor = float(np.power(-1, m_abs) * np.power(2, l))

        pre_factors = []
        powers = []
        for k in range(m_abs, l + 1):
            powers.append(k - m_abs)
            fac = float(
                sp.special.factorial(k) / sp.special.factorial(k - m_abs) *
                sp.special.binom(l, k) * sp.special.binom((l + k - 1) / 2, l))
            pre_factors.append(fac)

        if self.fused:
            self.register_buffer("_pre_factors", torch.tensor(pre_factors))
            self.register_buffer("_powers", torch.tensor(powers, dtype=torch.float32))
        else:
            self._pre_factors = pre_factors
            self._powers = powers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute associated Legendre polynomial.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape with :math:`P_l^m(x)` values.
        """
        m = self._m_abs
        x_pre_factor = torch.pow(1 - x.pow(2), m / 2) * self._x_pre_factor

        if not self.fused:
            sum_out = torch.zeros_like(x)
            for a, r in zip(self._pre_factors, self._powers):
                sum_out = sum_out + a * torch.pow(x, r)
        else:
            sum_out = torch.sum(
                self._pre_factors * torch.pow(x.unsqueeze(-1), self._powers),
                dim=-1)

        return sum_out * x_pre_factor * self._neg_m
