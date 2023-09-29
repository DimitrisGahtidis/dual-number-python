import numpy as np
from scipy.special import erfc

from .dual_number import DualNumber
from .utils.vectorization import vectorizer
def _sin(z):
    if isinstance(z, DualNumber):
        return DualNumber(np.sin(z.re), lambda seed: z.grad(np.cos(z.re)*seed))
    else:
        return np.sin(z)
def _cos(z):
    if isinstance(z, DualNumber):
        return DualNumber(np.cos(z.re), lambda seed: z.grad(-np.sin(z.re)*seed))
    else:
        return np.cos(z)
def _tan(z):
    if isinstance(z, DualNumber):
        return DualNumber(np.tan(z.re), lambda seed: z.grad(seed/(np.cos(z.re)**2)))
    else:
        return np.tan(z)
def _exp(z):
    if isinstance(z, DualNumber):
        return DualNumber(np.exp(z.re), lambda seed: z.grad(np.exp(z.re)*seed))
    else:
        return np.exp(z)
def _log(z):
    if isinstance(z, DualNumber):
        return DualNumber(np.log(z.re), lambda seed: z.grad(seed/(z.re)))
    else:
        return np.log(z)
def _power(z, n):
    if isinstance(z, DualNumber):
        return DualNumber(np.power(z.re,n), lambda seed: z.grad(n*np.power(z.re, n-1)*seed))
    else:
        return np.power(z, n)
def _sqrt(z):
    if isinstance(z, DualNumber):
        return DualNumber(np.sqrt(z.re), lambda seed: z.grad((0.5)*np.sqrt(z.re)*seed))
    else:
        return np.sqrt(z)
def _arccos(z):
    if isinstance(z, DualNumber):
        return DualNumber(np.arccos(z.re), lambda seed: z.grad(-seed/np.sqrt(1-z.re**2)))
    else:
        return np.arccos(z)
def _arcsin(z):
    if isinstance(z, DualNumber):
        return DualNumber(np.arcsin(z.re), lambda seed: z.grad(seed/np.sqrt(1-z.re**2)))
    else:
        return np.arcsin(z)
def _arctan(z):
    if isinstance(z, DualNumber):
        return DualNumber(np.arctan(z.re), lambda seed: z.grad(seed/(1+z.re**2)))
    else:
        return np.arctan(z)
def _sinh(z):
    if isinstance(z, DualNumber):
        return DualNumber(np.sinh(z.re), lambda seed: z.grad(np.cosh(z.re)*seed))
    else:
        return np.sinh(z)
def _cosh(z):
    if isinstance(z, DualNumber):
        return DualNumber(np.cosh(z.re), lambda seed: z.grad(np.sinh(z.re)*seed))
    else:
        return np.cosh(z)
def _tanh(z):
    if isinstance(z, DualNumber):
        return DualNumber(np.tanh(z.re), lambda seed: z.grad((1 - np.tanh(z.re)**2)*seed))
    else:
        return np.tanh(z)
def _arcsinh(z):
    if isinstance(z, DualNumber):
        return DualNumber(np.arcsinh(z.re), lambda seed: z.grad(seed/(1-z.re**2)))
    else:
        return np.arcsinh(z)
def _arccosh(z):
    if isinstance(z, DualNumber):
        return DualNumber(np.arccosh(z.re), lambda seed: z.grad(seed/np.sqrt(z.re**2-1)))
    else:
        return np.arccosh(z)
def _erf(z):
    if isinstance(z, DualNumber):
        return DualNumber(1-erfc(z.re), lambda seed: z.grad((2/np.sqrt(np.pi))*np.exp(-(z.re**2))*seed))
    else:
        return 1-erfc(z)

def sin(z):
    return vectorizer(_sin, z)(z)
def cos(z):
    return vectorizer(_cos, z)(z)
def tan(z):
    return vectorizer(_tan, z)(z)
def exp(z):
    return vectorizer(_exp, z)(z)
def log(z):
    return vectorizer(_log, z)(z)
def power(z, n):
    return vectorizer(_power, z)(z, n)
def sqrt(z):
    return vectorizer(_sqrt, z)(z)
def arccos(z):
    return vectorizer(_arccos, z)(z)
def arcsin(z):
    return vectorizer(_arcsin, z)(z)
def arctan(z):
    return vectorizer(_arctan, z)(z)
def sinh(z):
    return vectorizer(_sinh, z)(z)
def cosh(z):
    return vectorizer(_cosh, z)(z)
def tanh(z):
    return vectorizer(_tanh, z)(z)
def arcsinh(z):
    return vectorizer(_arcsinh, z)(z)
def arccosh(z):
    return vectorizer(_arccosh, z)(z)
def erf(z):
    return vectorizer(_erf, z)(z)