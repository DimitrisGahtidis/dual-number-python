from functools import partial, cache
import numpy as np
import math

class DualNumber:
    cacheless = True
    def __init__(self, re = 0, grad = None) -> None:
        self.re = re
        self.d = 0
        if DualNumber.cacheless:
            self.grad = partial(self._grad, grad=grad)
        else:
            self.grad = partial(self._grad, grad=cache(grad)) if grad is not None else partial(self._grad, grad=grad) 
    
    def _grad(self, seed = 1, grad = None):
        self.d += seed
        if grad is None:
            pass
        else:
            grad(seed)
        return 0

    # Arithmetic operations
    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.re + other.re, lambda seed: self.grad(seed) + other.grad(seed))
        elif isinstance(other, np.ndarray):
            return np.vectorize(self.__add__)(other)
        else:
            return DualNumber(self.re + other, lambda seed: self.grad(seed))
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.re - other.re, lambda seed: self.grad(seed) - other.grad(seed))
        elif isinstance(other, np.ndarray):
            return np.vectorize(self.__sub__)(other)
        else:
            return DualNumber(self.re - other, lambda seed: self.grad(seed))

    def __rsub__(self, other):
        if isinstance(other, np.ndarray):
            return np.vectorize(self.__rsub__)(other)
        else:
            return DualNumber(other - self.re, lambda seed: self.grad(-seed))
    
    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.re * other.re, lambda seed: self.grad(seed*other.re) * other.grad(self.re*seed))
        elif isinstance(other, np.ndarray):
            return np.vectorize(self.__mul__)(other)
        else:
            return DualNumber(self.re * other, lambda seed: self.grad(seed*other))
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.re/other.re, lambda seed: self.grad(seed/other.re) + other.grad(-self.re*seed/(other.re*other.re)))
        elif isinstance(other, np.ndarray):
            return np.vectorize(self.__truediv__)(other)
        else:
            return DualNumber(self.re/other, lambda seed: self.grad(seed/other))
    
    def __rtruediv__(self, other):
        if isinstance(other, np.ndarray):
            return np.vectorize(self.__rsub__)(other)
        else:
            return DualNumber(other/self.re, lambda seed: self.grad(-other*seed/(self.re*self.re)))
    
    def __pow__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.re**other.re, lambda seed: self.grad(other.re*self.re**(other.re-1)*seed) + other.grad(self.re**other.re*math.log(self.re)*seed))
        elif isinstance(other, np.ndarray):
            return np.vectorize(self.__pow__)(other)
        else:
            return DualNumber(self.re**other, lambda seed: self.grad(other*self.re**(other-1)*seed))
    
    def __rpow__(self, other):
        if isinstance(other, np.ndarray):
            return np.vectorize(self.__rsub__)(other)
        else:
            return DualNumber(other**self.re, lambda seed: self.grad(other**self.re*math.log(other)*seed))

    def __pos__(self):
        return self
    
    def __neg__(self):
        return DualNumber(-self.re, lambda seed: self.grad(-seed))

    def __abs__(self):
        return abs(self.re)

     # Logical operators
    def __gt__(self, other):
        if isinstance(other, DualNumber):
            return self.re > other.re
        else:
            return self.re > other
    def __lt__(self, other):
        if isinstance(other, DualNumber):
            return self.re < other.re
        else:
            return self.re < other
    def __ge__(self, other):
        if isinstance(other, DualNumber):
            return self.re >= other.re
        else:
            return self.re >= other
    def __le__(self, other):
        if isinstance(other, DualNumber):
            return self.re <= other.re
        else:
            return self.re <= other
    def __ne__(self, other):
        if isinstance(other, DualNumber):
            return self.re != other.re
        else:
            return self.re != other
    
    # String representation
    def __str__(self):
        if self.d >= 0:
            return f"({self.re}+{self.d} eps)"
        else:
            return f"({self.re}-{-self.d} eps)"
    # Console representation
    def __repr__(self):
        if self.d >= 0:
            return f"({self.re}+{self.d} eps)"
        else:
            return f"({self.re}-{-self.d} eps)"