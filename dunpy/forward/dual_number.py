import numpy as np
import math

class DualNumber:
    cacheless = True
    def __init__(self, re = 0, d = 0) -> None:
        self.re = re
        self.d = d

    # Arithmetic operations
    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.re + other.re, self.d + other.d)
        elif isinstance(other, np.ndarray):
            return np.vectorize(self.__add__)(other)
        else:
            return DualNumber(self.re + other, self.d)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.re - other.re, self.d - other.d)
        elif isinstance(other, np.ndarray):
            return np.vectorize(self.__sub__)(other)
        else:
            return DualNumber(self.re - other, self.d)

    def __rsub__(self, other):
        if isinstance(other, np.ndarray):
            return np.vectorize(self.__rsub__)(other)
        else:
            return DualNumber(other - self.re, -self.d)
    
    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.re * other.re, self.d*other.re * self.re*other.d)
        elif isinstance(other, np.ndarray):
            return np.vectorize(self.__mul__)(other)
        else:
            return DualNumber(self.re * other, self.d*other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.re/other.re, self.d/other.re  - self.re*other.d/(other.re*other.re))
        elif isinstance(other, np.ndarray):
            return np.vectorize(self.__truediv__)(other)
        else:
            return DualNumber(self.re/other, self.d/other)
    
    def __rtruediv__(self, other):
        if isinstance(other, np.ndarray):
            return np.vectorize(self.__rsub__)(other)
        else:
            return DualNumber(other/self.re, -other*self.d/(self.re*self.re))
    
    def __pow__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.re**other.re, other.re*self.re**(other.re-1) * self.d + self.re**other.re*math.log(self.re)*other.d)
        elif isinstance(other, np.ndarray):
            return np.vectorize(self.__pow__)(other)
        else:
            return DualNumber(self.re**other, other*self.re**(other-1)*self.d)
    
    def __rpow__(self, other):
        if isinstance(other, np.ndarray):
            return np.vectorize(self.__rsub__)(other)
        else:
            return DualNumber(other**self.re, other**self.re*math.log(other)*self.d)

    def __pos__(self):
        return self
    
    def __neg__(self):
        return DualNumber(-self.re, -self.d)

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