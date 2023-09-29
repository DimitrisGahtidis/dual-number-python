import numpy as np

def vectorizer(func, input):
    if hasattr(input, '__iter__'):
        return np.vectorize(func)
    else:
        return func