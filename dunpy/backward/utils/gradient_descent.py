from ..dual_number import DualNumber
def update(container, lr: float):
    if hasattr(container, '__iter__'):
        for dual_number in container:
            update(dual_number, lr)
    elif isinstance(container, DualNumber):
        container.re -= lr*container.d

def zero_grad(container):
    if hasattr(container, '__iter__'):
        for dual_number in container:
            zero_grad(dual_number)
    elif isinstance(container, DualNumber):
        container.d = 0