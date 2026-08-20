import numpy


def default(value):
    if type(value).__module__ == numpy.__name__:
        if isinstance(value, numpy.ndarray):
            return value.tolist()
        else:
            return value.item()

    return None
