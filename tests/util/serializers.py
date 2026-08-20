import numpy


def primitive(value):
    if type(value).__module__ == numpy.__name__:
        return value.tolist() if isinstance(value, numpy.ndarray) else value.item()

    return None
