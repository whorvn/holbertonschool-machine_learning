#!/usr/bin/env python3
def add_arrays(arrays):
    if len(set(matrix_shape(arrays))) != 1:
        return None
    return [sum(i) for i in zip(*arrays)]