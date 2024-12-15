#!/usr/bin/env python3
"doc"


def poly_derivative(poly):
    "doc"
    coef = []
    for i in range(len(poly)):
        a = poly[i]*i
        coef.append(a)
    return coef[1:]
