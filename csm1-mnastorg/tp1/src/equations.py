#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# Fonction f(t,y), second membre d'équations différentielles d'ordre 1
# écrite sous la forme d'un problème de Cauchy, y'(t) = f(t,y(t)) avec
# y(0)=y0

a = -1.
b = 1.

def f_affine(t,y):
    """Fonction affine pour y' = ay+b. Les coefficients a et b sont des
    variables globales du module.

    """
    return a*y+b

def sol_affine(t,y0):
    """Pour une fonction affine, on connait la solution exacte. C'est
    y(t0+s) = y0*exp(a*s) - b*(1-exp(a*s))/a, soit y(t) =
    y0*exp(a*(t-t0)) - b*(1-exp(a*(t-t0)))/a

    """
    t0 = t[0]
    return y0*np.exp(a*(t-t0)) - b * (1.-np.exp(a*(t-t0)))/a

def f_poly(t,y):
    """Fonction polynomiale pour y' = a*y^2 +b"""

    return 1. - y**2

def sol_poly1(t):
    """Solution exacte pour y0 = 0"""
    
    return np.tanh(t)

def sol_poly2(t):
    """Solution exacte pour y0=2"""

    return (3*np.exp(2*t) + 1)/(3*np.exp(2*t) - 1)
