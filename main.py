import math
import numpy as np
from numpy import linalg
def cylinder_area(r: float,h: float) -> float:
    """Obliczenie pola powierzchni walca.
    Szczegółowy opis w zadaniu 1.

    Parameters:
    r (float): promień podstawy walca
    h (float): wysokosć walca

    Returns:
    float: pole powierzchni walca
    """
    if r > 0 and h > 0:
        return 2*math.pi*r**2 + 2*math.pi*r*h
    return np.NaN

def fib(n:int):
    """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego. 
    Szczegółowy opis w zadaniu 3.
    
    Parameters:
    n (int): liczba określająca ilość wyrazów ciągu do obliczenia 
    
    Returns:
    np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.
    """
    if n <= 0 or type(n) is not int:
        return None
    elif n == 1:
        return np.array([1])
    elif n == 2:
        return np.array([1, 1])
    else:
        fib_seq = np.array([1, 1])
        for i in range(2, n):
            fib_seq = np.append(fib_seq, (fib_seq[i - 1] + fib_seq[i - 2]))
        return fib_seq.reshape(1, -1)

def matrix_calculations(a:float):
    """Funkcja zwraca wartości obliczeń na macierzy stworzonej 
    na podstawie parametru a.  
    Szczegółowy opis w zadaniu 4.
    
    Parameters:
    a (float): wartość liczbowa 
    
    Returns:
    touple: krotka zawierająca wyniki obliczeń 
    (Minv, Mt, Mdet) - opis parametrów w zadaniu 4.
    """
    M = np.array([[a, 1, -a], [0, 1, 1], [-a, a, 1]])

    Mdet = linalg.det(M)

    Minv = np.NaN
    if Mdet != 0:
        Minv = linalg.inv(M)

    Mt = M.T

    return Minv, Mt, Mdet


def custom_matrix(m:int, n:int):
    """Funkcja zwraca macierz o wymiarze mxn zgodnie 
    z opisem zadania 7.  
    
    Parameters:
    m (int): ilość wierszy macierzy
    n (int): ilość kolumn macierzy  
    
    Returns:
    np.ndarray: macierz zgodna z opisem z zadania 7.
    """
    if m <= 0 or n <= 0 or (type(n) or type(m)) is not int:
        return None
    mx = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if i > j:
                mx[i][j] = i
            else:
                mx[i][j] = j
    return mx