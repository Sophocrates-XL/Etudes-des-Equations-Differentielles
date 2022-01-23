import numpy as np
from scipy.linalg import inv, expm

class HeatEquation:

    def __init__(self, A: np.ndarray, B: np.ndarray, f: np.ndarray, diffusivity: float):
        self.__A = np.copy(A)
        self.__D = np.diag(np.sum(self.__A, axis = 1))
        self.__L = self.__D - self.__A
        self.__B = np.copy(B)
        self.__diffusivity = diffusivity
        self.__f = np.copy(f)
        self.__steady_state = inv(self.__L + self.__B) @ self.__f / self.__diffusivity

    def getA(self):
        return np.copy(self.__A)

    def getD(self):
        return np.copy(self.__D)

    def getL(self):
        return np.copy(self.__L)

    def getB(self):
        return np.copy(self.__B)

    def getDiffusivity(self):
        return np.copy(self.__diffusivity)

    def getf(self):
        return np.copy(self.__f)

    def getSteadyState(self) -> np.ndarray:
        return np.copy(self.__steady_state)

    def getStateAtTime(self, phi0: np.ndarray, t: float) -> np.ndarray:
        return expm(-self.__diffusivity * t * (self.__L + self.__B)) @ (phi0 - self.__steady_state) + self.__steady_state



