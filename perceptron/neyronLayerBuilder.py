import random as rnd

from perceptron.hideLayerNeyron import HideLayerNeyron
from perceptron.neyron import Neyron
from perceptron.neyronLayer import NeyronLayer


class NeyronLayerBuilder:
    def __init__(self, __n_size: int, __h_size: int, __m_size: int):
        self.__n_size = __n_size
        self.__h_size = __h_size
        self.__m_size = __m_size
        """Hide matrix NxH"""
        self.__HideMatrix = None
        """Out matrix HxM"""
        self.__OutMatrix = None

        self.__HideRestriction = None
        self.__OutRestriction = None

        self.__hideNeyrons = None
        self.__outNeyrons = None

    def randomInit(self, __start: float, __stop: float):
        self.__HideMatrix = [[rnd.uniform(__start, __stop) for _ in range(0, self.__h_size)] for _ in
                             range(0, self.__n_size)]
        self.__OutMatrix = [[rnd.uniform(__start, __stop) for _ in range(0, self.__m_size)] for _ in
                            range(0, self.__h_size)]

        self.__HideRestriction = [rnd.uniform(__start, __stop) for _ in range(0, self.__h_size)]
        self.__OutRestriction = [rnd.uniform(__start, __stop) for _ in range(0, self.__m_size)]

        self.__hideNeyrons = [HideLayerNeyron(self.__HideMatrix[i], self.__HideRestriction[i]) for i in
                          range(0, self.__h_size)]
        self.__outNeyrons = [Neyron(self.__HideMatrix[i], self.__HideRestriction[i]) for i in range(0, self.__m_size)]

    def teach(self, __shape: list, __out: list):
        hideOut = []
        for hideNeyron in self.__hideNeyrons:
            hideOut.append(hideNeyron.test_shape(__shape))
        outOut = []
        for outNeyron in self.__outNeyrons:
            outOut.append(outNeyron.test_shape(__shape))



                if i == j:
                    self.__WMatrix[i][j] = 0
                else:
                    self.__WMatrix[i][j] += __shape[i] * __shape[j]

    def build(self) -> NeyronLayer:
        return NeyronLayer([self.__WMatrix[i] for i in range(0, len(self.__WMatrix))])

    def print_current_weight_map(self):
        print("---------------------------------------")
        for i in range(0, len(self.__WMatrix)):
            print(self.__WMatrix[i])
        print("---------------------------------------")
