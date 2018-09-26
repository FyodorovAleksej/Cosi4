import random as rnd

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

        hide_t = list(zip(*self.__HideMatrix))
        self.__hideNeyrons = [Neyron(list(hide_t[i]), self.__HideRestriction[i]) for i in
                              range(0, self.__h_size)]
        out_t = list(zip(*self.__OutMatrix))
        self.__outNeyrons = [Neyron(list(out_t[i]), self.__OutRestriction[i]) for i in range(0, self.__m_size)]

    def teach(self, __shape: list, __out: list, __alpha: float, __beta: float, __D: float):
        flagStart = True
        MAX_DK = 0
        countTeach = 0
        while (MAX_DK >= __D) or flagStart:
            MAX_DK = 0
            flagStart = False
            hideOut = []
            for hideNeyron in self.__hideNeyrons:
                hideOut.append(hideNeyron.test_shape(__shape))
            outOut = []
            for outNeyron in self.__outNeyrons:
                outOut.append(outNeyron.test_shape(hideOut))

            # ---------------OUT-Layer---------------------
            for j in range(0, len(self.__OutMatrix)):
                for k in range(0, len(self.__OutMatrix[j])):
                    yk = outOut[k]
                    dk = __out[k] - yk
                    MAX_DK = max(MAX_DK, abs(dk))
                    temp = __alpha * yk * (1 - yk) * dk
                    self.__OutMatrix[j][k] += temp * hideOut[j]
                    if j == 0:
                        self.__OutRestriction[k] += temp

            # ---------------HIDE-Layer---------------
            for i in range(0, len(self.__HideMatrix)):
                for j in range(0, len(self.__HideMatrix[i])):
                    ej = 0
                    for k in range(0, len(outOut)):
                        yk = outOut[k]
                        dk = __out[k] - yk
                        MAX_DK = max(MAX_DK, abs(dk))
                        proizv = yk * (1 - yk)
                        ej += dk * proizv * self.__OutMatrix[j][k]
                    gj = hideOut[j]
                    temp = __beta * gj * (1 - gj) * ej
                    self.__HideMatrix[i][j] += temp * __shape[i]
                    if i == 0:
                        self.__HideRestriction[j] += temp

            out_t = list(zip(*self.__OutMatrix))
            for i in range(0, self.__m_size):
                self.__outNeyrons[i].refresh(list(out_t[i]), self.__OutRestriction[i])
            hide_t = list(zip(*self.__HideMatrix))
            for i in range(0, self.__h_size):
                self.__hideNeyrons[i].refresh(list(hide_t[i]), self.__HideRestriction[i])
            countTeach += 1
        print("TEACH WAS COMPLETE = " + str(countTeach))

    def build(self) -> NeyronLayer:
        return NeyronLayer(self.__hideNeyrons, self.__outNeyrons)
