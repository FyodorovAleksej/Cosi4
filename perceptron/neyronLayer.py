from perceptron.neyron import Neyron


class NeyronLayer:
    def __init__(self, __weight_map: list):
        self.__neyrons = [Neyron(__weight_map[i]) for i in range(0, len(__weight_map))]
        self.__size = len(__weight_map)

    def test_shape(self, __test_shape: list) -> list:
        assert len(__test_shape) == self.__size
        iterations = 1000
        cur = 0
        Xi = __test_shape.copy()
        flag = True
        while flag:
            current = Xi.copy()
            for i in range(0, self.__size):
                current[i] = self.__neyrons[i].test_shape(Xi)
            if Xi == current:
                flag = False
            if cur == iterations:
                print("out of operations")
                flag = False
            Xi = current
            cur += 1
        return Xi