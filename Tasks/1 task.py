import numpy as np
from math import inf


class ULD:
    def __init__(self, type):
        self.type = type
        self.epsilon = 1
        self.bit_M = 0
        self.maxI = 0
        self.minI = 0

    def estimate(self):
        while True:
            self.epsilon /= 2
            if self.type(1 + self.epsilon) == self.type(1):
                break
            self.bit_M += 1
        self.epsilon *= self.type(2)
        return self.epsilon, self.bit_M

    def max_index(self):
        i = self.type(1)
        while True:
            i *= 2
            if self.type(i) == inf:
                break
            self.maxI += 1
        return self.maxI

    def min_index(self):
        i = self.type(1)
        while True:
            i /= 2
            if self.type(i) == self.type(0):
                break
            self.minI += 1
        return self.minI

    def compare(self):
        return self.type(1), self.type(1 + self.epsilon / 2), self.type(1 + self.epsilon), self.type(1 + self.epsilon / 2 + self.epsilon)


uld32 = ULD(np.float32)
uld64 = ULD(np.float64)
print(' 32: ULP number - ' + str(uld32.estimate()[0]) + '\n \t the number of digits in the mantissa - '
      + str(uld32.estimate()[1]) + '\n \t max power - ' + str(uld32.max_index())
      + '\n \t min power - ' + str(uld32.min_index())
      + '\n \t compare - ' + str(uld32.compare()))
print('\n 64: ULP number - ' + str(uld64.estimate()[0]) + '\n \t the number of digits in the mantissa - '
      + str(uld64.estimate()[1]) + '\n \t max power - ' + str(uld64.max_index())
      + '\n \t min power - ' + str(uld64.min_index())
      + '\n \t compare - ' + str(uld64.compare()))

