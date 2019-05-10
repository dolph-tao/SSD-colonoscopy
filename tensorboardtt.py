from numpy import random

# with open('11.txt', "r+") as f:
#     read_data = f.read()
#     f.truncate()
import numpy
for i in range(0,20):
    a = numpy.random.randint(3)
    print(a)

from torchvision import  transforms