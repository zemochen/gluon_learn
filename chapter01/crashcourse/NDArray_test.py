from mxnet import ndarray as nd
import numpy as np

print(nd.zeros((3,4)))


def _print_console(arg):
    print(arg)


x = nd.ones((3,4))


y = nd.random_normal(0,1,shape=(3,4))
print(y)
print(y.shape)
print(x)

y = nd.array(x)
z = y.asnumpy()

print([z,y])


