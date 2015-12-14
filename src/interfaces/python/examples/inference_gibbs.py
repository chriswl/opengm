import opengm
import numpy

from time import time

shape = [20, 20]
nl = 100
unaries = numpy.random.rand(*shape + [nl])
potts = opengm.PottsFunction([nl] * 2, 0.0, 0.4)
gm = opengm.grid2d2Order(unaries=unaries, regularizer=potts)


inf = opengm.inference.Gibbs(gm, parameter=opengm.InfParam(steps=10000))
# start inference (in this case unverbose infernce)

t0 = time()
v = inf.verboseVisitor()
inf.infer(v)
t1 = time()

print t1 - t0

# get the result states
argmin = inf.arg()
# print the argmin (on the grid)
print argmin.reshape(*shape)
