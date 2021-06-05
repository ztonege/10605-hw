import numpy as np
import math

x = np.array([3,4,5,6])
y = np.array([4,3,2,1])

'''
z = [np.array([0.16,-0.26,-1.12,0.01]),
     np.array([1.08,-1.14,0.8,-0.8]),
     np.array([1.94,0.08,1.83,0.17]),
     np.array([-0.88,-1.01,1.36,-0.07])]
'''

z = [np.array([1,-1,-1,1]),
     np.array([1,-1,1,-1]),
     np.array([1,1,1,1]),
     np.array([-1,-1,1,-1])]
# z = [np.array([0.16,-0.26,-1.12,0.01])]

def LSH_err(x, y, z):
    x_bits = []
    y_bits = []
    for plane in z:
        if x.dot(plane) >= 0:
            x_bits.append(1)
        else:
            x_bits.append(0)
        if y.dot(plane) >= 0:
            y_bits.append(1)
        else:
            y_bits.append(0)

    h = 0
    b = len(z)
    for i in range(len(x_bits)): 
        if x_bits[i] != y_bits[i]:
            h += 1
    estimate = 1 - math.cos(h / b * math.pi)
    true = 1 - x.dot(y) / math.sqrt((x ** 2).sum()) / math.sqrt((y ** 2).sum())
    return x_bits, y_bits ,estimate, true, np.abs(estimate - true)


x_bits, y_bits, estimate, true, err = LSH_err(x,y,z)
print("x_bits: ", x_bits)
print("y_bits: ", y_bits)
print("estimate: %.4f" %estimate)
print("true: %.4f" %true)
print("error: %.4f" %err)
