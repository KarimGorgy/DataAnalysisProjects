import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def func(t):
    return t*2
x =np.array([1,2,3,4,4])
plt.figure(1)
plt.subplot(211)
plt.plot(x,func(x))
plt.subplot(212)
plt.plot([1,2,3])
plt.plot([4,5,6,6,6])
plt.show()