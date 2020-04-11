import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
a = np.loadtxt('Sharp_char.txt')

x = a[:,[1]]
y = a[:,[0]]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

c = np.hstack([x*x*x, x*x, x, np.ones(x.shape)])
v = np.linalg.pinv(c) @ y

c1 = np.hstack([1/x, np.ones(x.shape)])
v1 = np.linalg.pinv(c1) @ y


plt.plot(x, y, 'ro')
plt.plot(x,v[0]*x*x*x + v[1]*x*x + v[2]*x + v[3],)
plt.plot(x,v1[0]/x + v1[1])
plt.show()

