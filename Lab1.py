import  matplotlib.pyplot as plt
import numpy as np
CenterPoint = np.array([0, 0])
CovPoints = np.array([[0.6, .2], [0.2, 0.2]])
n=1000
Points = np.random.multivariate_normal(CenterPoint, CovPoints, n).T
display(Points)
print(Points.shape)

A= np.cov(Points)
display(A)

e, v = np.linalg.eig(A)
print("Eigen Values :")
print(e)
print("Eigen Vectors :")
print(v)

plt.figure(figsize = (18,8))
plt.scatter(Points[0,:], Points[1, :], color = "b", alpha = 0.2)
for ie,iv in zip(e, v.T):
  print(iv)
  plt.plot([0, 3*ie * iv[0]], [0, 3*ie* iv[1]], "r--", lw=3)
plt.show()