#Random data
X = np.random.randint(10,100,100)
X = X.reshape(20,5)
print(X)
#Tinh trung binh cot
mean = np.mean(X, axis=0)
X_centered = X - mean
#Tinh cov
cov_matrix = np.cov(X_centered.T)
print(cov_matrix)
#
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
#Sap xep
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvectors_sorted = eigenvectors[:, sorted_indices]
#lay 3 cai cao nhat
components = eigenvectors_sorted[:, :3]

X_reduced = np.dot(X_centered, components)
print(X_reduced.shape)

CenterPoint = np.array([0,0,0])
CovPoints = np.cov(X_reduced.T)
n=20
Points = np.random.multivariate_normal(CenterPoint, CovPoints, n).T
display(Points)
print(Points.shape)

A = np.cov(Points)
display(A)

e, v = np.linalg.eig(A)
print("Eigen Values :")
print(e)
print("Eigen Vectors :")
print(v)

plt.figure(figsize = (10,6))
plt.scatter(Points[0,:], Points[1, :], color = "b", alpha = 0.2)
for ie,iv in zip(e, v.T):
  print(iv)
  plt.plot([0, 3*np.sqrt(ie) * iv[0]], [0, 3*np.sqrt(ie)* iv[1]], "r--", lw=3)
plt.show()