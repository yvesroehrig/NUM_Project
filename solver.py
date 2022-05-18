import numpy as np
from scipy.linalg import qr, solve_triangular


# Lösungsvefahren nach explizit Euler
def explizitEuler(tend, h, x0, f):
	t = [0.]
	x = [x0]

	while t[-1] < tend-h/2:
		x.append(x[-1]+h*f(t[-1],x[-1]))
		t.append(t[-1]+h)
	return np.array(t, dtype=np.float64), np.array(x, dtype=np.float64)



# Lösungsvefahren nach implizit Euler
def implizitEuler(tend, h, x0, f, df):
	t = [0.]
	x = [x0]
	e = np.eye(x0.shape[0])

	# Verfahrensfunktion für implizit Euler
	def G(r1, tk, xk):
		return r1 - f(tk+h, xk+h*r1)

	# Partielle Ableitung nach r1 der Verfahrensfunktion
	def dG(r1, tk, xk):
		return e - h*df(tk+h, xk+h*r1)

	def newton(r1, tk, xk, tol=1e-12, maxIter=20):
		k = 0
		delta = np.ones_like(xk)*10*tol
		while np.linalg.norm(delta,np.inf) > tol and k < maxIter:
			A = dG(r1, tk, xk)
			b = G(r1, tk, xk)
			Q,R = np.linalg.qr(A)
			delta = solve_triangular(R,Q.T@b)
			r1 -= delta
			k += 1
		return r1

	r1 = f(t[-1],x[-1])
	while t[-1] < tend-h/2:
		r1 = newton(r1, t[-1], x[-1])
		x.append(x[-1]+h*r1)
		t.append(t[-1]+h)
	return np.array(t, dtype=np.float64), np.array(x, dtype=np.float64)



# Lösungsvefahren nach implizit Trapez
def implizitTrapez(xend, h, y0, f, df):
	ids = np.eye(y0.shape[0])

	x = [0.]
	y = [y0]

	def G(r, xk, yk):
		return r - f(xk+h, yk+h*r)

	def dG(r, xk, yk):
		return ids- df(xk+h,yk+h*r)*h

	tol = 1e-11
	kmax = 10

	r2 = f(0,y0)
	while x[-1] < xend-h/2:
		r1 = f(x[-1],y[-1])
		k=0
		while np.linalg.norm(G(r2,x[-1],y[-1])) > tol and k < kmax:
			A = dG(r2,x[-1],y[-1])
			B = G(r2,x[-1],y[-1])
			Q, R = qr(A)
			delta = solve_triangular(R,Q.T@B, lower=False)
			r2 -= delta
			k += 1

		y.append(y[-1]+h/2*(r1+r2))
		x.append(x[-1]+h)

	return np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)
