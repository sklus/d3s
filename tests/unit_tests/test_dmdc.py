import numpy as np
from pprint import pprint

from d3s.algorithms import dmdc

class UnstableSystem1(object):
    """
    Unstable system which DMDc can stabilize
    """
    def __init__(self, x0=np.array([1,0]).T):
        self.x = x0
        self.A = np.array(
            [[1.5, 0],
             [0, 0.1]]
        )
        self.B = np.array([1,0]).reshape(-1,1)

    def step(self, u):
        x = self.x
        x_prime = self.A @ x + self.B @ u
        self.x = x_prime
        return x_prime

def K(x):
    """ compute state-dependent control variable u """
    u = -1*x[0] + np.random.randn()
    return u.reshape(-1, 1)

# simulate system to generate data matrices
m = 1000 # number of sample steps from the system.
n = 2 # dimensionality of state space
q = 1 # dimensionality of control space

# State snapshotting
x0 = np.array([4,7]).reshape(-1, 1)
snapshots = np.zeros((n, m))
snapshots[:, 0] = np.squeeze(x0)

# Control snapshotting
U = np.zeros((q, m-1))
sys = UnstableSystem1(x0)
for k in range(m-1):
    u_k = K(sys.x)
    y = sys.step(u_k)
    snapshots[:, k+1] = np.squeeze(y)
    U[:, k] = u_k

X = snapshots[:, :m-1]
Y = snapshots[:, 1:m]

A_approx, B_approx, Phi = dmdc(X, Y, U)
print("\nApproximation of A")
print("True")
pprint(sys.A)
print("Predicted")
pprint(A_approx)

print("\nApproximation of B")
print("True")
pprint(sys.B)
print("Predicted")
pprint(B_approx)


