import numpy as np
import matplotlib.pyplot as plt

n = 200
tl = 0.0
tr = 1.0
l = 1.0


def restrict(vec):
    n = int(np.size(vec)/2)
    vec_coarse = np.zeros(n)
    for i in range(n):
        vec_coarse[i] = (vec[2*i]+vec[2*i+1])/2

    return vec_coarse


def prolong(vec):
    n = np.size(vec)
    vec_fine = np.ones(n*2)*vec[0]/2
    for i in range(1, n*2):
        if (i % 2 == 0):
            vec_fine[i] = (vec[int(i/2) - 1] + vec[int(i/2)])/2
        else:
            vec_fine[i] = vec[int(i/2)]

    return vec_fine


# iter stands for itereation times
def jIter(A, b, x0, iter):
    n = len(b)
    x = x0.copy()
    for _ in range(iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            sum_j = np.dot(A[i, :], x).item() - A[i, i] * x[i]
            x_new[i] = (b[i] - sum_j) / A[i, i]
        x = x_new
    return x

def residual(su, aw, ap, ae, phi):
    n = np.size(su)
    res = np.zeros(n)
    for i in range(1, n-1):
        res[i] = su[i] - (-aw[i]*phi[i-1]+ap[i]*phi[i]-ae[i]*phi[i+1])
        res[0] = su[0] - ap[0]*phi[0] + ae[0]*phi[1]
        res[n-1] = su[n-1] - ap[n-1]*phi[n-1] + aw[n-1]*phi[n-2]

    return res


def getMatrix(num):
    aw = np.zeros(num)
    ap = np.zeros(num)
    ae = np.zeros(num)
    su = np.zeros(num)
    for i in range(1, num-1):
        aw[i] = 1
        ae[i] = 0
        su[i] = l/num
        ap[i] = aw[i]+ae[i]

    aw[0] = 0
    ae[num-1] = 0
    ae[0] = 0
    aw[num-1] = 1
    su[0] = l/n+2*tl
    # su[num-1] = l/n - 2*tr
    su[num-1] = l/n
    ap[0] = 2
    ap[num-1] = 1
    return aw, ap, ae, su
    

def sol(aw, ap, ae, su):
    n = np.size(aw)
    p = np.zeros(n)
    q = np.zeros(n)
    phi = np.zeros(n)
    p[0] = ae[0]/ap[0]
    q[0] = su[0]/ap[0]
    for i in range(1, n):
        p[i] = ae[i]/(ap[i] - aw[i]*p[i-1])
        q[i] = (su[i] + aw[i]*q[i-1])/(ap[i]-aw[i]*p[i-1])

    phi[n-1] = q[n-1]
    for i in range(n-1, 0, -1):
        phi[i-1] = p[i-1]*phi[i]+q[i-1]
    return phi

if __name__ == '__main__':
    # prepare matrix
    aw, ap, ae, su = getMatrix(n)
    aw2, ap2, ae2, su2 = getMatrix(int(n/2))
    A = np.diag(ap) + np.diag(-ae[:-1], 1) + np.diag(-aw[1:], -1)
    phi = np.zeros(n)

    # Forward - smoothing
    phi = jIter(A, su, phi, 3)
    
    res = 1e6
    cnt = 0
    res_list = []

    L2b = np.linalg.norm(su, 2)

    while(res > 1e-8*L2b):
        # finest mesh
        phi = jIter(A, su, phi, 2)
        r = residual(su, aw, ap, ae, phi)
        res = np.linalg.norm(r, 2)
        res_list.append(res)
        cnt += 1
        print("iter: ", cnt, " res:", res)
        # ------------- restriction -----------------
        r2 = restrict(r)
        e2 = sol(aw2, ap2, ae2, r2)
        # e20 = np.zeros(int(n/2))
        # e2 = gsIter(A2, r2, e20, 4)
        # ------------ prolongation -----------------
        ef = prolong(e2)
        phi = phi + ef
        # Backward - smoothing
        phi = jIter(A, su, phi, 1)

    phix = np.zeros(n-2)
    phixx = np.zeros(n-4)
    for i in range(n-2):
        phix[i] = (phi[i+2]-phi[i])/(2*l/n)
    
    for i in range(n-4):
        phixx[i] = (phix[i+2]-phix[i])/(2*l/n)

    plt.subplot(1,3,1)
    plt.plot(res_list)
    plt.yscale('log')
    plt.title("mean residual")

    plt.subplot(1,3,2)
    plt.plot(phi)
    plt.title("solution")

    plt.subplot(1,3,3)
    plt.plot(phix)
    plt.title("ux")
    plt.ylim([0.9,1.1])

    plt.show()