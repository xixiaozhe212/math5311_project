import numpy as np
import matplotlib.pyplot as plt

q = 1.0
n = 200
n2 = 100
n3 = 50
tl = 0.0
tr = 0.0
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
def gsIter(aw, ap, ae, su, phi, itr):
    n = np.size(aw)
    phi0 = phi
    for i in range(itr):
        phi[0] = (ae[0]*phi0[1]+su[0])/ap[0]
        for i in range(n-1):
            phi[i] = (aw[i]*phi[i-1]+ae[i]*phi0[i+1]+su[i])/ap[i]

        phi[n-1] = (aw[n-1]*phi[n-2]+su[n-1])/ap[n-1]
        for i in range(n):
            phi0[i] = phi[i]

    return phi


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
        aw[i] = num/l
        ae[i] = aw[i]
        su[i] = q*l/num
        ap[i] = aw[i]+ae[i]

    aw[0] = 0
    ae[num-1] = 0
    ae[0] = num/l
    aw[num-1] = num/l
    su[0] = q*l/n+2*num/l*tl
    su[num-1] = q*l/n+2*num/l*tr
    ap[0] = aw[0]+ae[0] + 2*num/l
    ap[num-1] = aw[num-1]+ae[num-1] + 2*num/l
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
    aw3, ap3, ae3, su3 = getMatrix(int(n/4))
    phi = np.zeros(n)
    phi = gsIter(aw, ap, ae, su, phi, 3)

    res = 1
    cnt = 0
    res_list = []
    while(res > 1e-8):
        # finest mesh
        phi = gsIter(aw, ap, ae, su, phi, 2)
        r = residual(su, aw, ap, ae, phi)
        res = np.mean(r)
        res_list.append(res)
        cnt += 1
        print("iter: ", cnt, " res:", res)
        # ------------- restriction -----------------
        r2 = restrict(r)
        e2 = gsIter(aw2, ap2, ae2, r2, np.zeros(int(n/2)), 10)
        r4 = residual(r2, aw2, ap2, ae2, e2)
        rr = restrict(r4)
        #e4 = gsIter(aw3,ap3,ae3,rr,np.zeros(int(n/4)),10)
        e4 = sol(aw3, ap3, ae3, rr)
        # ------------ prolongation -----------------
        eff = prolong(e4)
        e2 = e2+eff
        e2 = gsIter(aw2, ap2, ae2, r2, e2, 2)
        ef = prolong(e2)
        phi = phi + ef

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
    plt.plot(phixx)
    plt.title("uxx")
    plt.ylim([-0.9,-1.1])

    plt.show()

