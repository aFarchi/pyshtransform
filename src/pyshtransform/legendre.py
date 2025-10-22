import numpy as np

def pre_glq(x1, x2, n):
    zero = np.zeros(n)
    w = np.zeros(n)
    m = (n+1) // 2
    xm = (x2 + x1) / 2
    xu = (x2 - x1) / 2
    for i in range(1, m+1):
        z=np.cos(np.pi * (i-0.25) / (n+0.5))
        for the_iter in range(1000):
            p1=1
            p2=0
            for j in range(1, n+1):
                p3 = p2
                p2 = p1
                p1 = ((2*j-1)*z*p2-(j-1)*p3) / j
            pp = n * (z * p1 - p2) / (z * z-1)
            z1 = z
            z = z1-p1 / pp
            if abs(z-z1) <= 1e-15:
                break
        zero[i-1] = xm + xu * z
        zero[n+1-i-1] = xm - xu * z
        w[i-1] = 2 * xu / ((1-z * z) * pp *pp)
        w[n+1-i-1] = w[i-1]
    return zero, w


def plmbar_d1(lmax, z):
    p = np.zeros((lmax + 1) * (lmax + 2) // 2)
    dp1 = np.zeros((lmax + 1) * (lmax + 2) // 2)

    scalef = 1.0e-280

    sqr = np.zeros(2 * lmax + 1)
    for l in range(1, 2 * lmax + 2):
        sqr[l - 1] = np.sqrt(l)

    f1 = np.zeros(((lmax + 1) * (lmax + 2)) // 2)
    f2 = np.zeros(((lmax + 1) * (lmax + 2)) // 2)
    k = 3
    for l in range(2, lmax + 1):
        k += 1
        f1[k - 1] = sqr[2 * l - 2] * sqr[2 * l] / l
        f2[k - 1] = (l - 1) * sqr[2 * l] / sqr[2 * l - 4] / l
        for m in range(1, l - 1):
            k += 1
            f1[k - 1] = (
                    sqr[2 * l] * sqr[2 * l - 2] /
                    sqr[l + m - 1] / sqr[l - m - 1]
            )
            f2[k - 1] = (
                    sqr[2 * l] * sqr[l - m - 2] * sqr[l + m - 2] /
                    sqr[2 * l - 4] / sqr[l + m - 1] / sqr[l - m - 1]
            )
        k += 2

    u = np.sqrt((1.0 - z) * (1.0 + z))
    pm2 = 1.0
    p[0] = 1.0
    dp1[0] = 0.0
    pm1 = sqr[2] * z
    p[1] = pm1
    dp1[1] = sqr[2]
    k = 2
    for l in range(2, lmax + 1):
        k += l
        plm = f1[k - 1] * z * pm1 - f2[k - 1] * pm2
        p[k - 1] = plm
        dp1[k - 1] = l * (sqr[2 * l] / sqr[2 * l - 2] * pm1 - z * plm) / u ** 2
        pm2 = pm1
        pm1 = plm

    pmm = scalef
    rescalem = 1.0 / scalef
    kstart = 1
    for m in range(1, lmax):
        rescalem = rescalem * u
        kstart += m + 1
        pmm = pmm * sqr[2 * m] / sqr[2 * m - 1]
        p[kstart - 1] = pmm * rescalem
        dp1[kstart - 1] = -m * z * p[kstart - 1] / u ** 2
        pm2 = pmm
        k = kstart + m + 1
        pm1 = z * sqr[2 * m + 2] * pmm
        p[k - 1] = pm1 * rescalem
        dp1[k - 1] = (sqr[2 * m + 2] * p[k - m - 2] - z * (m + 1) * p[k - 1]) / u ** 2
        for l in range(m + 2, lmax + 1):
            k += l
            plm = z * f1[k - 1] * pm1 - f2[k - 1] * pm2
            p[k - 1] = plm * rescalem
            dp1[k - 1] = (sqr[2 * l] * sqr[l - m - 1] * sqr[l + m - 1] / sqr[2 * l - 2] * p[k - l - 1] - z * l * p[
                k - 1]) / u ** 2
            pm2 = pm1
            pm1 = plm

    rescalem = rescalem * u
    kstart += lmax + 1
    pmm = pmm * sqr[2 * lmax] / sqr[2 * lmax - 1]
    p[kstart - 1] = pmm * rescalem
    dp1[kstart - 1] = -lmax * z * p[kstart - 1] / u ** 2

    return p, dp1

