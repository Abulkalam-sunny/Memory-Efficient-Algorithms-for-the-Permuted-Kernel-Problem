"""
Time complexity of Algorithm 3 versus Georgiades' algorithm.  All costs are log2.

    T_Georgiades = n! / (r+2)!

    T_Alg3 = T_ISD + T_tab + T_BS + L_fin,   with  k = r + 1,  nu = n - w + 2d,
                                                   n0 = n - k - w + d,
        T_ISD = (k^2 n + (1 + eps) C(k,d) d n) / (1 - (1-p)^N)            
                p   = C(w,d) C(n-w,k-d) / C(n,k)
                eps = C(n-k,n0) C(w,d) d q^(-d n0)   
        T_tab = C(n, w-2d) * d * nu * (nu-1)
        T_BS  = sum_{j=0..d} L_j + nu (nu-1) / q * sum_{j=0..d-1} L_j
        L_j   = n! / ((nu-2j)! q^j),                       j = 0..d       
        L_fin = n! / ((r+2-d)! q^d) 
        
    N = N_check^UT_{w,d} with k = r + 1:
        N = C(n,w) (q-1)^d q^{d(d-1)/2} (q^d-1)^{w-2d} [k d]_q / [n d]_q


Constraints:  d >= 1,  2d <= w <= n - r - 2 + d,  N >= 1.
"""
from math import lgamma, log, log2, log1p, expm1


def lfact(x):                       # log2(x!)
    return lgamma(x + 1) / log(2)


def lbinom(a, b):                   # log2 C(a, b)
    if b < 0 or b > a:
        return float("-inf")
    return lfact(a) - lfact(b) - lfact(a - b)


def lgauss(a, b, q):                # log2 of the Gaussian binomial [a b]_q
    return sum(log2(q ** (a - i) - 1) - log2(q ** (i + 1) - 1) for i in range(b))


def lsum(*xs):                      # log2(2^x1 + 2^x2 + ...)
    m = max(xs)
    return m + log2(sum(2 ** (x - m) for x in xs))


def log2_T_georgiades(n, r):
    return lfact(n) - lfact(r + 2)


def log2_N_ut(n, k, q, d, w):
    """log2 of the lower bound N_check^UT_{w,d} of Theorem 1 (code of dimension k)."""
    return (lbinom(n, w) + d * log2(q - 1) + d * (d - 1) / 2 * log2(q)
            + (w - 2 * d) * log2(q ** d - 1) + lgauss(k, d, q) - lgauss(n, d, q))


def log2_eps(n, k, q, d, w):
    """log2 of eps = C(n-k,n0) C(w,d) d q^(-d n0), the relative cost of the calls to
    Algorithm 1 within an ISD iteration (n0 = n - k - w + d)."""
    n0 = n - k - w + d
    return lbinom(n - k, n0) + lbinom(w, d) + log2(d) - d * n0 * log2(q)


def log2_T_isd(n, k, q, d, w, logN):
    """log2 of T_ISD = (k^2 n + (1+eps) C(k,d) d n) / (1 - (1-p)^N),
    with p = C(w,d) C(n-w,k-d) / C(n,k) for a code of dimension k."""
    log_one_plus_eps = lsum(0.0, log2_eps(n, k, q, d, w))
    log_iter = lsum(log2(k * k * n), lbinom(k, d) + log2(d * n) + log_one_plus_eps)
    log_p = lbinom(w, d) + lbinom(n - w, k - d) - lbinom(n, k)
    s = logN + log_p                     # log2(N p)
    if s > 6:                            # N p > 64: success probability is 1 - e^{-64} ~ 1
        log_P = 0.0
    elif s < -30:                        # N p tiny: 1 - (1-p)^N ~ N p
        log_P = s
    elif log_p < -50:                    # p tiny: (1-p)^N ~ e^{-N p}
        log_P = log2(-expm1(-2.0 ** s))
    else:                                # moderate p and N: exact formula
        log_P = log2(-expm1(2.0 ** logN * log1p(-2.0 ** log_p)))
    return log_iter - log_P


def log2_T_alg3(n, r, q, d, w):
    """log2 of the total cost of Algorithm 3, or None if (d, w) is not
    admissible."""
    if d < 1 or w < 2 * d or w > n - r - 2 + d:
        return None
    k = r + 1                            # dimension of the code spanned by B = (A;1)
    logN = log2_N_ut(n, k, q, d, w)
    if logN < 0:
        return None
    nu = n - w + 2 * d
    T_isd = log2_T_isd(n, k, q, d, w, logN)
    T_tab = lbinom(n, w - 2 * d) + log2(d) + log2(nu * (nu - 1))
    L = [lfact(n) - lfact(nu - 2 * j) - j * log2(q) for j in range(d + 1)]
    inspected = [L[j] + log2(nu * (nu - 1)) - log2(q) for j in range(d)]
    T_bs = lsum(*L, *inspected)
    L_fin = lfact(n) - lfact(r + 2 - d) - d * log2(q)
    return dict(total=lsum(T_isd, T_tab, T_bs, L_fin), logN=logN, T_isd=T_isd,
                T_tab=T_tab, T_bs=T_bs, L_fin=L_fin)


def optimize(n, r, q):
    """Best (d, w) for Algorithm 3: returns (d, w, result dict).
    If no UT-structured subcode is expected for any (d, w), Algorithm 3
    reduces to Georgiades' algorithm and we report d = w = 0."""
    best = None
    for d in range(1, r + 1):
        for w in range(2 * d, n - r - 2 + d + 1):
            res = log2_T_alg3(n, r, q, d, w)
            if res and (best is None or res["total"] < best[2]["total"]):
                best = (d, w, res)
    if best is None:
        best = (0, 0, dict(total=log2_T_georgiades(n, r), logN=float("nan"), T_isd=0.0,
                           T_tab=0.0, T_bs=0.0, L_fin=0.0))
    return best


def table(param_sets):
    head = (f"{'n':>4} {'r':>4} {'q':>6} | {'T_G':>7} | {'d':>2} {'w':>3} "
            f"{'log_2(N)':>8} {'T_ISD':>7} {'T_tab':>7} {'T_BS':>7} {'L_fin':>7} "
            f"{'T_Alg3':>7} | {'gain':>5}")
    print(head)
    print("-" * len(head))
    for (n, r, q) in param_sets:
        TG = log2_T_georgiades(n, r)
        d, w, res = optimize(n, r, q)
        print(f"{n:>4} {r:>4} {q:>6} | {TG:>7.1f} | {d:>2} {w:>3} {res['logN']:>8.1f} "
              f"{res['T_isd']:>7.1f} {res['T_tab']:>7.1f} {res['T_bs']:>7.1f} "
              f"{res['L_fin']:>7.1f} {res['total']:>7.1f} | {TG - res['total']:>5.1f}")


if __name__ == "__main__":
    table([(64,27,2048),(92,43,2048),(118,59,2048),(61,28,997),(87,42,1409),(111,55,1889),(69,41,251),(94,54,509),(106,47,4093)])