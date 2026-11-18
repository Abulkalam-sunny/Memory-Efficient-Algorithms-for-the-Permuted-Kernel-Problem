# Algorithm 3 of the paper (memory-efficient algorithms for PKP), in SageMath.
#
# The instances have a planted solution x_star and are already in the shape of
# Fig. 1, so Algorithm 2 and the matrices Q and P are not needed (P = identity):
#   rows 1..d     of A are  ( I_d | U | M | 0 )         columns 1..w      are z_I, z_U, z_M
#   rows d+1..r   of A are  ( H | I_{r-d} | 0 )         columns w+1..w+n0 are z_F
#   the last coordinate follows from  sum x_i = sigma_1  (the all-ones row of B).
# Hence the structured equations are G x = s with s = 0, and the last r+1-d
# coordinates z_D are affine functions of the first w+n0 ones.
# The script counts every step of the algorithm and compares the averages with the
# formulas of the paper (nu = n - w + 2d):
#   table entries     C(n, w-2d) d nu(nu-1)  for all sets V,  d nu(nu-1) in memory
#   nodes L_j         n! / ((nu-2j)! q^j),   j = 0, ..., d                   
#   inspected pairs   nu(nu-1)/q * L_j,      j = 0, ..., d-1  (lookups by the nodes
#                     at depth j, part of T_BS)
#   completions       L_fin = n! / ((r+2-d)! q^d)                             
#   roots of (eq:8)  about L_fin  (a random quadratic has one root on average)
# It also checks that every output x is a solution and that x_star is found.


from itertools import combinations, permutations

n, r, q, d, w = 17, 7, 157, 3, 10      # n! < q^r
n0 = n - r - 1 - w + d                 # number of free coordinates z_F
k = w + n0                             # z_I, z_U, z_M, z_F come first
TRIALS = 100                           # number of random instances
F = GF(q)
assert factorial(n) < q^r and 2*d <= w and n0 >= 1


def planted_instance():
    """Random PKP instance (A, b) with a planted solution x_star, in the shape above."""
    b = vector(F, sample(list(F), n))                    # n distinct entries
    x_star = vector(F, sample(list(b), n))               # a random arrangement of b
    rows = []
    for i in range(d):                                   # ( I_d | U | M | 0 )
        tries = 0
        while True:                                      # until the row is orthogonal to x_star
            tries += 1
            if tries > q^2:
                return planted_instance()                # no such row (possible when w = 2d)
            g = zero_vector(F, n)
            g[i] = 1                                     # I_d
            g[d+i] = randint(1, q-1)                     # u_ii, nonzero
            for l in range(i+1, d):
                g[d+l] = randint(0, q-1)                 # u_il, l > i
            for j in range(2*d, w):
                g[j] = randint(1, q-1)                   # M, no zero column
            if g * x_star == 0:
                break
        rows.append(g)
    for j in range(r - d):                               # ( H | I_{r-d} | 0 )
        while True:
            g = zero_vector(F, n)
            for c in range(k):
                g[c] = randint(0, q-1)                   # H
            g[k + j] = 1                                 # I_{r-d}
            if g * x_star == 0:
                break
        rows.append(g)
    return matrix(F, rows), b, x_star


def algorithm3(A, b):
    """Algorithm 3.  Returns all x in S_n(b) with A x = 0, and the counts of each step,
    or None if no free coordinate y has e^T e != 0 (then the instance is skipped)."""
    count = {'tab': 0, 'mem': 0, 'L': [0]*(d+1), 'insp': [0]*d, 'fin': 0, 'roots': 0}
    U, M, H = A[:d, d:2*d], A[:d, 2*d:w], A[d:, :k]     # planted UT structure
    s = zero_vector(F, d)                                # right-hand side of G x = s
    sigma1, sigma2 = sum(b), sum(v^2 for v in b)
    Ry = PolynomialRing(F, 'y')

    # Complete: x = a + y e, where y is one free coordinate.  e does not depend on the
    # enumerated coordinates, and we choose y such that e^T e != 0.
    def e_vector(yp):
        e_head = vector(F, [1 if c == yp else 0 for c in range(k)])
        e_D = -H * e_head                                # z_D = -H (z_I, z_U, z_M, z_F)
        return vector(F, list(e_head) + list(e_D) + [-sum(e_head) - sum(e_D)])
    free = list(range(w, k))                             # positions of z_F
    choices = [yp for yp in free if e_vector(yp) * e_vector(yp) != 0]
    if not choices:
        return None
    yp = choices[0]
    e = e_vector(yp)
    others = [p for p in free if p != yp]                # the n0 - 1 enumerated coordinates
    solutions = []

    def complete(zI, zU, zM, used):
        b1 = [v for v in b if v not in used]             # b', the n - w unused entries
        for zF in permutations(b1, int(n0 - 1)):         # S_{n0-1}(b')
            count['fin'] += 1                            # one completion candidate
            head = vector(F, zI + zU + zM + [0]*n0)      # first k coordinates, y = 0
            for p, v in zip(others, zF):
                head[p] = v
            z_D = -H * head                              # with y = 0
            last = sigma1 - sum(head) - sum(z_D)         # sum of x_i = sigma_1
            a = vector(F, list(head) + list(z_D) + [last])
            f = Ry([a*a - sigma2, 2*(a*e), e*e])         # degree 2
            roots = f.roots(multiplicities=False)
            count['roots'] += len(roots)
            for y0 in roots:
                x = a + y0*e
                if set(x) == set(b):                     # x in S_n(b)
                    solutions.append(x)

    def back_sub(i, zI, zU, zM, used):                   # rows i = d-1, ..., 0 (0-indexed)
        if i < 0:
            complete(zI, zU, zM, used)
            return
        theta = s[i] - (M*vector(F, zM))[i] - sum(U[i, l]*zU[l] for l in range(i+1, d))
        pairs = tables[i].get(theta, [])                 # all pairs with key theta
        count['insp'][d-1-i] += len(pairs)               # inspected by a node at depth d-1-i
        for v1, v2 in pairs:
            if v1 not in used and v2 not in used:        # discard reused entries
                zI[i], zU[i] = v1, v2
                count['L'][d-i] += 1                     # one node at depth j = d - i
                back_sub(i-1, zI, zU, zM, used | {v1, v2})

    for V in combinations(b, int(w - 2*d)):              # sets V of w - 2d entries
        b_rest = [v for v in b if v not in V]            # b \ V
        tables = []
        for i in range(d):                               # T_i = S_2(b \ V), key v1 + u_ii v2
            T = {}
            for v1, v2 in permutations(b_rest, int(2)):
                T.setdefault(v1 + U[i, i]*v2, []).append((v1, v2))
            tables.append(T)
        size = sum(len(pairs) for T in tables for pairs in T.values())
        count['tab'] += size                             # table entries built
        count['mem'] = max(count['mem'], size)           # entries stored at once
        for zM in permutations(V):                       # z_M in S_{w-2d}(V)
            count['L'][0] += 1                           # one node at depth 0
            back_sub(d-1, [0]*d, [0]*d, list(zM), set(zM))
    return solutions, count


total = {'tab': 0, 'mem': 0, 'L': [0]*(d+1), 'insp': [0]*d, 'fin': 0, 'roots': 0, 'sol': 0}
recovered, skipped, instance = 0, 0, 0
while instance < TRIALS:
    A, b, x_star = planted_instance()
    result = algorithm3(A, b)
    if result is None:                                   # no y with e^T e != 0
        skipped += 1
        continue
    solutions, count = result
    assert all(A*x == 0 and set(x) == set(b) for x in solutions)
    recovered += x_star in solutions
    for key in ['tab', 'mem', 'fin', 'roots']:
        total[key] += count[key]
    for j in range(d+1):
        total['L'][j] += count['L'][j]
    for j in range(d):
        total['insp'][j] += count['insp'][j]
    total['sol'] += len(solutions)
    instance += 1
    if instance % 10 == 0:
        print("instance %d done" % instance)

nu = n - w + 2*d
L = [factorial(n) / (factorial(nu-2*j) * q^j) for j in range(d+1)]
L_fin = factorial(n) / (factorial(r+2-d) * q^d)
rows = ([("table entries, all sets V", total['tab'], binomial(n, w-2*d) * d * nu * (nu-1)),
         ("table entries in memory", total['mem'], d * nu * (nu-1))]
        + [("L_%d" % j, total['L'][j], L[j]) for j in range(d+1)]
        + [("pairs inspected at depth %d" % j, total['insp'][j], nu*(nu-1)/q * L[j]) for j in range(d)]
        + [("L_fin (completions)", total['fin'], L_fin),
           ("roots of the quadratic", total['roots'], L_fin)])

print()
print("(n,r,q,d,w) = %s, averages over %d planted instances" % ((n, r, q, d, w), TRIALS))
print("%-28s %12s %12s %7s" % ("quantity", "measured", "predicted", "ratio"))
for name, measured, predicted in rows:
    measured = RR(measured / TRIALS)
    print("%-28s %12.1f %12.1f %7.3f" % (name, measured, RR(predicted), measured / RR(predicted)))
print("planted solution recovered in %d / %d instances" % (recovered, TRIALS))
print("solutions per instance: %.2f  (every output x satisfies A x = 0 and x in S_n(b))"
      % (total['sol'] / TRIALS))
print("instances skipped because no free coordinate has e^T e != 0: %d" % skipped)
