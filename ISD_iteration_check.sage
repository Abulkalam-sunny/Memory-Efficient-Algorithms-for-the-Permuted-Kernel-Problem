# Number of ISD iterations: Algorithms 1 and 2 of the paper, in SageMath.
#
# For random [n, k] codes, Algorithm 2 is repeated until it returns a UT-structured
# subcode of dimension d and support size w.  The average number of iterations is
# compared with the prediction 1 / (1 - (1-p)^N) of Section 4.2, where
# N = N_check^UT_{w,d} is given by (eq:11) and p = p(n, k, d, w).

from itertools import combinations

n, k, q, d, w = 39, 29, 41, 2, 10     # length, dimension, field size, subcode dimension, support size
TRIALS = 10000                        # number of random codes
F = GF(q)


def check_ut(G, K, J):
    """Procedure CheckUT of Algorithm 1.  G is systematic with respect to K (its
    columns indexed by K, in increasing order, form I_d), and J is the support.
    Returns (G, order) with G*P = (I_d | U | M | 0), or None.  The permutation P is
    the list order: column j of G*P is column order[j] of G."""
    L, R, chosen = [c for c in J if c not in K], list(range(d)), []
    while True:
        # a pair (c, rho) such that, among the rows in R, column c is nonzero only in row rho
        found = [(c, rho) for c in L for rho in R
                 if [i for i in R if G[i, c] != 0] == [rho]]
        if not found:
            break
        c, rho = found[0]
        chosen.append((c, rho))
        L.remove(c)
        R.remove(rho)
    if R:
        return None
    G = G.matrix_from_rows([rho for c, rho in chosen])               # reorder the rows as rho_1..rho_d
    # column K[rho] has its unit entry in row rho, so it goes to the block I_d;
    # the columns c_1..c_d go to the block U, the rest of J to M, the others to 0
    order = ([K[rho] for c, rho in chosen] + [c for c, rho in chosen]
             + L + [c for c in range(n) if c not in J])
    return G, order


def algorithm1(GB):
    """Algorithm 1.  Returns (G, order) with G*P = (I_d | U | M | 0), or None."""
    J = [c for c in range(n) if not GB.column(c).is_zero()]          # J = Supp(B)
    for K in combinations(J, d):                                     # every information set lies in J
        GK = GB.matrix_from_columns(K)
        if GK.det() == 0:                                            # K is not an information set
            continue
        out = check_ut(GK.inverse() * GB, K, J)                      # systematic w.r.t. K
        if out is not None:
            return out
    return None


def algorithm2(Y):
    """Algorithm 2 (one ISD iteration).  Returns the output of Algorithm 1, or None."""
    I = sorted(sample(range(n), k))                                  # random set of k positions
    YI = Y.matrix_from_columns(I)
    if YI.det() == 0:                                                # I is not an information set
        return None
    Yp = YI.inverse() * Y                                            # Y' = Y_I^(-1) Y
    # rows of Y' indexed by I: y_i is the row with its unit entry in column i
    row_of = {i: b for b, i in enumerate(I)}
    supp = {i: set(Yp.row(row_of[i]).support()) for i in I}         # nonzero columns of y_i
    for S in combinations(I, d):                                     # S subset of I, |S| = d
        if len(set().union(*[supp[i] for i in S])) == w:             # G_S has w nonzero columns
            GS = Yp.matrix_from_rows([row_of[i] for i in S])
            out = algorithm1(GS)
            if out is not None:
                return out
    return None

def algorithm2(Y):
    """Algorithm 2 (one ISD iteration). Returns the output of Algorithm 1, or None."""
    I = sample(range(n), k)
    YI = Y.matrix_from_columns(I)
    if YI.det() == 0:
        return None
    Yp = YI.inverse() * Y                                            # Y' = Y_I^(-1) Y
    supp = [set(Yp.row(b).support()) for b in range(k)]              # nonzero columns of each row
    for S in combinations(range(k), d):
        if len(set().union(*[supp[b] for b in S])) == w:             # G_S has w nonzero columns
            out = algorithm1(Yp.matrix_from_rows(S))
            if out is not None:
                return out
    return None



# prediction of Section 4.2: 1 / (1 - (1-p)^N), with N = N_check^UT_{w,d}
gauss = lambda a, b: prod((q^(a-i) - 1) / (q^(i+1) - 1) for i in range(b))   # [a b]_q
N = binomial(n, w) * (q-1)^d * q^(d*(d-1)//2) * (q^d - 1)^(w-2*d) * gauss(k, d) / gauss(n, d)
prob = binomial(w, d) * binomial(n-w, k-d) / binomial(n, k)
predicted = 1 / (1 - (1 - RR(prob))^RR(N))
# probability that a fixed set I is an information set of a random [n, k] code
p_info = RR(q^(k*(n-k)) / gauss(n, k))

counts = []
for _ in range(TRIALS):
    Y = random_matrix(F, k, n)                                       # a random [n, k] code
    while Y.rank() < k:
        Y = random_matrix(F, k, n)
    it, out = 0, None
    while out is None:                                               # repeat Algorithm 2
        it += 1
        out = algorithm2(Y)
    counts.append(it)

    # check the output: G*P = (I_d | U | M | 0) and the rows of G lie in the code
    G, order = out
    assert sorted(order) == list(range(n))                           # P is a permutation
    GP = G.matrix_from_columns(order)
    U = GP.matrix_from_columns(range(d, 2*d))
    assert GP.matrix_from_columns(range(d)) == identity_matrix(F, d)
    assert all(U[i, j] == 0 for i in range(d) for j in range(i))     # upper triangular
    assert all(U[i, i] != 0 for i in range(d))                       # nonzero diagonal
    assert all(not GP.column(j).is_zero() for j in range(2*d, w))    # M has no zero column
    assert GP.matrix_from_columns(range(w, n)).is_zero()             # support size w
    assert Y.stack(G).rank() == k                                    # subcode of the code

mean = RR(sum(counts) / TRIALS)
error = sqrt(sum((x - mean)^2 for x in counts) / (TRIALS - 1) / TRIALS)
print("(n,k,q,d,w) =", (n, k, q, d, w), "  random codes:", TRIALS)
print("N_check   = %.2f" % N)
print("predicted = %.2f" % predicted)
print("measured  = %.2f +- %.2f" % (mean, error))
print("ratio     = %.3f" % (mean / predicted))
print("(I is an information set with probability %.3f, which the prediction ignores)" % p_info)
