import simpy as sp


def test():
    x, t, a, n, m, i = sp.symbols("x t a n m i")
    n._is_int = True
    m._is_int = True
    f = sp.cos(n * sp.pi * (x - t) / a) * sp.cos(m * sp.pi * x / a) - sp.sin(n * sp.pi * (x - t) / a) * sp.sin(
        m * sp.pi * x / a
    )
    T = 2 * a
    ans = sp.integrate(f, (t, -a, a)) / T  # 0 when n, m are integers.

    fim = sp.cos(n * sp.pi * (x - t) / a) * sp.sin(m * sp.pi * x / a) + sp.sin(n * sp.pi * (x - t) / a) * sp.cos(
        m * sp.pi * x / a
    )
    ans2 = sp.integrate(fim, (t, -a, a)) / T  # 0 when n, m are integers.
    breakpoint()
