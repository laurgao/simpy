from typing import List, Literal, NamedTuple

import simpy as sp


@sp.expr.cast
def table(a, b, c, type: Literal["positive", "negative", "zero"]):
    """table of solutions for the ode ay'' + by' + cy = 0

    this only works for homogenous linear second order ODEs with constant coefficients.`

    Args:
        a: coefficient of y''
        b: coefficient of y'
        c: coefficient of y
        type: sign of discriminant
    """
    alpha = -b / (2 * a)
    d = b**2 - 4 * a * c
    T = sp.sqrt(-d) / (2 * a) if type == "negative" else sp.sqrt(d) / (2 * a)
    c1, c2, x = sp.symbols("c1 c2 x")

    if type == "positive":
        return sp.exp(alpha * x) * (c1 * sp.cos(T * x) + c2 * sp.sin(T * x))
    elif type == "negative":
        return sp.exp(alpha * x) * (c1 * sp.cosh(T * x) + c2 * sp.sinh(T * x))

    return c1 * sp.exp(alpha * x) + c2 * x * sp.exp(alpha * x)


# actually fuck this i just wanna solve it lol
# def table_sepvar_bcs(case, x):
#     """i guess funciton assumes that the shit matches

#     it's for sth like
#     y'' + lambda
#     """
#     if case == 1:
#         ode_soln = sp.sin(x * sp.sqrt(lambda_))


def test_table():
    expr = table(1, 1, -sp.expr.Rat(1, 4), "zero")
    c1, c2, x = sp.symbols("c1 c2 x")
    assert expr == c1 * sp.exp(-x / 2) + c2 * x * sp.exp(-x / 2)
    breakpoint()
    assert expr.diff(x).diff(x) + expr.diff(x) - sp.expr.Rat(1, 4) * expr == 0


def test_table2():
    n = sp.symbols("n", integer=True, positive=True)
    lambda_ = -((n - sp.expr.Rat(1, 2)) ** 2)
    expr = table(1, 1, lambda_, "positive")
    c1, c2, x = sp.symbols("c1 c2 x")
    breakpoint()


class BoundaryCondition(NamedTuple):
    value: int
    x_value: int
    coeff1: sp.Expr  # coeff of u term
    coeff2: sp.Expr  # coeff of u' term

    @property
    def is_homogenous(self):
        return self.value == 0


class PDE(NamedTuple):
    cxx: sp.Expr
    cx: sp.Expr
    ct: sp.Expr
    ctt: sp.Expr


# class LinearODE:
#     # a list of coeffs probably lol where like the index is the ...
#     coeffs: List[sp.Expr]


class ODE(NamedTuple):
    # sorry im lazy
    c0: sp.Expr  # coeff of y
    c1: sp.Expr  # of y'
    c2: sp.Expr  # of y''

    @property
    def discriminant(self):
        return self.c1**2 - 4 * self.c0 * self.c2


def solve_pde_with_seperation_of_variables(pde: PDE, bcs: List[BoundaryCondition]) -> sp.Expr:
    """
    ## STEP 1: create seperate ODEs
    # let u(x, t) = X(x) * T(t)


    # then it becomes
    cxx * X'' * T + cx * X' * T + ct * X * T' + ctt * X * T'' = 0
    (cxx * X''+ cx * X')/X  = (-ct * T' - ctt * T'')/T = -lambda_

    ode_x = cxx * X'' + cx * X' + lambda_ * X = 0
    ode_t = -ctt * T'' + -ct * T' + lambda_ * T = 0

    ## STEP 2: solve the seperate ODEs
    and we solve these two ODEs using the table.
    we solve ode_x with BCs using the table that does that shit directly.
    that table should also return possible lambda values.
    then we solve ode_t with the possible lambda values, seperating into cases where D is positive, negative, or
    zero if applicable.

    ## STEP 3: combine the solutions of the ODEs to make the final form of the solution.
    finally, now we have both solutions! all we need to do is to
    all possible lambda/n
    and like make a linear combination of them with summation class by case (due to homogeniety of the pde)
    """
    cxx, cx, ct, ctt = pde

    X, T = sp.symbols("X T", function=True)  # R->R functions

    # lambda_ can be everything except 0
    lambda_ = sp.symbols("lambda", latex="\lambda")  # its latex fn should auto escape tbh

    # cxx * X'' * T + cx * X' * T + ct * X * T' + ctt * X * T'' = 0
    # (cxx * X''+ cx * X')/X  = (-ct * T' - ctt * T'')/T = -lambda_
    # ode_x = cxx * X'' + cx * X' + lambda_ * X = 0
    # ode_t = -ctt * T'' + -ct * T' + lambda_ * T = 0

    ode_x = ODE(cxx, cx, lambda_)
    ode_t = ODE(-ctt, -ct, lambda_)

    # solve ODE x and then plug in boundary conditions
    ode_x.discriminant.symbols
    case_pos = table(ode_x.c2, ode_x.c1, ode_x.c0, "positive")
    case_zero = table(ode_x.c2, ode_x.c1, ode_x.c0, "zero")
    case_neg = table(ode_x.c2, ode_x.c1, ode_x.c0, "negative")

    # somehow, we have to be able to tell that D = -4 * lambda
    # can be negative or positive but not zero because lambda can't be 0

    breakpoint()

    # solve ODE t with the possible lambda values


def test_wave_eqn():
    """This is the one from the exam."""
    pde = PDE(1, 0, -4, -1)
    bcs = [
        BoundaryCondition(0, 0, 0, 1),
        BoundaryCondition(0, sp.pi, 0, 1),
    ]
    ans = solve_pde_with_seperation_of_variables(pde, bcs)


"""
What if I were to make a function that does seperation of variables?
inputs:
- pde
- 2 boundary conditions

output:
- expr representing the solution



how to represent a pde?
coeff of u_xx, u_x, u, u_tt, u_t
assume homogenous
no u_xt term allowed


pde: cxx * u_xx + cx * u_x + ct * u_t + ctt * u_tt = 0

how to represent BCs?
    

class Summation(Expr):
    var: Symbol
    start_index: any int or inf
    end_index: any int or inf
    inner: Expr

    def latex(self):
        def bracket(e: Expr) -> str:
            if isinstance(e, Sum):
                return bracketfy(e)
            return e.latex()

        return \\sum_{self.var.latex()=self.start_index.latex()}^wrap(self.end_index.latex()) bracket(self.inner)
    
    # def eval(self):
    #     return sum(self.inner.subs({self.var: i}) for i in range(self.start_index, self.end_index + 1))

"""
