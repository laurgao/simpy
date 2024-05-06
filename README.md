# simpy

_A worse version of [sympy](https://www.sympy.org)_

Current version can do algebraic & trigonometric simplifications, perform any differentiation, and integrate most of AP calc functions including polynomials, rational functions, trig functions, logs, and exponentials.

## Quick start

Real installation instructions are coming soon! For now, just clone the repo and `pip install requirements.txt`. You can run `pytest .` in the root directory to run my tests. Look at `test_integrals.py` and `test_khan_academy_integrals.py` to see some sample integrals we can do :)

For example, these polynomial integrals Laura had to do for homework:

$$
\begin{aligned}
\frac{23}{378000} &= \int_{5}^{6} \frac{x}{90} \cdot \frac{(x-5)^2}{350} dx \\\\
\frac{2589}{56000} &= \int_{6}^{15} \left(\frac{1}{15} - \frac{1}{360} \cdot (x-6)\right) \cdot \frac{(x-5)^2}{350} dx \\\\
\frac{37}{224} &= \int_{15}^{30} \left(\frac{1}{15} - \frac{1}{360} \cdot (x-6)\right) \cdot \left(1 - \frac{(40-x)^2}{875}\right) dx
\end{aligned}
$$

Can be done like so:

```python
from src import simpy as sp
from fractions import Fraction as F

x = sp.symbols("x")

I1 = sp.integrate((x/90 * (x-5)**2 / 350), (x, 5, 6))
I2 = sp.integrate((F(1, 15) - F(1, 360) * (x-6))*(x-5)**2 / 350, (x, 6, 15))
I3 = sp.integrate((F(1, 15) - F(1, 360) * (x-6))*(1 - (40-x)**2/875), (x, 15, 30))
print(I1, I2, I3)
```

Note: we don't support floats right now so please use fractions!

## Please make issues!

This project is actively undergoing development; please let me know about any bugs you encounter by making a github issue! If there's an integral that we currently can't solve, create an issue and tag "new integral."
