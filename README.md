## simpy

*A worse version of [sympy](https://www.sympy.org)*

Current version can do algebraic simplifications and integrate polynomial functions. For example, these integrals Laura had to do for homework:

$$
\begin{aligned}
\frac{23}{378000} &= \int_{5}^{6} \frac{x}{90} \cdot \frac{(x-5)^2}{350} \, dx \\\\
\frac{2589}{56000} &= \int_{6}^{15} \left(\frac{1}{15} - \frac{1}{360} \cdot (x-6)\right) \cdot \frac{(x-5)^2}{350} \, dx \\\\
\frac{37}{224} &= \int_{15}^{30} \left(\frac{1}{15} - \frac{1}{360} \cdot (x-6)\right) \cdot \left(1 - \frac{(40-x)^2}{875}\right) \, dx
\end{aligned}
$$

Can be done like so:

```python
import simpy as sp
from fractions import Fraction as F

x = sp.symbols("x")

I1 = sp.integrate((x/90 * (x-5)**2 / 350), (x, 5, 6))
I2 = sp.integrate((F(1, 15) - F(1, 360) * (x-6))*(x-5)**2 / 350, (x, 6, 15))
I3 = sp.integrate((F(1, 15) - F(1, 360) * (x-6))*(1 - (40-x)**2/875), (x, 15, 30))
```