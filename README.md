# simpy

_A worse version of [sympy](https://www.sympy.org)_

Current version can do algebraic & trigonometric simplifications, perform differentiation, and can perform ALMOST ALL AP calc integrals including polynomials, rational functions, trig functions, logs, exponentials, and combinations of them.

## Quick start

Real installation instructions are coming soon! For now, just clone the repo and `pip install requirements.txt`. You can run `pytest .` in the root directory to run my tests. Look at `test_integrals.py` and `test_khan_academy_integrals.py` to see some sample integrals we can do :)

For example, these integrals:

$$
\begin{aligned}
&\int \tan^{-1}(x) \,dx = -\frac{\ln \left( \left| x^2 + 1 \right| \right)}2 + x \cdot \tan^{-1} (x)
\\
&\int \frac{x - 5}{-2x + 2} \,dx = 2 \cdot \ln(|-x + 1|) - \frac{x}{2}
\\
&\int_{0}^{\pi/6} \sec\left(2x\right) \cdot \tan\left(2x\right) \,dx = \frac12
\\
&\int_{15}^{30} \left(\frac{1}{15} - \frac{1}{360} \cdot (x-6)\right) \cdot \left(1 - \frac{(40-x)^2}{875}\right) \,dx = \frac{37}{224}
\end{aligned}
$$

Can be done like so:

```python
from src import simpy as sp
from fractions import Fraction as F

x = sp.symbols("x")

ans1 = sp.integrate(sp.atan(x))
ans2 = sp.integrate((x - 5)/(-2*x + 2))
ans3 = sp.integrate(sp.sec(2*x)*sp.tan(2*x), (0, sp.pi/6))
ans4 = sp.integrate((F(1, 15) - F(1, 360) * (x-6))*(1 - (40-x)**2/875), (x, 15, 30))

print(ans1, ans2, ans3, ans4, sep="\n")
```

## Please make issues!

This project is actively undergoing development; please let me know about any bugs you encounter by making a github issue! If there's an integral that we currently can't solve, create an issue and tag "new integral."

# Code style

This project uses black and isort with line width of 120.

Run `make style` to format accordingly. And `make check-style` to check your edits meet the formatting guidelines.

If you use vscode, you can add this to `settings.json` to automatically format when you save.

```json
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": "always",
            "source.fixAll": "always"
        },
        "editor.rulers": [120],
    },
    "black-formatter.args": ["--line-length", "120"],
    "isort.args": ["--profile", "black", "--line-length", "120"],
```
