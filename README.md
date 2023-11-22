## simpy

*A worse version of [sympy](https://www.sympy.org)*

Current version can do algebraic simplifications and integrate polynomial functions. For example, these integrals Laura had to do for homework:

$$
\begin{aligned}
I_1 &= \int_{5}^{6} \frac{x}{90} \cdot \frac{(x-5)^2}{350} \, dx &&= \frac{23}{378000} \\
I_2 &= \int_{6}^{15} \left(\frac{1}{15} - \frac{1}{360} \cdot (x-6)\right) \cdot \frac{(x-5)^2}{350} \, dx &&= \frac{2589}{56000} \\
I_3 &= \int_{15}^{30} \left(\frac{1}{15} - \frac{1}{360} \cdot (x-6)\right) \cdot \left(1 - \frac{(40-x)^2}{875}\right) \, dx &&= \frac{37}{224}
\end{aligned}
$$