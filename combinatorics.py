import functools
import math
from typing import List


def generate_permutations(i: int, n: int) -> List[List[int]]:
    """
    generate all permutations of i numbers x_i that sum up to n, where 0<=x_i<=n 

    i: number of terms
    n: sum of terms

    This function uses a recursive approach to explore each possible value for x_i from 0 to
    n, and then recursively calls itself to handle the remaining numbers and the reduced sum
    """
    def generate(current_permutation: List[int], current_sum: int, remaining_numbers: int):
        if remaining_numbers == 0:
            if current_sum == n:
                results.append(current_permutation[:])
            return
        
        start = 0
        end = min(n, n - current_sum)  # Maximum value that can be added without exceeding the sum n
        for value in range(start, end + 1):
            current_permutation.append(value)
            generate(current_permutation, current_sum + value, remaining_numbers - 1)
            current_permutation.pop()

    results = []
    generate([], 0, i)
    return results

def multinomial_coefficient(xs: List[int], n: int) -> int:
    if sum(xs) > n:
        return "Invalid input: sum of xs must not exceed n."
    
    # Calculate factorial of n and the individual components x1, x2 ..., and the remaining items
    numerator = math.factorial(n)
    
    # now, calculate the denominator
    mul = lambda x, y: x * y
    add = lambda x, y: x + y
    xs_factorials = [math.factorial(x) for x in xs]
    xs_factorials_product = functools.reduce(mul, xs_factorials)
    xs_sum = functools.reduce(add, xs)
    denominator = math.factorial(n - xs_sum) * xs_factorials_product
    
    return numerator // denominator  # Integer division for exact result
