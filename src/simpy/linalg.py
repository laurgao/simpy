import numpy as np


def invert(matrix: np.ndarray):
    """
    Input: matrix of Exprs, so numpy can't natively invert it >:(
    Output: inverted matrix of Exprs. Returns None if the matrix is singular.
    """
    assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]
    n = matrix.shape[0]
    if n != 2:
        raise NotImplementedError("Only 2x2 matrices are supported. There's no need to expand this until I allow partial fractions for trinomials.+")
    
    a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    det = (a * d - b * c).simplify()
    if det == 0:
        return None

    return np.array([[d / det, -b / det], [-c / det, a / det]])


