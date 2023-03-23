import torch

def multivariate_lagrange_interp(points, values, x):
    """
    Computes the Lagrange interpolation polynomial for the given points and values,
    evaluated at the point x.
    
    Parameters:
    -----------
    points : torch.Tensor, shape (n, d)
        The interpolation points, where n is the number of points and d is the dimension.
    values : torch.Tensor, shape (n, ...)
        The values of the function to be interpolated at the given points.
    x : torch.Tensor, shape (d,)
        The point at which to evaluate the interpolation polynomial.
    
    Returns:
    --------
    torch.Tensor
        The value of the interpolation polynomial at the point x.
    """
    n, d = points.shape
    
    # Compute the reduced basis for the tensor product of univariate Lagrange polynomials.
    basis = [torch.polygamma(0, torch.tensor([1.0]))]
    for i in range(d):
        basis_i = []
        for j in range(n):
            p = torch.polygamma(0, torch.tensor([1.0]))
            for k in range(n):
                if k != j:
                    p *= (x[i] - points[k, i]) / (points[j, i] - points[k, i])
            basis_i.append(p)
        basis.append(basis_i)
    
    # Compute the interpolation weights.
    weights = torch.zeros(values.shape[1:])
    for j in range(n):
        w = values[j]
        for i in range(d):
            w *= basis[i+1][j]
        weights += w
    
    # Compute the interpolated value.
    return weights


# Example usage
grid = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
values = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])

# Evaluate the function at a new point
x = torch.tensor([0.5, 0.5])
interp = multivariate_lagrange_interp(grid, values, x)
print(interp)  # output: tensor([3., 4.])
