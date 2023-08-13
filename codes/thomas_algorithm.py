def thomas_algorithm(a, b, c, d):
    n = len(d)
    c_new = [c[0] / b[0]]
    d_new = [d[0] / b[0]]

    print("Initial c_new:", c_new)
    print("Initial d_new:", d_new)

    for i in range(1, n):
        m = 1.0 / (b[i] - a[i] * c_new[i - 1])
        c_new.append(c[i] * m)
        d_new.append((d[i] - a[i] * d_new[i - 1]) * m)

        print(f"Iteration {i}:")
        print("Intermediate c_new:", c_new)
        print("Intermediate d_new:", d_new)

    x = [0] * n
    x[n - 1] = d_new[n - 1]

    print("\nBack substitution:")

    for i in range(n - 2, -1, -1):
        x[i] = d_new[i] - c_new[i] * x[i + 1]
        print(f"Step {i}: x[{i}] =", x[i])

    return x

# Example usage
a = [2, 1, 0, 0]  # Lower diagonal
b = [4, 4, 4, 4]  # Main diagonal
c = [0, 0, 1, 2]  # Upper diagonal
d = [6, 5, 5, 6]  # Right-hand side

solution = thomas_algorithm(a, b, c, d)
print("\nFinal Solution:", solution)
