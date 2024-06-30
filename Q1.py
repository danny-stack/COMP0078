import numpy as np
import matplotlib.pyplot as plt


# Define a function to calculate the weights
def get_weight(X, y):
    w = np.matmul(np.linalg.pinv(X), y)
    return w

# Define a function to create a basis matrix for polynomial regression
def basis(X, k):
    Phi = np.zeros((len(X), k))
    for i in range(k):
        Phi[:, i] = X ** i
    return Phi

'''
Define a function to create a new sin basis matrix for Q3
Replace the original basis function for Q3
'''
# def basis(X, k):
#     Phi = np.zeros((len(X), k))
#     for i in range(1, k + 1):
#         Phi[:, i - 1] = np.sin(i * np.pi * X)
#     return Phi

# Question 1
def plot_reg():

    X = np.array([1, 2, 3, 4])
    y = np.array([3, 2, 0, 5])

    plt.figure()

    # Loop through different polynomial degrees (k) for regression
    for i in range(1, 5):
        # Generate the basis anc calculate the weights
        Phi = basis(X, i)
        w = get_weight(Phi, y)

        # Calculate the Mean Squared Error
        mse = np.mean((np.matmul(Phi, w) - y) ** 2)
        print('When k = %d, MSE = %.6f' % (i, mse))
        print('When k = %d, w =' % i, w)


        # Plot the regression line
        linex = np.linspace(0, 5, 100)
        linePhi = basis(linex, i)
        liney = np.matmul(linePhi, w)
        plt.plot(linex, liney, label='k = %d' % i)

    # Plot the original data points
    plt.plot(X, y, 'o', label='data points')

    plt.axis([0, 5, -5, 8])
    plt.legend()
    plt.grid(True)
    plt.title("Regression Fitting")
    plt.show()


plot_reg()
