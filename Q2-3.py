import numpy as np
import matplotlib.pyplot as plt
from Q1 import basis, get_weight


'''
For Q3, we just change the original basis function in Q2 and the rest routines are 
kept the same. We just repeat the Q2 b-d using the new basis function for Q3.
'''


# Calculate Mean Squared Error (MSE) for training data
def mse_train(X, y, k):
    # Create a basis phi using X and degree k
    Phi = basis(X, k)

    # Calculate the weight
    w = get_weight(Phi, y)

    # Calculate the Mean Squared Error (MSE) for training data
    mse = np.mean((np.matmul(Phi, w) - y) ** 2)
    return mse

# Calculate Mean Squared Error (MSE) for testing data
def mse_test(X, y, x_test, y_test, k):
    # Create a basis phi using X and degree k
    Phi = basis(X, k)
    # Calculate the weight
    w = get_weight(Phi, y)

    # Create a basis phi for testing using X_test and degree k
    Phi_test = basis(x_test, k)
    # Calculate the Mean Squared Error (MSE) for testing data
    MSE_test = np.mean((np.matmul(Phi_test, w) - y_test) ** 2)
    return MSE_test

# Generate a dataset with random noise added to a sine function
def gen_dataset(x, var, k):
    random = np.random.normal(0, var, k)
    y = (np.sin(2 * np.pi * x)) ** 2 + random
    return y

# Generate 30 training points
number = 30
sigma = 0.07
X = np.random.uniform(0, 1, number)
y = gen_dataset(X.reshape(-1), sigma, number)

# Generate sin function
sinx = np.linspace(0, 2, 100)
siny = np.sin(2 * np.pi * sinx) ** 2


# Q2 a i.
# Plot sin function and random data points
def plot_sin():
    plt.plot(sinx, siny, c='black')
    plt.scatter(X, y, c='black', s=1)
    plt.xlabel('x')
    plt.ylabel('$S_{0.07,30}$')
    plt.xlim([0, 1])
    plt.title('Function Plot with Data Points')
    plt.show()

# Q2 a ii.
def data_fit():
    # Fit data using different k values
    for k in [2, 5, 10, 14, 18]:
        Phi = basis(X, k)
        w = get_weight(Phi, y)

        # Plot the regression line
        linex = np.linspace(0, 1, 100)
        linePhi = basis(linex, k)
        liney = np.matmul(linePhi, w)
        plt.plot(linex, liney, label=f'k = {k}')

    # Plot the original data points
    plt.plot(X.reshape(-1), y, 'o', label='data points')
    plt.ylim([0, 1.2])
    plt.legend()
    plt.grid(True)
    plt.title("Regression Fitting")
    plt.show()


# Q2 b
# Generate new training data points
X_train = np.random.uniform(0, 1, number)
y_train = gen_dataset(X.reshape(-1), sigma, number)

# Calculate MSE for all different value of k
k = list(range(1, 19))
log_MSE = []
for dim in k:
    log_MSE.append(np.log(mse_train(X_train, y_train, dim)))

# plot the training error versus the polynomial dimension
def log_train():
    plt.figure()
    plt.plot(k, log_MSE)
    plt.grid()
    plt.xticks(np.arange(1, len(k) + 1))
    plt.xlabel('k')
    plt.ylabel('$\log (te_k(S))$')
    plt.title('log of Training Error')

    # New title for sin basis plot for Q3 b
    # plt.title('log of Training Error Using New Basis $\sin(k\pi x)$')
    plt.show()


# Q2 c
# Generate new testing data points
test_number = 1000
X_test = np.random.uniform(0, 1, test_number)
y_test = gen_dataset(X_test, sigma, test_number)

# Calculate testing MSE for all different value of k
log_MSE_test = []
for dim in k:
    log_MSE_test.append(np.log(mse_test(X_train, y_train, X_test, y_test, dim)))

# plot the testing error versus the polynomial dimension
def log_test():
    plt.figure()
    plt.plot(k, log_MSE_test)
    plt.grid()
    plt.xticks(np.arange(1, len(k) + 1))
    plt.xlabel('k')
    plt.ylabel('$\log (tse_k(S,T))$')
    plt.title('log of Testing Error')

    # New title for sin basis plot for Q3 c
    # plt.title('log of Testing Error Using New Basis $\sin(k\pi x)$')
    plt.show()

# Q2 d
'''
Calculate and plot average training and testing errors 
for specific k values after 100 loops
'''
log_avg_train = []
log_avg_test = []
def avg_error():
    for dim in k:
        MSE_sum_train = 0
        MSE_sum_test = 0
        for _ in range(100):
            x_train = np.random.uniform(0, 1, 30)
            x_test = np.random.uniform(0, 1, 1000)
            y_train = gen_dataset(x_train, 0.07, 30)
            y_test = gen_dataset(x_test, 0.07, 1000)
            MSE_sum_train += mse_train(x_train, y_train, dim)
            MSE_sum_test += mse_test(x_train, y_train, X_test, y_test, dim)
        log_avg_train.append(np.log(MSE_sum_train / 100))
        log_avg_test.append(np.log(MSE_sum_test / 100))

    plt.figure()
    plt.plot(k, log_avg_train)
    plt.grid()
    plt.xticks(np.arange(1, len(k) + 1))
    plt.xlabel('k')
    plt.ylabel('$\log (te_k(S))$')
    plt.title('log of Average Training Error')
    
    # New title for sin basis plot for Q3 d
    # plt.title('log of Average Training Error Using New Basis $\sin(k\pi x)$')
    plt.show()

    plt.figure()
    plt.plot(k, log_avg_test)
    plt.grid()
    plt.xticks(np.arange(1, len(k) + 1))
    plt.xlabel('k')
    plt.ylabel('$\log (tse_k(S,T))$')
    plt.title('log of Average Testing Error')

    # New title for sin basis plot for Q3 d
    # plt.title('log of Average Testing Error Using New Basis $\sin(k\pi x)$')
    plt.show()

#Q2 a i
plot_sin()
#Q2 a ii
data_fit()
#Q2 b
log_train()
#Q2 c
log_test()
#Q2 d
avg_error()
