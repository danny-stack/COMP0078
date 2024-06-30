import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

# data preparation
data = pd.read_csv('./Boston-filtered.csv')
label_name = 'MEDV'
# print(data.head())

X = data.drop([label_name], axis=1).values  # training data
y = data[label_name].values  # label


# split the data into 1/3 testing and 2/3 training
def data_split(X,y):
    test_size = int(len(X) / 3)
    random_seed = np.random.randint(1,100)

    np.random.seed(random_seed)
    indices = np.random.permutation(len(X))

    X_test = X[indices[:test_size]]
    y_test = y[indices[:test_size]]
    X_train = X[indices[test_size:]]
    y_train = y[indices[test_size:]]
    return X_train, X_test, y_train, y_test


# mean square error
def MSE(y_pred, y):
    return np.mean((y - y_pred) ** 2)


# return parameter theta
def get_w(X, y):
    # X_bias = np.column_stack((np.ones(len(X)), X))
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta

runs = 20

train_errors = []
test_errors = []
# Q4 (a)
for i in range(runs):
    # get training and testing dataset
    X_train, X_test, y_train, y_test = data_split(X, y)

    # fit the data with a constant function
    X_b = np.ones((X_train.shape[0], 1))
    X_b_test = np.ones((X_test.shape[0], 1))
    theta = get_w(X_b, y_train)

    y_pred_train = X_b @ theta
    y_pred_test = X_b_test @ theta

    # Calculate MSE for both traning and testing dataset
    MSE_train = MSE(y_pred_train, y_train)
    MSE_test = MSE(y_pred_test, y_test)

    train_errors.append(MSE_train)
    test_errors.append(MSE_test)


print(f"Regression Function: y = {theta[0]:.4f}")
print(f"Train Errors: {train_errors}")
print(f"Test Errors: {test_errors}")
print(f"Average Train Errors: {np.mean(train_errors)}")
print(f"Average Test Errors: {np.mean(test_errors)}")
print(f"Average Train Errors STD: {np.std(train_errors)}")
print(f"Average Test Errors STD: {np.std(test_errors)}")


# Q4 (c)
N = X.shape[0]
mse_train_average = []
mse_test_average = []
mse_train_average_std = []
mse_test_average_std = []
X_train, X_test, y_train, y_test = data_split(X, y)
for k in range(X_train.shape[1]):

    # X_train, X_test, y_train, y_test = data_split(X, y)
    mse_train_attr = []
    mse_test_attr = []

    for i in range(runs):
        # get training and testing dataset
        X_train, X_test, y_train, y_test = data_split(X, y)

        # perform linear regression with single attribute
        X_train_attr = X_train[:, k]
        X_test_attr = X_test[:, k]

        # incorporating a bias term
        X_train_bias = np.column_stack((X_train_attr, np.ones(len(X_train_attr))))

        theta = get_w(X_train_bias, y_train)
        y_pred_train = np.dot(X_train_bias, theta)
        # Calculate MSE for training dataset
        mse_train = MSE(y_train, y_pred_train)
        mse_train_attr.append(mse_train)

        # incorporating a bias term and calculating MSE for testing dataset
        X_test_bias = np.column_stack((X_test_attr, np.ones(len(X_test_attr))))
        y_pred_test = np.dot(X_test_bias, theta)
        mse_test = MSE(y_test, y_pred_test)
        mse_test_attr.append(mse_test)

    mse_train_average.append(np.mean(mse_train_attr))
    mse_test_average.append(np.mean(mse_test_attr))
    mse_train_average_std.append(np.std(mse_train_attr))
    mse_test_average_std.append(np.std(mse_test_attr))

for i, mse in enumerate(mse_train_average):
    print(f'Attribute {i}: Train MSE = {mse}, Test MSE = {mse_test_average[i]}')
    print(f'Attribute {i}: Train MSE STD  = {mse_train_average_std[i]}, Test MSE = {mse_test_average_std[i]}')

# plot average train and test MSE with respect to each attribute
features = range(1,13)
plt.figure(figsize=(10, 5))
plt.plot(features, mse_train_average, label='Train Mean', marker='o')
plt.plot(features, mse_test_average, label='Test Mean', marker='o')
plt.title('Mean of Train and Test MSE')
plt.xlabel('Features')
plt.ylabel('Mean MSE')
plt.legend()
plt.grid()
plt.show()


# Q4 (d)
mse_train_results = []
mse_test_results = []

for _ in range(20):
    # get training and testing dataset
    X_train, X_test, y_train, y_test = data_split(X, y)

    # incorporating a bias term  and perform linear regression with all attributes
    X_train_bias_d = np.column_stack((X_train, np.ones(len(X_train))))
    theta = get_w(X_train_bias_d, y_train)

    # Calculate MSE for training dataset
    y_pred_train_d = np.dot(X_train_bias_d, theta)
    mse_train_d = MSE(y_train, y_pred_train_d)
    mse_train_results.append(mse_train_d)

    # Calculate MSE for testing dataset
    X_test_bias_d = np.column_stack((X_test,np.ones(len(X_test))))
    y_pred_test = np.dot(X_test_bias_d, theta)
    mse_test = MSE(y_test, y_pred_test)
    mse_test_results.append(mse_test)

average_mse_train_d = np.mean(mse_train_results)
average_mse_test_d = np.mean(mse_test_results)
average_mse_train_d_std = np.std(mse_train_results)
average_mse_test_d_std = np.std(mse_test_results)

print(f'Average Train MSE = {average_mse_train_d}, Average Test MSE = {average_mse_test_d}')
print(f'Average Train STD = {average_mse_train_d_std}, Average Test STD = {average_mse_test_d_std}')


# Q5
# Normal Kernel with no vectorization
# def gaussian_kernel(X1, X2, sigma):
#     n1 = X1.shape[0]
#     n2 = X2.shape[0]
#     K = np.zeros((n1, n2))
#     for i in range(n1):
#         for j in range(n2):
#             diff = X1[i, :] - X2[j, :]
#             K[i, j] = np.exp(-np.dot(diff, diff) / (2 * sigma**2))
#     return K
#


# Kernel after vectorization
def gaussian_kernel(X1, X2, sigma):
    diff = X1[:, np.newaxis] - X2
    norm = np.sum(diff ** 2, axis=-1)
    K = np.exp(-norm / (2 * sigma ** 2))
    return K

# Q5 a
def ridge_regression(X, y, gamma_values, sigma_values):
    best_mse = float('inf')
    best_gamma = None
    best_sigma = None

    # store mse for each pair of gamma and sigma values
    mse_values_grid = np.zeros((len(gamma_values), len(sigma_values)))

    # perform five-fold cross-validation
    n = len(X)
    kfolds = 5
    kf = np.array_split(np.random.permutation(n), kfolds)

    for i, gamma in enumerate(gamma_values):
        for j, sigma in enumerate(sigma_values):
            mse_values = []
            for fold in range(kfolds):
                # perform five-fold cross-validation on each pair of gamma and sigma value
                train_indices = np.concatenate([kf[i] for i in range(kfolds) if i != fold])
                val_indices = kf[fold]

                X_train, X_val = X[train_indices], X[val_indices]
                y_train, y_val = y[train_indices], y[val_indices]

                # calculate kernel and alpha
                K_train = gaussian_kernel(X_train, X_train, sigma)
                K_val = gaussian_kernel(X_val, X_train, sigma)
                alpha = np.matmul(np.linalg.pinv(K_train + gamma * len(X_train) * np.identity(len(train_indices))), y_train)

                # predict the validation y
                y_val_pred = np.dot(K_val, alpha)

                mse = MSE(y_val,y_val_pred)
                mse_values.append(mse)

            mse_values_grid[i, j] = np.mean(mse_values)

            # find and update the best pair of gamma and sigma value
            average_mse = np.mean(mse_values)
            if average_mse < best_mse:
                best_mse = average_mse
                best_gamma = gamma
                best_sigma = sigma

    return best_gamma, best_sigma, best_mse, mse_values_grid


gamma = 2 ** np.arange(-40,-25,1.0)
sigma = 2 ** np.arange(7,13.5,0.5)

train_MSE_avarage = []
test_MSE_average = []

# perform ridge regression for 20 runs for Q5 d
for _ in range(20):

    X_train, X_test, y_train, y_test = data_split(X, y)

    best_gamma, best_sigma, best_mse, mse_values_grid = ridge_regression(X_train, y_train, gamma, sigma)

    # calculate K train and alpha
    K_train = gaussian_kernel(X_train, X_train, best_sigma)
    # alpha = np.linalg.solve(K_train + best_gamma * np.identity(len(X_train)), y_train)
    alpha = np.matmul(np.linalg.pinv(K_train + best_gamma * len(X_train) * np.identity(len(X_train))), y_train)

    # do prediction on training set
    y_train_pred = np.dot(K_train, alpha)
    train_mse = MSE(y_train, y_train_pred)

    # do prediction on testing set
    K_test = gaussian_kernel(X_test, X_train, best_sigma)
    y_test_pred = np.dot(K_test, alpha)
    test_mse = MSE(y_test, y_test_pred)


    train_MSE_avarage.append(train_mse)
    test_MSE_average.append(test_mse)

# 5 c and d
print(f'Best gamma: {best_gamma}')
print(f'Best sigma: {best_sigma}')
print(f'Test MSE with best parameters: {test_mse}')
print(f'Train MSE with best parameters: {train_mse}')
print(f'Test average MSE with best parameters: {np.mean(train_MSE_avarage)}')
print(f'Train average MSE with best parameters: {np.mean(test_MSE_average)}')
print(f'Test average MSE STD with best parameters: {np.std(train_MSE_avarage)}')
print(f'Train average MSE STD with best parameters: {np.std(test_MSE_average)}')

# 5 b
# plot the cross-validation error as a function of gamma and sigma.
gamma_coe = np.arange(-40,-25,1.0)
sigma_coe = np.arange(7,13.5,0.5)
gamma_grid, sigma_grid = np.meshgrid(sigma_coe, gamma_coe)
fig = plt.figure(figsize=(20, 15))
ax = plt.axes(projection='3d')
ax.set_xlabel('Sigma coefficients', fontsize=17)
ax.set_ylabel('Gamma coefficients', fontsize=17)
ax.set_zlabel('cross-validation errors', fontsize=17)
# print(gamma_values_grid.shape)
# print(sigma_values_grid.shape)
# print(mse_values_grid.shape)
surface = ax.plot_surface(gamma_grid, sigma_grid, mse_values_grid,cmap=cm.coolwarm)
fig.colorbar(surface)
plt.title('Cross-Validation Error')
plt.show()






