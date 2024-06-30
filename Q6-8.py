import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# set a random seed to generate random data points
seed = np.random.randint(0, 100)
np.random.seed(seed)


# generate random centers and their labels
def sampling():
    centers = np.random.rand(100, 2)
    labels = np.random.randint(0, 2, 100)
    return centers, labels


# KNN algorithm
def Knn_algorithm(centers, labels, test_points, v):
    majorities = []
    for i in test_points:
        # Calculate the Euclidean distance between x and all centers
        distances = np.linalg.norm(centers - i, ord=2, axis=1)

        # find the index of closest v centers
        nearest_indices = np.argsort(distances)[:v]

        # get the labels of these centers
        nearest_labels = labels[nearest_indices]

        # find the majority of the labels
        if np.mean(nearest_labels[:v]) > 0.5:
            majority_label = 1
        elif np.mean(nearest_labels[:v]) < 0.5:
            majority_label = 0
        else:
            # resolve corner cases by generating either the label or prediction uniformly at random
            majority_label = np.random.randint(0, 2)
        majorities.append(majority_label)

    return np.array(majorities)


# knn visualisation
def visualisation(x_data, y_data, k):
    # create a grid to draw decision boundary
    xx, yy = np.meshgrid(np.arange(0, 1.01, 0.005), np.arange(0, 1.01, 0.005))
    # print(xx.shape)
    # print(yy.shape)

    # concatenate grid points to matrix
    grid_points = np.c_[xx.reshape(-1), yy.reshape(-1)]

    # calculate the classification result using knn
    predictions = Knn_algorithm(x_data, y_data, grid_points, k).reshape(xx.shape[0], xx.shape[1])
    # print(predictions.shape)

    # draw datapoints and decision boundary
    plt.contourf(xx, yy, predictions, alpha=0.5, cmap='gist_earth')
    plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data, cmap='rainbow')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Sample Data')
    plt.show()

# Q6
# Produce a visualisation of an hS,v similar to the figure
centers, labels = sampling()
visualisation(centers, labels, 3)


# Generate data points based on probability distribution
def generate_data(centers, labels, N, K):
    gen_centers = np.random.rand(N, 2)
    gen_labels = []
    # generate n centers according to the probability
    for n in range(N):
        coin_flip = np.random.choice([0, 1], p=[0.2, 0.8])
        if coin_flip == 0:
            gen_labels.append(np.random.randint(0, 2))
        else:
            gen_labels.append(Knn_algorithm(centers, labels, gen_centers[n].reshape(1, -1), K).item())

    return gen_centers, np.array(gen_labels)


# Q7
# Produce a visualisation using Protocol A
def protocol_a():
    error_list = []
    # For each k
    for k in tqdm(range(1, 50), desc='k_loop'):
        total_error = 0
        # Do 100 runs
        for _ in tqdm(range(100), desc='n_loop'):
            # generate training and testing dataset
            x, y = sampling()
            x_train, y_train = generate_data(x, y, 4000, 3)
            x_test, y_test = generate_data(x, y, 1000, 3)

            # run knn algorithm
            predictions = Knn_algorithm(x_train, y_train, x_test, k)

            # estimate generalisation error
            errors = sum(1 for i in range(len(predictions)) if predictions[i] != y_test[i])
            total_error += errors / len(predictions)

        # generalization error is the mean of these 100 generalization errors
        avg_error = total_error / 100
        error_list.append(avg_error)

    return error_list


error = protocol_a()

# visualize the result
plt.plot(range(1, 50), error)
plt.title('Visualisation of Protocol A')
plt.xlabel('k - Nearest Neighbours')
plt.ylabel('Estimated Generalisation Error')
plt.savefig("Q7.png")
plt.show()


# Q8
# Produce a visualisation using Protocol b
numbers = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]

def protocol_B():
    Optimal_k = []
    # For each m
    for m in tqdm(numbers, desc='train_size loop'):
        optimal_K = 0
        # Do 100 runs
        for i in tqdm(range(100), desc='runs loop'):

            # generate training and testing dataset
            x, y = sampling()
            x_train, y_train = generate_data(x, y, m, 3)
            x_test, y_test = generate_data(x, y, 1000, 3)

            k_errors = []
            # For each k
            for k in tqdm(range(1, 50), desc='k_loop'):
                # run knn algorithm and estimate generalisation error
                predictions = Knn_algorithm(x_train, y_train, x_test, k)
                error = sum(1 for j in range(len(predictions)) if predictions[j] != y_test[j]) / len(predictions)
                k_errors.append(error)

            # The estimated optimal k is  the k with minimal estimated generalisation error
            index = np.argmin(k_errors)
            optimal_K += (index + 1)

        # The estimated optimal k is  the mean of these 100 run optimal 'k's.
        average_optimal_K = optimal_K / 100
        Optimal_k.append(average_optimal_K)

    return Optimal_k


optimal_k = protocol_B()

# visualize the result
plt.plot(numbers, optimal_k)
plt.title('Visualisation of Protocol B')
plt.ylabel('Optimal k-value')
plt.xlabel('Number of training points')
plt.savefig("Q8.png")
plt.show()
