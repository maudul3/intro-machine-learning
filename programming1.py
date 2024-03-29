from keras.datasets import mnist
from random import uniform
import numpy as np
import datetime
import matplotlib.pyplot as plt

OUTPUTS = 10


def sigmoid(z):
    """Sigmoid function which can act on entire numpy arrays

    Inputs:
        z (np.array[float]): input values for sigmoid function

    Outputs:
        np.array(float): output values of sigmoid function
    """
    return 1 / (1 + np.exp(-z))


def confusion_matrix(t, y):
    """Function used to generate confusion matrix on true/test values

    Form of matrix:
     -------------------------------------
    |  True Positive    | False Positive  |
    --------------------------------------
    |  False Negative   |  True Negative  |
     -------------------------------------

    Inputs:
        t (np.array[int]): true value
        y (np.array[int]): test value

    Output:
        np.array[int]: confusion matrix
    """
    y = [np.max(np.where(row == 1), initial=-1) for row in y]
    t = [np.max(np.where(row == 1), initial=-1) for row in t]

    confusion = np.zeros((10, 10))

    for t_val, y_val in zip(t, y):
        confusion[t_val][y_val] += 1

    return confusion


def run_experiment(
    train_x, test_x, train_y_bin, test_y_bin, learning_rate, momentum, hidden_units
):
    """Run an experiment in which we train and test a neural network
    for use identifying the digits in the MNIST dataset

    Inputs:
        train_x (np.array[float]): training array of images from MNIST,
            each containing 785 normalized pixels (0 - 1)
        test_x (np.array[float]): testing array of images from MNIST,
            each containing 785 normalized pixels (0 - 1)
        train_y_bin (np.array[int]): training array indicating
            which digit (0 - 9) is represented in the corresponding image.
            Binary notation used to identify the digit.
        test_y_bin (np.array[float]): testing array indicating
            which digit (0 - 9) is represented in the corresponding image.
            Binary notation used to identify the digit.
        learning_rate (float): determines how much the model is improved at
            each update
        momentum (float): determines how much previous updates to the model
            affect the next update
        hidden_units (int): number of hidden units in the neural network
    """
    # Modify true values to be 0.9 and 0.1 instead of 0 and 1
    train_y_nn = np.array(
        [[0.9 if val == 1 else 0.1 for val in row] for row in train_y_bin]
    )

    epoch = 0

    # Initialize weights for inpu
    IL_to_HL_weights = np.array(
        [[uniform(-0.05, 0.05) for _ in range(785)] for _ in range(hidden_units)]
    )

    # Hidden units + 1 is to include the bias
    HL_to_OL_weights = np.array(
        [
            [uniform(-0.05, 0.05) for _ in range(hidden_units + 1)]
            for _ in range(OUTPUTS)
        ]
    )

    # Initialize weights and accuracy vector
    accuracy_vector = []
    previous_HL_to_OL_weight_delta = np.zeros((OUTPUTS, hidden_units + 1))
    previous_IL_to_HL_weight_delta = np.zeros((hidden_units, 785))

    """Training the neural network"""
    while epoch <= 50:
        # Randomly sort the training data to reduce overfitting
        random_idx = [i for i in range(train_x.shape[0])]
        np.random.shuffle(random_idx)
        train_x = np.array([train_x[i] for i in random_idx])
        train_y_nn = np.array([train_y_nn[i] for i in random_idx])
        train_y_bin = np.array([train_y_bin[i] for i in random_idx])

        # Prediction matrix obtained by matrix multiplication of training set (60k x 785) with weights (785, 10) to get 60k binary vectors
        train_hidden_layer = sigmoid(np.matmul(train_x, IL_to_HL_weights.T))
        train_hidden_layer = np.insert(train_hidden_layer, 0, 1, axis=1)

        train_output_layer = sigmoid(np.matmul(train_hidden_layer, HL_to_OL_weights.T))

        train_predictions_matrix = np.array(
            [
                [1 if idx == np.argmax(row) else 0 for idx in range(10)]
                for row in train_output_layer
            ]
        )

        train_current_predictions = np.sum(train_predictions_matrix * train_y_bin)

        # Prediction matrix obtained by matrix multiplication of test set (60k x 785) with weights (785, 10) to get 60k binary vectors
        test_hidden_layer = sigmoid(np.matmul(test_x, IL_to_HL_weights.T))
        test_hidden_layer = np.insert(test_hidden_layer, 0, 1, axis=1)

        test_output_layer = sigmoid(np.matmul(test_hidden_layer, HL_to_OL_weights.T))

        test_prediction_matrix = np.array(
            [
                [1 if idx == np.argmax(row) else 0 for idx in range(10)]
                for row in test_output_layer
            ]
        )

        # Calculate current predictions
        test_current_predictions = np.sum(test_prediction_matrix * test_y_bin)

        accuracy_train = train_current_predictions / train_y_bin.shape[0]
        print("Epoch {} Accuracy for Training: {}".format(epoch, accuracy_train))

        accuracy_test = test_current_predictions / test_y_bin.shape[0]
        print("Epoch {} Accuracy for Test: {}".format(epoch, accuracy_test))

        accuracy_vector.append((epoch, accuracy_train, accuracy_test))

        for train_y_vec, train_x_vec in zip(train_y_nn, train_x):
            """Forward Propagate"""
            # prediction vector obtained by multipying single x vector (1 x 785) and weights (10 x 785) to get predictions
            hidden_layer = sigmoid(np.matmul(IL_to_HL_weights, train_x_vec))

            # Add in bias to the hidden layer
            hidden_layer = np.insert(hidden_layer, 0, 1)

            # Determine output layer
            output_layer = sigmoid(np.matmul(HL_to_OL_weights, hidden_layer))

            # Determine error for output layer
            OL_error = output_layer * (1 - output_layer) * (train_y_vec - output_layer)

            # Determine error for hidden layer
            HL_error = (
                hidden_layer
                * (1 - hidden_layer)
                * np.sum(HL_to_OL_weights.T * OL_error, axis=1)
            )

            """Backpropagate for OL --> HL"""
            # Update Hidden to Output Layer Weights
            # Outer product multiplies each Hj by the error value for each output
            # Return vector is 21 x 10 -- each row contains all outputs deltas for Hj
            HL_to_OL_weight_delta = (
                learning_rate * np.outer(OL_error, hidden_layer)
                + momentum * previous_HL_to_OL_weight_delta
            )
            HL_to_OL_weights += HL_to_OL_weight_delta
            previous_HL_to_OL_weight_delta = HL_to_OL_weight_delta

            # Update Input to Hidden Layer Weights
            IL_to_HL_weight_delta = (
                np.outer(HL_error[1::], learning_rate * train_x_vec)
                + momentum * previous_IL_to_HL_weight_delta
            )
            IL_to_HL_weights += IL_to_HL_weight_delta
            previous_IL_to_HL_weight_delta = IL_to_HL_weight_delta

        epoch += 1
    """Training Ends"""

    """Save Confusion Matrix"""
    np.savetxt(
        "ConfusionMatrixForLearningRate{}_Momentum{}_HiddenUnits{}_Inputs{}_{}.csv".format(
            learning_rate,
            momentum,
            hidden_units,
            train_y_bin.shape[0],
            datetime.datetime.now().strftime("%m_%d_%Y_%H%M"),
        ),
        confusion_matrix(test_y_bin, test_prediction_matrix),
        delimiter=",",
    )

    """Plotting Accuracy"""
    epochs = [v[0] for v in accuracy_vector]
    training_accuracies = [v[1] for v in accuracy_vector]
    test_accuracies = [v[2] for v in accuracy_vector]

    plt.plot(epochs, training_accuracies, label="Train")
    plt.plot(epochs, test_accuracies, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "Neural network with Momentum = {} and Hidden Units = {}".format(
            momentum, hidden_units
        )
    )
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(
        "NNWithLearningRate{}_Momentum{}_HiddenUnits{}_Inputs{}_{}.png".format(
            learning_rate,
            momentum,
            hidden_units,
            train_y_bin.shape[0],
            datetime.datetime.now().strftime("%m_%d_%Y_%H%M"),
        )
    )
    plt.close()


if __name__ == "__main__":
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    # Normalize pixel values and reduce matrix into linear vector
    train_x = np.array([mat.flatten() for mat in np.divide(train_x, 255.0)])
    test_x = np.array([mat.flatten() for mat in np.divide(test_x, 255.0)])

    # Add in bias to each 784 x 1 vector to make it 785 x 1
    train_x = np.insert(train_x, 0, 1, axis=1)
    test_x = np.insert(test_x, 0, 1, axis=1)

    # Create train_y and test_y binary vectors
    # Create train_y and test_y binary vectors
    train_y_bin = np.array(
        [[1 if idx == y_val else 0 for idx in range(10)] for y_val in train_y]
    )

    test_y_bin = np.array(
        [[1 if idx == y_val else 0 for idx in range(10)] for y_val in test_y]
    )

    # Vectors to store experiment parameter information
    momentums = [0, 0.25, 0.5]
    hidden_units = [20, 50, 100]

    # Run experiment by varying the number of hidden units
    for h in hidden_units:
        print("Starting experiment for hidden units: ", h)
        run_experiment(train_x, test_x, train_y_bin, test_y_bin, 0.1, 0.9, h)

    # Run experiment by varing the value of the momentum term
    for m in momentums:
        print("Starting experiment for momentum: ", m)
        run_experiment(train_x, test_x, train_y_bin, test_y_bin, 0.1, m, 100)

    # Run experiment on half the training data
    run_experiment(
        train_x[0:30000], test_x, train_y_bin[0:30000], test_y_bin, 0.1, 0.9, 100
    )

    # Run experiment on a quarter of the training data
    run_experiment(
        train_x[0:15000], test_x, train_y_bin[0:15000], test_y_bin, 0.1, 0.9, 100
    )
