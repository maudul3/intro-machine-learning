from keras.datasets import mnist
from random import uniform
import numpy as np
import datetime
import matplotlib.pyplot as plt

OUTPUTS = 10

def confusion_matrix(t, y):
    y = [np.max(np.where(row == 1), initial = -1) for row in y]
    t = [np.max(np.where(row == 1), initial = -1) for row in t]

    confusion = np.zeros((10,10))

    for y_val, t_val in zip(y, t):
        confusion[t_val][y_val] += 1

    return confusion

def run_experiment(train_x, test_x, train_y_bin, test_y_bin, learning_rate):
    # Initialize weights vector for each output
        # Index 
        epoch = 0

        # Initialize weights
        weights = np.array(
            [
                [ uniform(-0.005, 0.005) for _ in range(785) ] 
                for _ in range(OUTPUTS)
            ]
        )

        accuracy_vec = []

        while (epoch < 70):
            
            # Preditction matrix obtained by matrix multiplication of training set (60k x 785) with weights (785, 10) to get 60k binary vectors
            train_prediction_matrix = np.array(
                [ 
                    [ 1 if idx == np.max(np.where(row > 0), initial = -1) else 0 for idx in range(10) ] 
                    for row in np.matmul(train_x, weights.T) 
                ]
            )

            test_prediction_matrix = np.array(
                [ 
                    [ 1 if idx == np.max(np.where(row > 0), initial = -1) else 0 for idx in range(10) ] 
                    for row in np.matmul(test_x, weights.T) 
                ]
            )

            # Calculate current predictions
            train_current_predictions = np.sum(train_prediction_matrix * train_y_bin)
            test_current_predictions = np.sum(test_prediction_matrix * test_y_bin)

            accuracy_train = train_current_predictions / train_y_bin.shape[0]
            accuracy_test = test_current_predictions / test_y_bin.shape[0]
            
            print ("Epoch {} Accuracy for Training: {}".format(epoch, accuracy_train))
            print ("Epoch {} Accuracy for Test: {}".format(epoch, accuracy_test ))

            accuracy_vec.append((epoch, accuracy_train, accuracy_test))
            
            for train_y_vec, x_vec in zip(train_y_bin, train_x):

                # prediction vector obtained by multipying single x vector (1 x 785) and weights (10 x 785) to get predictions
                prediction_vec = np.array([0 if val < 0 else 1 for val in np.matmul(weights, x_vec)])

                # update weights
                weights = weights + learning_rate*np.outer( (train_y_vec - prediction_vec), x_vec )
            
            epoch += 1

        np.savetxt(
            "ConfusionMatrixForLearningRate{}_{}.csv".format(
                learning_rate, datetime.datetime.now().strftime("%m_%d_%Y_%H%M")
            ), 
            confusion_matrix(test_y_bin, test_prediction_matrix), 
            delimiter=","
        )
        
        epochs = [v[0] for v in accuracy_vec]
        training_accuracies = [v[1] for v in accuracy_vec]
        test_accuracies = [v[2] for v in accuracy_vec]

        plt.plot(epochs, training_accuracies, label="train")
        plt.plot(epochs, test_accuracies, label="test")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Perceptron with learning rate = {}".format(learning_rate))
        plt.legend()
        plt.savefig('PerceptronWithLearningRate{}_{}.png'.format(
            learning_rate, datetime.datetime.now().strftime("%m_%d_%Y_%H%M")
            )
        )
        plt.close()


if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    # Normalize pixel values and reduce matrix into linear vector
    train_x = np.array([mat.flatten() for mat in np.divide(train_x, 255.)])
    test_x = np.array([mat.flatten() for mat in np.divide(test_x, 255.)])

    # Add in bias to each 784 x 1 vector to make it 785 x 1 
    train_x = np.insert(train_x, 0, 1, axis=1)
    test_x = np.insert(test_x, 0, 1, axis=1)

    # Create train_y and test_y binary vectors
    train_y_bin = np.array(
        [
            [ 1 if idx == y_val else 0 for idx in range(10) ] 
            for y_val in train_y
        ]
    )

    test_y_bin = np.array(
        [
            [ 1 if idx == y_val else 0 for idx in range(10) ] 
            for y_val in test_y
        ]
    )

    # Initialize eta vector
    eta = [0.001, 0.01, 0.1]

    for learning_rate in eta:
        print ("Starting experiment for : ", learning_rate)
        run_experiment(train_x, test_x, train_y_bin, test_y_bin, learning_rate)
        break
