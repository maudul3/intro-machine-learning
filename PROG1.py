from keras.datasets import mnist
from random import uniform
import numpy as np
import datetime
from math import exp
import matplotlib.pyplot as plt

OUTPUTS = 10

def sigmoid(z):
    try:
        return 1/( 1 + np.exp(-z) )
    except:
        print (z, " ", round(z,10))

def sum_jth_weight_for_all_k(weights, output_error):
    """
    
    """
    return np.sum(weights.T*output_error, axis = 1)

def run_experiment(
    train_x, test_x, train_y_bin, test_y_bin, learning_rate, momentum, hidden_units
):
        train_y_nn = np.array(
        [
            [ 0.9 if val == 1 else 0.1 for val in row ] 
            for row in train_y_bin
        ]
        )

        test_y_nn = np.array(
        [
           [ 0.9 if val == 1 else 0.1 for val in row ] 
            for row in test_y_bin
        ]
    )
    # Initialize weights vector for each output
        # Index 
        epoch = 0

        # Initialize weights
        IL_to_HL_weights = np.array(
            [
                [ uniform(-0.05, 0.05) for _ in range(785) ] 
                for _ in range(hidden_units)
            ]
        )

        # Hidden units + 1 is to include the bias
        HL_to_OL_weights = np.array(
            [
                [ uniform(-0.05, 0.05) for _ in range(hidden_units + 1) ] 
                for _ in range(OUTPUTS)
            ]
        )

        # Initialize weights and accuracy vector
        accuracy_vec = []
        previous_HL_to_OL_weight_delta = np.zeros((hidden_units + 1, OUTPUTS))
        previous_IL_to_HL_weight_delta = np.zeros((785, hidden_units))

        while (epoch <= 50):
            
            # Preditction matrix obtained by matrix multiplication of training set (60k x 785) with weights (785, 10) to get 60k binary vectors
            train_hidden_layer = np.array([
                [ sigmoid(val) for val in row ]
                for row in np.matmul(train_x, IL_to_HL_weights.T)
            ])

            train_hidden_layer = np.insert(train_hidden_layer, 0, 1, axis=1)
            
            train_output_layer = np.array([
                [ sigmoid(val) for val in row ]
                for row in np.matmul(train_hidden_layer, HL_to_OL_weights.T)
            ])

            train_predictions_matrix = np.array(
                [ 
                    [ 1 if idx == np.max(np.where(row >= 0.9), initial = -1) else 0 for idx in range(10) ] 
                    for row in train_output_layer
                ]
            )

            train_current_predictions = np.sum(
                train_predictions_matrix * train_y_bin
            )
            

            # Calculate current predictions
            '''train_current_predictions = np.sum(train_prediction_matrix * train_y_bin)
            test_current_predictions = np.sum(test_prediction_matrix * test_y_bin)'''

            accuracy_train = train_current_predictions / train_y_bin.shape[0]
            #accuracy_test = test_current_predictions / test_y_bin.shape[0]
            
            print ("Epoch {} Accuracy for Training: {}".format(epoch, accuracy_train))
            #print ("Epoch {} Accuracy for Test: {}".format(epoch, accuracy_test ))

            accuracy_vec.append((epoch, accuracy_train))
            count=0
            print (count)
            for train_y_vec, x_vec in zip(train_y_nn, train_x):
        
                
                # prediction vector obtained by multipying single x vector (1 x 785) and weights (10 x 785) to get predictions
                hidden_layer = np.array([sigmoid(val) for val in np.matmul(IL_to_HL_weights, x_vec)])
                
                 # Add in bias to the hidden layer
                hidden_layer = np.insert(hidden_layer, 0, 1)

                # Determine output layer
                output_layer = np.array([sigmoid(val) for val in np.matmul(HL_to_OL_weights, hidden_layer)])

                # Determine error for output layer
                OL_error = output_layer * (1 - output_layer) * (train_y_vec - output_layer)

                # Determine error for hidden layer
                HL_error = hidden_layer * (1 - hidden_layer) * np.sum(HL_to_OL_weights.T*OL_error, axis = 1) 

                '''Backpropagate for OL --> HL'''
                # Outer product multiplies each Hj by the error value for each output
                # Return vector is 21 x 10 -- each row contains all outputs deltas for Hj
                HL_to_OL_weight_delta = (
                    learning_rate*np.outer(hidden_layer, OL_error) + momentum*previous_HL_to_OL_weight_delta
                )
                # Update Hidden Layer Weights
                HL_to_OL_weights += HL_to_OL_weight_delta.T

                previous_HL_to_OL_weight_delta = HL_to_OL_weight_delta

                IL_to_HL_weight_delta = (
                    np.outer(learning_rate*x_vec, (HL_error[1::]) ) + momentum*previous_IL_to_HL_weight_delta
                )

                IL_to_HL_weights += IL_to_HL_weight_delta.T

                previous_IL_to_HL_weight_delta = IL_to_HL_weight_delta

            epoch += 1

        '''np.savetxt(
            "ConfusionMatrixForLearningRate{}_{}.csv".format(
                learning_rate, datetime.datetime.now().strftime("%m_%d_%Y_%H%M")
            ), 
            confusion_matrix(test_y_bin, test_prediction_matrix), 
            delimiter=","
        )
        '''
        epochs = [v[0] for v in accuracy_vec]
        training_accuracies = [v[1] for v in accuracy_vec]
        #test_accuracies = [v[2] for v in accuracy_vec]

        plt.plot(epochs, training_accuracies, label="train")
        #plt.plot(epochs, test_accuracies, label="test")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Neural network with learning rate = {}".format(learning_rate))
        plt.legend()
        '''plt.savefig('PerceptronWithLearningRate{}_{}.png'.format(
            learning_rate, datetime.datetime.now().strftime("%m_%d_%Y_%H%M")
            )
        )'''
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
    learning_rates = [0.001, 0.01, 0.1]

    momentums = []

    hidden_units = [20, 50, 100]

    for h in hidden_units:
        print ("Starting experiment for : ", h)
        run_experiment(
            train_x,
            test_x,
            train_y_bin,
            test_y_bin, 
            0.1,
            0.9,
            h
        )
        break
