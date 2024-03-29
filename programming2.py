import numpy as np
from numpy.core.fromnumeric import std
import pandas as pd
from functools import reduce

# In raw data: 1813 spam, 2788 not spam
# Train data = 907 spam, 1394 not spam
# Test data = 906 spam, 1394 not spam


def normal_distribution(sample, mean, stdev):
    """Normal distribution

    Inputs:
        sample (float): sample data point
        mean (float): mean of distribution
        stdev (float): standard deviation of distribution

    Outputs:
        float: probability of sample within distribution
    """
    return np.log(
        1
        / (np.sqrt(2 * np.pi) * stdev)
        * np.exp(-((sample - mean) ** 2) / (2 * stdev**2))
    )


def calculate_probability(test_row, mean_row, stdev_row, class_prob):
    """Calculate probability for a sample given mean, stdev,
    and probability for a given class

    Inputs:
        test_row(list[float]): row of numerical features for test data
        mean_row(list[float]): row of means of features for training data
            of the given class
        stdev_row(list[flow]): row of standard deviations of features for
            training data of the given class
        class_prob(list[flow]): probability for the given class

    Outputs:
        float: probability that test data is in the given class
    """
    prob = np.log(class_prob)

    for i in range(len(mean_row)):
        prob += normal_distribution(test_row[i], mean_row[i], stdev_row[i])

    return prob


def true_pos(tup):
    """Identifies if predicted value is true positive

    Inputs:
        tuple(int, int): tuple[0] is predicted value and tuple[1]
            is true value

    Outputs:
        int: 1 if predicted value is spam and true value is spam else 0
    """
    if (tup[0] == tup[1]) and tup[1] == 1:
        return 1
    else:
        return 0


def true_neg(tup):
    """Identifies if predicted value is true negative

    Inputs:
        tuple(int, int): tuple[0] is predicted value and tuple[1]
            is true value
    Outputs:
        int: returns 1 if predicted value is not-spam and true value is not-spam else 0
    """
    if (tup[0] == tup[1]) and tup[1] == 0:
        return 1
    else:
        return 0


def false_pos(tup):
    """Identifies if predicted value is false positive

    Inputs:
        tuple(int, int): tuple[0] is predicted value and tuple[1]
            is true value
    Outputs:
        int: returns 1 if predicted value is spam and true value is not-spam else 0
    """
    if (tup[0] != tup[1]) and tup[1] == 0:
        return 1
    else:
        return 0


def false_neg(tup):
    """Identifies if predicted value is false negative

    Inputs:
        tuple(int, int): tuple[0] is predicted value and tuple[1]
            is true value

    Outputs:
        int: returns 1 if predicted value is not-spam and true value is spam else 0
    """
    if (tup[0] != tup[1]) and tup[1] == 1:
        return 1
    else:
        return 0


if __name__ == "__main__":
    # Read data from csv
    df = pd.read_csv(
        "/Users/drewmahler/Desktop/School/CS545/CS545Code/spambase_data.csv"
    )

    # Randomize dataset to avoid any inherent ordering bias
    df = df.sample(frac=1, random_state=3).reset_index(drop=True)

    # Create training dataset
    train_df = pd.concat(
        [df[df.spam_flag == 0].head(1394), df[df.spam_flag == 1].head(907)]
    )

    # Anti-join the training data to produce the test dataset
    test_df = (
        df.merge(train_df.ID, how="left", on="ID", indicator=True)
        .query("_merge == 'left_only' ")
        .drop(labels="_merge", axis=1)
    )

    # Determine probability of spam and not spam
    not_spam_count, spam_count = train_df.groupby(["spam_flag"]).count().values.tolist()

    # Find P(spam) and P(not spam)
    spam_prob = spam_count[0] / (spam_count[0] + not_spam_count[0])
    not_spam_prob = not_spam_count[0] / (spam_count[0] + not_spam_count[0])

    # Create dataframe with means and standard deviation for the spam flags
    stdev_df = train_df.drop(labels="ID", axis=1).groupby(["spam_flag"]).std()

    mean_df = train_df.drop(labels="ID", axis=1).groupby(["spam_flag"]).mean()

    # Convert standard dev. and mean dataframes to lists
    not_spam_stdev, spam_stdev = stdev_df.values.tolist()

    not_spam_mean, spam_mean = mean_df.values.tolist()

    # Modify standard dev rows to ensure there are no zero values
    not_spam_stdev = [0.0000001 if val == 0 else val for val in not_spam_stdev]
    spam_stdev = [0.0000001 if val == 0 else val for val in spam_stdev]

    # Convert the training data into a list of lists for probabilistic evaluation
    test_rows = test_df.drop(labels=["ID"], axis=1).values.tolist()

    # List that will eventually be populated with tuples in form (prediction, true value)
    predictions_and_true = []

    # Loop through each row in test set
    for row in test_rows:
        # probability this sample is spam
        sample_spam_prob = calculate_probability(row, spam_mean, spam_stdev, spam_prob)

        # probability that this sample is not spam
        sample_not_spam_prob = calculate_probability(
            row, not_spam_mean, not_spam_stdev, not_spam_prob
        )

        # Determine predicted class
        if sample_spam_prob > not_spam_prob:
            prediction = 1
        else:
            prediction = 0

        # Append to list of prediction and true values as (prediction, true value)
        predictions_and_true.append((prediction, row[-1]))

    # Determine the necessary components of the confusion matrix
    true_positives = reduce(lambda x, y: x + y, map(true_pos, predictions_and_true))
    true_negatives = reduce(lambda x, y: x + y, map(true_neg, predictions_and_true))
    false_positives = reduce(lambda x, y: x + y, map(false_pos, predictions_and_true))
    false_negatives = reduce(lambda x, y: x + y, map(false_neg, predictions_and_true))

    print("True positives:", true_positives)
    print("True negatives:", true_negatives)
    print("False positives:", false_positives)
    print("False negatives:", false_negatives)

    print(
        "Accuracy: ",
        (true_positives + true_negatives)
        / (true_positives + true_negatives + false_negatives + false_positives),
    )
    print("Recall: ", true_positives / (true_positives + false_negatives))
    print("Precision: ", true_positives / (true_positives + false_positives))
