import numpy as np
import pandas as pd
from functools import reduce

# In raw data: 1813 spam, 2788 not spam
# Train data = 907 spam, 1394 not spam
# Test data = 906 spam, 1394 not spam

def normal_distribution(sample, mean, stdev):
    return  np.log(1 / ( np.sqrt(2*np.pi) * stdev) ) * np.exp (-(sample - mean)**2 / ( 2*stdev**2 ) )

def calculate_probability(training_row, mean_row, stdev_row, class_prob):

    prob = class_prob

    for i in range(len(mean_row)):
        prob *= normal_distribution(training_row[i], mean_row[i], stdev_row[i])

    return prob

def true_pos(tup):
    if (tup[0] == tup[1]) and tup[1] == 1:
        return 1
    else:
        return 0

def true_neg(tup):
    if (tup[0] == tup[1]) and tup[1] == 0:
        return 1
    else:
        return 0

def false_pos(tup):
    if (tup[0] != tup[1]) and tup[1] == 0:
        return 1
    else:
        return 0

def false_neg(tup):
    if (tup[0] != tup[1]) and tup[1] == 1:
        return 1
    else:
        return 0

if __name__ == '__main__':
    df = pd.read_csv("/Users/drewmahler/Desktop/School/CS545/CS545Code/spambase_data.csv")

    # Create training dataset
    train_df = pd.concat([
        df[df.spam_flag == 0].head(1394),
        df[df.spam_flag == 1].head(907) 
    ])
    
    # Anti-join the training data to produce the test dataset
    test_df = (
        df.merge(train_df.ID, how='left', on='ID', indicator=True)
        .query("_merge == 'left_only' ")
        .drop(labels="_merge", axis=1)
    )

    # Determine probability of spam and not spam
    not_spam_count, spam_count = (
        train_df
        .groupby(['spam_flag'])
        .count() 
        .values.tolist()
    )

    spam_prob = spam_count[0] / (spam_count[0] + not_spam_count[0])
    not_spam_prob = not_spam_count[0] / (spam_count[0] + not_spam_count[0])

    # Create dataframe with means and standard deviation for the spam flags
    stdev_df = (
        train_df
        .drop(labels="ID", axis=1)
        .groupby(['spam_flag']).std()
    )
    mean_df = (
        train_df
        .drop(labels="ID", axis=1)
        .groupby(['spam_flag']).mean()
    )
    
    # Convert standard dev. and mean dataframes to lists
    not_spam_stdev, spam_stdev = (
        stdev_df
        .values.tolist()
    )

    not_spam_mean, spam_mean = (
        mean_df
        .values.tolist()
    )

    # Modify standard dev rows to ensure they are no zero values
    not_spam_stdev = [0.0001 if val == 0 else val for val in not_spam_stdev]
    spam_stdev = [0.0001 if val == 0 else val for val in spam_stdev]

    # Convert the training data into a list of lists for probabilistic evaluation
    train_rows = (
        train_df
        .drop(labels=["ID"], axis=1)
        .values.tolist()
    )
    
    predictions_and_true = []

    for row in train_rows:
        
        sample_spam_prob = calculate_probability(
            row,
            spam_mean,
            spam_stdev,
            spam_prob
        )

        sample_not_spam_prob = calculate_probability(
            row,
            not_spam_mean,
            not_spam_stdev,
            not_spam_prob
        )
        print ("New set:")
        print ("Spam: ", sample_spam_prob)
        print ("Not spam: ", sample_not_spam_prob)

        if sample_spam_prob > not_spam_prob:
            prediction = 1
        else:
            prediction = 0

        predictions_and_true.append((prediction, row[-1]))

    true_positives = reduce(lambda x, y: x + y, map(true_pos, predictions_and_true))
    true_negatives = reduce(lambda x, y: x + y, map(true_neg, predictions_and_true))
    false_positives = reduce(lambda x, y: x + y, map(false_pos, predictions_and_true))
    false_negatives = reduce(lambda x, y: x + y, map(false_neg, predictions_and_true))

    x = 1