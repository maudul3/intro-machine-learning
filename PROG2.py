import numpy as np
import pandas as pd

# In raw data: 1813 spam, 2788 not spam
# Train data = 907 spam, 1394 not spam
# Test data = 906 spam, 1394 not spam

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
