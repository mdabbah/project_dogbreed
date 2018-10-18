import pandas as pd
import numpy as np

labels = pd.read_csv('..\labels.csv')
labels['breed_id'] = np.nan
classes = list(set(labels['breed']))

for c in classes:
    labels.loc[labels['breed'] == c, ['breed_id']] = np.int(classes.index(c))

labels.to_csv('..\label_updated.csv', index=False)
