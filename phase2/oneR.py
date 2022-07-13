from files import get_dataset
from sklearn.model_selection import train_test_split
import numpy as np

df = get_dataset()
y = df[["cardio"]]
df_num = df.select_dtypes(exclude='object')
df_num_cat = df_num.drop([ "smoke", "alco", "active"], axis=1)
print(df_num_cat)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


