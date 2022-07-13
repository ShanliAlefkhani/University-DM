import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


def get_onehot_dataset():
    scaler = MinMaxScaler()
    input_file = "cardio_train.csv"
    df = pd.read_csv(input_file, header=0)

    df_cat = df.select_dtypes(include='object')
    encoder = OneHotEncoder(sparse=False, handle_unknown='error')
    df_cat_encoded = encoder.fit_transform(df_cat)
    categorical_columns = [f'{col}_{cat}' for i, col in enumerate(df_cat.columns) for cat in encoder.categories_[i]]
    df_num = df.select_dtypes(exclude='object')
    df_num_ordinal = df_num.drop(["id", "smoke", "alco", "active", "cardio"], axis=1)
    df_num_cat = df_num.drop(["age", "height", "weight", "ap_hi", "ap_lo"], axis=1)
    scaler.fit(df_num_ordinal)
    df_num_ordinal = pd.DataFrame(scaler.transform(df_num_ordinal),
                                  columns=["age", "height", "weight", "ap_hi", "ap_lo"])
    df_num = df_num_ordinal.join(df_num_cat)
    df_cat = pd.DataFrame(df_cat_encoded, columns=categorical_columns)
    df = df_num.join(df_cat).drop("id", axis=1)
    return df

def get_dataset():
    scaler = MinMaxScaler()
    input_file = "phase2/cardio_train.csv"
    df = pd.read_csv(input_file, header=0)

    df_cat = df.select_dtypes(include='object')
    df_num = df.select_dtypes(exclude='object')
    df_num_ordinal = df_num.drop(["id", "smoke", "alco", "active", "cardio"], axis=1)
    df_num_cat = df_num.drop(["age", "height", "weight", "ap_hi", "ap_lo"], axis=1)
    scaler.fit(df_num_ordinal)
    df_num_ordinal = pd.DataFrame(scaler.transform(df_num_ordinal),
                                  columns=["age", "height", "weight", "ap_hi", "ap_lo"])
    df_num = df_num_ordinal.join(df_num_cat)
    df = df_num.join(df_cat).drop("id", axis=1)
    return df
