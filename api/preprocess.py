import numpy as np
import pandas as pd 
import pickle


# load onehot encoder
with open("preprocessors/OnehotEncoding.pkl", "rb") as f:
    onehot_encoder = pickle.load(f)

# load scaler
with open("preprocessors/StandardScaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# TODO
COLUMNS_TO_REMOVE = [
    "Over18",
    "EmployeeCount",
    "EmployeeNumber"
]

# TODO
COLUMNS_TO_ONEHOT_ENCODE = [
"BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime"
]


def preprocess(sample: dict) -> np.ndarray:
    sample_df = pd.DataFrame(sample, index=[0])

    sample_df = drop_columns(sample_df)
    sample_df = encode_columns(sample_df)
    sample_df = create_features(sample_df)
    scaled_sample_values = scale(sample_df.values)
    scaled_sample_values = scaled_sample_values.reshape(1, -1)
    return scaled_sample_values


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=COLUMNS_TO_REMOVE)


def encode_columns(df: pd.DataFrame) -> pd.DataFrame:
    # create a new dataframe with one-hot encoded columns
    encoded_df = pd.DataFrame(onehot_encoder.transform(df[COLUMNS_TO_ONEHOT_ENCODE]).toarray())
    # set new column names
    column_names = onehot_encoder.get_feature_names(COLUMNS_TO_ONEHOT_ENCODE)
    encoded_df.columns = column_names
    # drop raw columns, and add one-hot encoded columns instead
    df = df.drop(columns=COLUMNS_TO_ONEHOT_ENCODE, axis=1)
    df = df.join(encoded_df)

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # create MeanAttritionYear feature
    df["MeanAttritionYear"] = df["TotalWorkingYears"] / (df["NumCompaniesWorked"] + 1)
    # create YearsAtCompanyCat
    bins = pd.IntervalIndex.from_tuples([(-1, 5), (5, 10), (10, 15), (15, 100)])
    cat_YearsAtCompany = pd.cut(df["YearsAtCompany"].to_list(), bins)
    cat_YearsAtCompany.categories = [0, 1, 2, 3]
    df["YearsAtCompanyCat"] = cat_YearsAtCompany

    return df


def scale(arr: np.ndarray) -> np.ndarray:
    return scaler.transform(arr)
