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
'MonthlyIncome', 'Department_Sales', 'JobRole_Human Resources', 'JobRole_Sales Executive'
]

# TODO
COLUMNS_TO_ONEHOT_ENCODE = [
'BusinessTravel_0', 'BusinessTravel_1', 'BusinessTravel_2',
       'Department_Human Resources', 'Department_Research & Development',
       'Department_Sales', 'EducationField_Human Resources',
       'EducationField_Life Sciences', 'EducationField_Marketing',
       'EducationField_Medical', 'EducationField_Other',
       'EducationField_Technical Degree', 'Gender_Male',
       'JobRole_Healthcare Representative', 'JobRole_Human Resources',
       'JobRole_Laboratory Technician', 'JobRole_Manager',
       'JobRole_Manufacturing Director', 'JobRole_Research Director',
       'JobRole_Research Scientist', 'JobRole_Sales Executive',
       'JobRole_Sales Representative', 'MaritalStatus_Divorced',
       'MaritalStatus_Married', 'MaritalStatus_Single', 'OverTime_Yes'
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

    # TODO

    return df


def scale(arr: np.ndarray) -> np.ndarray:
    return scaler.transform(arr)
