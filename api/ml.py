import pickle

# TODO



with open('models/logistic_regression.pkl', 'rb') as f:
    model = pickle.load(f)


def predict(sample):
    return model.predict(sample).tolist()[0]