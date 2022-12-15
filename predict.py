import numpy as np

def predict_flower(trained_model, input_features):
    """
    predict iris type

    Args:
        trained_model (scikit-learn estimator): trained classifier
        input_features (numpy array like): 4 numeric values
    Returns:
        (str) the model's prediction
    """

    return trained_model.predict(input_features.reshape(1, -1))