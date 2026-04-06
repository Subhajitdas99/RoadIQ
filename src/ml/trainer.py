import numpy as np
from sklearn.linear_model import LinearRegression
from src.ml.features import build_geo_features

class GeoTrainer:

    def __init__(self):
        self.model = LinearRegression()
        self.trained = False

    def train(self, geo_history):

        if len(geo_history) < 5:
            return  # Not enough data yet

        X = []
        y = []

        # Predict next risk based on current state
        for i in range(len(geo_history) - 1):

            features = build_geo_features(geo_history[i])
            next_risk = geo_history[i+1]["risk"]

            X.append(features)
            y.append(next_risk)

        X = np.array(X)
        y = np.array(y)

        self.model.fit(X, y)
        self.trained = True

    def predict(self, features):

        if not self.trained:
            return None

        pred = self.model.predict([features])[0]
        return max(0, min(100, round(float(pred), 2)))


geo_trainer = GeoTrainer()
