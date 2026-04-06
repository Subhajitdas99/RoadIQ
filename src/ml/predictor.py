import numpy as np

# 🧠 Phase-5.5 ML Predictor (MVP)
# Later you will replace this with trained model (sklearn / torch)

class RiskPredictor:

    def __init__(self):
        # Fake learned weights (simulate trained model)
        self.weights = np.array([
            0.45,  # current risk
            0.25,  # density
            0.15,  # high priority count
            0.10,  # medium count
            0.05   # road health inverse
        ])

    def build_features(self, risk, density, high, medium, road_health):
        return np.array([
            risk,
            density * 100,
            high,
            medium,
            100 - road_health
        ])

    def predict(self, risk, density, high, medium, road_health):

        x = self.build_features(risk, density, high, medium, road_health)

        score = float(np.dot(self.weights, x))

        predicted = max(0, min(100, round(score, 2)))

        if predicted > risk + 10:
            trend = "increasing"
        elif predicted < risk - 10:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "predicted_risk": predicted,
            "trend": trend
        }


# Global predictor instance
risk_model = RiskPredictor()
