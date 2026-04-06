# ======================================================
# 🧠 PHASE-12 NEURAL CITY BRAIN
# ======================================================

import numpy as np
from sklearn.linear_model import SGDRegressor

# Online-learning neural-like model
model = SGDRegressor(max_iter=1, learning_rate="constant", eta0=0.0001)

# Memory buffers
X_memory = []
y_memory = []

trained = False


def prepare_features(lat, lon, risk, density):
    """
    Convert geo + risk info into ML features
    """
    return np.array([lat, lon, risk, density], dtype=float)


def train_step(lat, lon, risk, density):
    """
    Online training step — runs every detection
    """
    global trained

    X = prepare_features(lat, lon, risk, density).reshape(1, -1)

    # target = future risk approximation
    target = np.array([risk])

    X_memory.append(X[0])
    y_memory.append(target[0])

    if len(X_memory) > 10:
        model.partial_fit(np.array(X_memory), np.array(y_memory))
        trained = True


def predict_future_risk(lat, lon, risk, density):
    """
    Neural prediction for future road risk
    """
    if not trained:
        return None

    X = prepare_features(lat, lon, risk, density).reshape(1, -1)
    pred = model.predict(X)[0]

    return round(float(max(0, min(100, pred))), 2)
