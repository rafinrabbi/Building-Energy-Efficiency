import numpy as np # linear algebra
import pandas as pd 


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


df = pd.read_csv("energy_efficiency_data.csv")
df.head()

scaler = StandardScaler()
X = df.drop(columns=["Heating_Load", "Cooling_Load"])
X_scaled = scaler.fit_transform(X)

y_heating = df["Heating_Load"]
y_cooling = df["Cooling_Load"]

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_heating, test_size=0.2, random_state=42)


# rf = RandomForestRegressor(random_state=42)
# rf.fit(X_train, y_train)

# y_pred = rf.predict(X_test)

# mae= mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2= r2_score(y_test, y_pred)

# print(f" MAE : {mae:.2f}, RMSE : {rmse:.2f}, R2 Score : {r2:.2f}")

# param_dist = {
#     'n_estimators': [50, 100, 200, 300],
#     'max_depth': [None, 10, 20, 30, 50],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt']
# }

# random_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), 
#                                    param_distributions=param_dist,
#                                    n_iter=20, 
#                                    cv=5, 
#                                    scoring='r2', 
#                                    verbose=1, 
#                                    n_jobs=-1)

# random_search.fit(X_train, y_train)
# best_params = random_search.best_params_
# print("Best Parameters:", best_params)

# Build Autoencoder
input_dim = X_scaled.shape[1]
encoding_dim = 16  # You can tune this

input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
encoder = Model(inputs=input_layer, outputs=encoded)

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train Autoencoder
history = autoencoder.fit(X_scaled, X_scaled, epochs=100, batch_size=32, shuffle=True, validation_split=0.1, verbose=1)

# Plot loss curves
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Autoencoder Loss Curves')
plt.savefig("autoencoder_loss.png")  # Save the plot as an image file
# plt.show()  # You can comment this out if running in a non-interactive environment

# Encode features
X_encoded = encoder.predict(X_scaled)
print(X_encoded)

# Now use X_encoded for regression
# Train/test split on encoded features
X_train_enc, X_test_enc, y_train, y_test = train_test_split(X_encoded, y_heating, test_size=0.2, random_state=42)

# XGBoost
xgb = XGBRegressor(random_state=42, n_jobs=-1)
xgb.fit(X_train_enc, y_train)
y_pred_xgb = xgb.predict(X_test_enc)

# CatBoost
cat = CatBoostRegressor(verbose=0, random_state=42)
cat.fit(X_train_enc, y_train)
y_pred_cat = cat.predict(X_test_enc)

# Random Forest
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf.fit(X_train_enc, y_train)
y_pred_rf = rf.predict(X_test_enc)

# Stacking
estimators = [
    ('xgb', xgb),
    ('cat', cat),
    ('rf', rf)
]
stack = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor(n_jobs=-1, random_state=42))
stack.fit(X_train_enc, y_train)
y_pred_stack = stack.predict(X_test_enc)

# Results
def print_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2 Score: {r2:.2f}")

print_metrics("XGBoost", y_test, y_pred_xgb)
print_metrics("CatBoost", y_test, y_pred_cat)
print_metrics("Random Forest", y_test, y_pred_rf)
print_metrics("Stacked Model", y_test, y_pred_stack)
