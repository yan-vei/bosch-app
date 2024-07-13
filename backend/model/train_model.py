import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, Dense, Input, Add
from tensorflow.keras import Model
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('../data/preprocessed.csv')

eingang_ts = ['date_EINGANGSDATUM_UHRZEIT', 'time_EINGANGSDATUM_UHRZEIT', 'weekday_EINGANGSDATUM_UHRZEIT']
# other data:
package_data = ['LAENGE_IN_CM', 'BREITE_IN_CM', 'HOEHE_IN_CM', 'GEWICHT_IN_KG', 'count_PACKSTUECKART=BEH',
                'count_PACKSTUECKART=CAR', 'count_PACKSTUECKART=GBP', 'count_PACKSTUECKART=PAL',
                'count_PACKSTUECKART=PKI', 'count_PACKSTUECKART=UNKNOWN', 'PACKAGE_COUNT']
auftragsnummer = ['category_AUFTRAGSNUMMER=DSGA', 'category_AUFTRAGSNUMMER=RBMANUSHIP', 'category_AUFTRAGSNUMMER=return']
land = ['LAND=AT', 'LAND=AUT', 'LAND=BE', 'LAND=BR', 'LAND=CH', 'LAND=CN', 'LAND=CZ', 'LAND=DE', 'LAND=DK', 'LAND=DR',
        'LAND=ES', 'LAND=FCA', 'LAND=FR', 'LAND=HU', 'LAND=IE', 'LAND=IN', 'LAND=IT', 'LAND=JP', 'LAND=KR', 'LAND=MX',
        'LAND=NL', 'LAND=None', 'LAND=PL', 'LAND=RO', 'LAND=RU', 'LAND=TR', 'LAND=UK', 'LAND=US']
sonderfahrt = ['SONDERFAHRT']
dienstleister = ['DIENSTLEISTER=DHL', 'DIENSTLEISTER=None', 'DIENSTLEISTER=TNT', 'DIENSTLEISTER=UPS']

step_1_features = eingang_ts + sonderfahrt
step_2_features = step_1_features + package_data + auftragsnummer
step_3_features = step_2_features + land
step_4_features = step_3_features
step_5_features = step_4_features + dienstleister

def augment_data(data, num_augmentations=3):
    augmented_data = data.copy()
    for _ in range(num_augmentations):
        temp_data = data.copy()
        # Apply random noise
        temp_data['PROCESSING'] += np.random.normal(0, 0.01, size=temp_data['PROCESSING'].shape)
        augmented_data = pd.concat([augmented_data, temp_data], axis=0)
    return augmented_data

data_augmented = augment_data(data, num_augmentations=3)
X = data_augmented[step_5_features]
y = data_augmented['PROCESSING']
X = np.tile(X, (4, 1))
y = np.tile(y, 4)

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

sequence_length = 10
X_reshaped = []
y_reshaped = []

for i in range(len(X_scaled) - sequence_length):
    X_reshaped.append(X_scaled[i:i + sequence_length])
    y_reshaped.append(y_scaled[i + sequence_length])

X_reshaped = np.array(X_reshaped)
y_reshaped = np.array(y_reshaped)

X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_reshaped, test_size=0.2, random_state=42)

class MetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1} - loss: {logs['loss']:.4f} - val_loss: {logs['val_loss']:.4f} - mae: {logs['mae']:.4f} - val_mae: {logs['val_mae']:.4f} - mse: {logs['mse']:.4f} - val_mse: {logs['val_mse']:.4f}")
input_shape = (sequence_length, X_train.shape[2])  # (sequence_length, num_features)

# TCN
inputs = Input(shape=input_shape)
conv1 = Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu')(inputs)
dropout1 = Dropout(0.3)(conv1)
conv2 = Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu')(dropout1)
dropout2 = Dropout(0.3)(conv2)
flatten = Flatten()(dropout2)

# Bi-LSTM
x = Bidirectional(LSTM(25, return_sequences=True))(inputs) #25 units

x = Flatten()(x)
x = Dense(50, activation='relu')(x)
outputs = Dense(1)(x)  # Assuming regression task, hence one output node
model = Model(inputs=inputs, outputs=outputs)
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])
#model.summary()

#history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[MetricsCallback()])
#loss, mae, mse = model.evaluate(X_test, y_test)

model.save('model.keras')
