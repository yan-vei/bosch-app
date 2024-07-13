from tensorflow import keras
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Layer, Dropout, Dense, Flatten
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data/preprocessed.csv")

class TransformerDecoderLayer(Layer):
    def __init__(self, num_heads, dff, d_model, rate=0.1, **kwargs):
        super(TransformerDecoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = Sequential([Dense(dff, activation='relu'), Dense(d_model)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training, look_ahead_mask=None):
        attn_output = self.mha(x, x, x, attention_mask=look_ahead_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# Custom R2 score metric
def r2score(y_true, y_pred):
    return tf.py_function(r2_score, [y_true, y_pred], tf.double)

model = keras.models.load_model("model/transformer_ffnn.h5", custom_objects={'TransformerDecoderLayer': TransformerDecoderLayer,
                                                                       'r2score': r2_score})

# Sort data into categories (data coming in at once (sometimes belonging to the same feature))
# timestamp data (date, time, weekday, secs elapsed since eingang):

eingang_ts = ['date_EINGANGSDATUM_UHRZEIT', 'time_EINGANGSDATUM_UHRZEIT', 'weekday_EINGANGSDATUM_UHRZEIT']
verpackt_ts = ['date_VERPACKT_DATUM_UHRZEIT', 'time_VERPACKT_DATUM_UHRZEIT',
               'weekday_VERPACKT_DATUM_UHRZEIT', 'secs_VERPACKT_DATUM_UHRZEIT']
auftragsnummer_ts = ['date_AUFTRAGANNAHME_DATUM_UHRZEIT', 'time_AUFTRAGANNAHME_DATUM_UHRZEIT',
                     'weekday_AUFTRAGANNAHME_DATUM_UHRZEIT', 'secs_AUFTRAGANNAHME_DATUM_UHRZEIT']
lieferschein_ts = ['date_LIEFERSCHEIN_DATUM_UHRZEIT', 'time_LIEFERSCHEIN_DATUM_UHRZEIT',
                   'weekday_LIEFERSCHEIN_DATUM_UHRZEIT', 'secs_LIEFERSCHEIN_DATUM_UHRZEIT']
auftragannahme_ts = ['date_AUFTRAGANNAHME_DATUM_UHRZEIT', 'time_AUFTRAGANNAHME_DATUM_UHRZEIT',
                     'weekday_AUFTRAGANNAHME_DATUM_UHRZEIT', 'secs_AUFTRAGANNAHME_DATUM_UHRZEIT']
bereitgestellt_ts = ['date_BEREITGESTELLT_DATUM_UHRZEIT', 'time_BEREITGESTELLT_DATUM_UHRZEIT',
                     'weekday_BEREITGESTELLT_DATUM_UHRZEIT', 'secs_BEREITGESTELLT_DATUM_UHRZEIT']
TA_ts = ['weekday_TA_DATUM_UHRZEIT', 'date_TA_DATUM_UHRZEIT', 'time_TA_DATUM_UHRZEIT', 'secs_TA_DATUM_UHRZEIT']

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
# target data
processing = ['PROCESSING']

step_1_features = eingang_ts + sonderfahrt + processing
step_2_features = step_1_features + verpackt_ts + auftragsnummer_ts + package_data + auftragsnummer + processing
step_3_features = step_2_features + land + auftragannahme_ts + auftragannahme_ts + lieferschein_ts
step_4_features = step_3_features + bereitgestellt_ts
step_5_features = step_4_features + TA_ts + dienstleister

data_step_5 = data[step_5_features].loc[(data['date_VERPACKT_DATUM_UHRZEIT']>20000000) &
                                        (data['date_AUFTRAGANNAHME_DATUM_UHRZEIT']>20000000) &
                                        (data['date_LIEFERSCHEIN_DATUM_UHRZEIT']>20000000) &
                                        (data['date_BEREITGESTELLT_DATUM_UHRZEIT']>20000000) &
                                        (data['date_TA_DATUM_UHRZEIT']>20000000)]

to_be_dropped = ['date_EINGANGSDATUM_UHRZEIT', 'time_EINGANGSDATUM_UHRZEIT', 'weekday_EINGANGSDATUM_UHRZEIT', 'SONDERFAHRT']
X = data_step_5.drop(columns=['PROCESSING'])
for col in to_be_dropped:
    X = X.drop(columns=[col])
y = data_step_5['PROCESSING']

# Define scalar for the output variable
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(np.array(y).reshape(-1, 1))

def make_prediction(input):
    """
    Make a prediction based on the entry date and priority.
    Other features are loaded from historical data.
    """

    # Sample other data randomly from the dataframe
    random_row = np.random.randint(0, len(X))

    data_p = X.loc[random_row].to_numpy()
    all_data = np.concatenate((input, data_p))
    reshaped_data = all_data.reshape((1, 78, 1))

    prediction = model.predict(reshaped_data)[0][0]

    # scalar to a 2D array
    prediction_reshaped = np.array([[prediction]])

    # inverse transform
    descaled_prediction = scaler_y.inverse_transform(prediction_reshaped)

    return descaled_prediction[0][0] / 86400 # Return the time in days, because prediction is in seconds