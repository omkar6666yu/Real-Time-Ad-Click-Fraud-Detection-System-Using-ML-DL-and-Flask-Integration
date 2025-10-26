 # models/dl_models.py
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def build_ann(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

def build_gru(seq_len, feat_dim):
    inp = layers.Input(shape=(seq_len, feat_dim))
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.GRU(128, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

def build_lstm(seq_len, feat_dim):
    inp = layers.Input(shape=(seq_len, feat_dim))
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

def build_cnn_lstm(seq_len, feat_dim):
    inp = layers.Input(shape=(seq_len, feat_dim))
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inp)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(128)(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

def train_tf_model(model, X, y, epochs=10, batch_size=128, val_split=0.1, model_path=None):
    es = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=val_split, callbacks=[es], verbose=2)
    if model_path:
        model.save(model_path)
    return model, history
