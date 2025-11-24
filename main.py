import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input, concatenate, Dropout, StringLookup, Normalization, CategoryEncoding
from tensorflow.keras.models import Model

df = pd.read_csv('placement_dataset.csv') 

X = df.drop('Placement got', axis=1)
y = df['Placement got']

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.10, random_state=42, stratify=y
)

cgpa_train_vals = X_train['CGPA'].to_numpy().astype('float32')
vit_train_raw = X_train['VIT Branch'].to_numpy()
subject_train_raw = X_train['Subject Branch'].to_numpy()

cgpa_test_vals = X_test['CGPA'].to_numpy().astype('float32')
vit_test_raw = X_test['VIT Branch'].to_numpy()
subject_test_raw = X_test['Subject Branch'].to_numpy()

cgpa_norm_layer = Normalization(axis=None)
cgpa_norm_layer.adapt(cgpa_train_vals)

vit_lookup = StringLookup(
  vocabulary=np.unique(vit_train_raw),
  mask_token=None
)
vit_encoder = CategoryEncoding(
  num_tokens=vit_lookup.vocabulary_size(),
  output_mode="one_hot"
)

subject_lookup = StringLookup(
  vocabulary=np.unique(subject_train_raw),
  mask_token=None
)
subject_encoder = CategoryEncoding(
  num_tokens=subject_lookup.vocabulary_size(),
  output_mode="one_hot"
)

cgpa_in = Input(shape=(1,), name='cgpa_input', dtype=tf.float32)
vit_in = Input(shape=(1,), name='vit_input', dtype=tf.string)
subj_in = Input(shape=(1,), name='subject_input', dtype=tf.string)

cgpa_feats = cgpa_norm_layer(cgpa_in)

vit_stage1 = vit_lookup(vit_in)
vit_feats = vit_encoder(vit_stage1)

subject_stage1 = subject_lookup(subj_in)
subject_feats = subject_encoder(subject_stage1)

merged = concatenate([cgpa_feats, vit_feats, subject_feats])

dense_stack = Dense(128, activation='relu')(merged)
dense_stack = Dropout(0.20)(dense_stack)
dense_stack = Dense(64, activation='relu')(dense_stack)
dense_stack = Dropout(0.20)(dense_stack)
dense_stack = Dense(32, activation='relu')(dense_stack)
placement_out = Dense(1, activation='sigmoid', name='placement_output')(dense_stack)

placement_model = Model(
inputs=[cgpa_in, vit_in, subj_in],outputs=placement_out
)

placement_model.compile(
optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
loss='binary_crossentropy',
metrics=['accuracy']
)

placement_model.summary()

train_batches = {
'cgpa_input': cgpa_train_vals,
'vit_input': vit_train_raw,
'subject_input': subject_train_raw
}

training_history = placement_model.fit(
  train_batches,
  y_train,
  epochs=10,
  batch_size=256,
  validation_split=0.10,
  verbose=1
)

test_batches = {
'cgpa_input': cgpa_test_vals,
'vit_input': vit_test_raw,
'subject_input': subject_test_raw
}

test_loss, test_acc = placement_model.evaluate(test_batches, y_test, verbose=0)

print(f"Test Loss: {test_loss:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")

placement_model.save('placement_predictor_model.h5')
print("Model Saved")
