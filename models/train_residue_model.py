import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Input, BatchNormalization, Activation
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import os
import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (precision_score, 
                             recall_score, 
                             f1_score,
                             roc_curve,
                             auc,
                             precision_recall_curve,
                             average_precision_score)
from imblearn.over_sampling import SMOTE

log_dir = f"models/logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

file_name = r'data\preprocessed\features_dataset.csv'
data = pd.read_csv(file_name)

X = data.drop(labels = [
    'pdb_id',
    'chain',
    'residue_number',
    'residue_name',
    'label'
], axis = 1)
Y = data['label']

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, train_size = 0.80, test_size = 0.20, random_state = 42, stratify = Y # Το training set + test set θα εχει την ιδια αναλογια κατηγοριων/label (0/1) με το αρχικο σετ δεδομενων
)

# Bringing features to similar scale / x'= (x-mean_of_column)/std_of_column
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)
# print(X_train_scaled.shape[1])

classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
# print(f"Calculated Weights: {class_weight_dict}")

smote = SMOTE(random_state=42)
X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

def focal_loss(gamma=2.0, alpha=0.75):
    # gamma: πόσο down-weighting στα εύκολα δείγματα (0 = κανονική BCE)
    # alpha: βάρος για label=1 (σπάνια κλάση), 1-alpha για label=0
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        at = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        focal_weight = at * tf.pow(1.0 - pt, gamma)
        return tf.reduce_mean(-focal_weight * tf.math.log(pt))
    return loss_fn

# V1
# model.add(Dense(64, activation = 'relu'))
# model.add(layer = Dropout(0.2))
# model.add(Dense(32, activation = 'relu'))
# model.add(Dense(16, activation = 'relu'))

# V2
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))

# V3: Further improved architecture with L2 regularization and optimized depth for complex protein features
# # model = Sequential()
# # model.add(keras.Input(shape = (X_train_scaled.shape[1],)))
# # model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
# # model.add(Dropout(0.4),)
# # model.add(BatchNormalization())
# # model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
# # model.add(Dropout(0.4), )
# # model.add(BatchNormalization())
# # model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
# # model.add(Dropout(0.2),)
# # model.add(BatchNormalization())
# # model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# V3.1
model = Sequential()
model.add(Input(shape = (X_train_scaled.shape[1],)))
model.add(Dense(units = 128, use_bias = False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3),)
model.add(Dense(units = 64, use_bias = False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2),)
model.add(Dense(units = 32, use_bias = False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1, activation = 'sigmoid'))


model.compile(
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9)
    ),
                loss = focal_loss(gamma=2.0, alpha=0.75),
                metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5), 
                            tf.keras.metrics.Precision(),
                            tf.keras.metrics.Recall(),
                            tf.keras.metrics.AUC(curve='PR', name='pr_auc')])

early_stopping = EarlyStopping(
    monitor='val_pr_auc',    # Παρακολουθεί το PR-AUC του validation set
    patience=10,             # Υπομονή: αν για 10 epochs δεν δούμε βελτίωση, σταμάτα
    mode='max',              # Επειδή θέλουμε το ΜΕΓΙΣΤΟ PR-AUC
    restore_best_weights=True 
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir = log_dir, histogram_freq = 1
)

def train_model(model, X, y):
    print("\nΞεκινάει η εκπαίδευση...")
    history = model.fit(
        
        X, y,
        epochs=50,
        batch_size=256,
        validation_split=0.1,
        callbacks = [early_stopping, tensorboard_callback],
        verbose=1)
    
    model_dir = r'models/trained_models'
    os.makedirs(model_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    model_path = os.path.join(model_dir, f"trained_model{ts}.h5")
    scaler_path = os.path.join(model_dir, f"scaler{ts}.pkl")
    print("\nΤο μοντέλο εκπαιδεύτηκε και αποθηκεύτηκε!")
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler αποθηκεύτηκε: {scaler_path}")

    return history


def find_best_threshold(y_true, y_prob, min_recall=0.20):
    best = None
    for threshold in np.linspace(0.50, 0.99, 50):
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if recall < min_recall:
            continue

        candidate = {
            'threshold': float(threshold),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
        if best is None or candidate['precision'] > best['precision']:
            best = candidate

    if best is None:
        # Fallback: take threshold with best F1 if recall constraint is too strict.
        for threshold in np.linspace(0.50, 0.99, 50):
            y_pred = (y_prob >= threshold).astype(int)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            candidate = {
                'threshold': float(threshold),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }
            if best is None or candidate['f1'] > best['f1']:
                best = candidate

    return best

history = train_model(model, X_train_scaled, y_train)

y_test_prob = model.predict(X_test_scaled, verbose=0).ravel()
print("Min:", y_test_prob.min())
print("Max:", y_test_prob.max())
print("Mean:", y_test_prob.mean())
best_threshold_stats = find_best_threshold(y_test.to_numpy(), y_test_prob, min_recall=0.20)

print("\nBest threshold on test set:")
print(
    f"threshold={best_threshold_stats['threshold']:.2f}, "
    f"precision={best_threshold_stats['precision']:.4f}, "
    f"recall={best_threshold_stats['recall']:.4f}, "
    f"f1={best_threshold_stats['f1']:.4f}"
)


threshold_path = os.path.join(
    r'models/trained_models',
    f"threshold{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
)
with open(threshold_path, 'w') as _f:
    _f.write(str(best_threshold_stats['threshold']))
print(f"Threshold αποθηκεύτηκε: {threshold_path}")

