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
from sklearn.model_selection import train_test_split, GroupShuffleSplit
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

file_name = 'data/preprocessed/features_dataset_wcn.csv'
data = pd.read_csv(file_name)

X = data.drop(labels = [
    'pdb_id',
    'chain',
    'residue_number',
    'residue_name',
    'label'
], axis = 1)
Y = data['label']

groups = data['pdb_id']

outer_gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_val_idx, test_idx = next(outer_gss.split(X, Y, groups=groups))

inner_gss = GroupShuffleSplit(n_splits=1, train_size=0.9, random_state=43)
train_relative_idx, validation_relative_idx = next(
    inner_gss.split(
        X.iloc[train_val_idx],
        Y.iloc[train_val_idx],
        groups=groups.iloc[train_val_idx]
    )
)
train_idx = train_val_idx[train_relative_idx]
validation_idx = train_val_idx[validation_relative_idx]

x_train = X.iloc[train_idx]
x_validation = X.iloc[validation_idx]
x_test = X.iloc[test_idx]
y_train = Y.iloc[train_idx]
y_validation = Y.iloc[validation_idx]
y_test = Y.iloc[test_idx]

train_proteins = set(groups.iloc[train_idx])
validation_proteins = set(groups.iloc[validation_idx])
test_proteins = set(groups.iloc[test_idx])

train_validation_overlap = train_proteins & validation_proteins
train_test_overlap = train_proteins & test_proteins
validation_test_overlap = validation_proteins & test_proteins
print(
    f"Πρωτεΐνες σε train: {len(train_proteins)} | "
    f"validation: {len(validation_proteins)} | test: {len(test_proteins)}"
)
print(
    f"Επικαλύψεις train/validation: {len(train_validation_overlap)} | "
    f"train/test: {len(train_test_overlap)} | "
    f"validation/test: {len(validation_test_overlap)}"
)
assert not train_validation_overlap, "Data leakage: κοινές πρωτεΐνες σε train και validation set!"
assert not train_test_overlap, "Data leakage: κοινές πρωτεΐνες σε train και test set!"
assert not validation_test_overlap, "Data leakage: κοινές πρωτεΐνες σε validation και test set!"

print(
    f"Αναλογία θετικής κλάσης — train: {y_train.mean():.4f} | "
    f"validation: {y_validation.mean():.4f} | test: {y_test.mean():.4f}"
)

# x_train, x_test, y_train, y_test = train_test_split(
#     X, Y, train_size = 0.80, test_size = 0.20, random_state = 42, stratify = Y # Το training set + test set θα εχει την ιδια αναλογια κατηγοριων/label (0/1) με το αρχικο σετ δεδομενων
# )

# Bringing features to similar scale / x'= (x-mean_of_column)/std_of_column
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_validation_scaled = scaler.transform(x_validation)
X_test_scaled = scaler.transform(x_test)
# print(X_train_scaled.shape[1])

classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
# print(f"Calculated Weights: {class_weight_dict}")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_scaled,
    y_train
)

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
# Η είσοδος παραμένει δυναμική και προσαρμόζεται αυτόματα στα 24 features.
model.add(Input(shape = (X_train_resampled.shape[1],)))
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
    mode='max',              
    restore_best_weights=True 
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir = log_dir, histogram_freq = 1
)

def train_model(model, X, y, validation_data):
    print("\nΞεκινάει η εκπαίδευση...")
    history = model.fit(
        
        X, y,
        epochs=50,
        batch_size=256,
        validation_data=validation_data,
        callbacks = [early_stopping, tensorboard_callback],
        verbose=1)
    
    model_dir = r'models/trained_models'
    scaler_dir = r'models/scalers'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    model_path = os.path.join(model_dir, f"trained_model{ts}.h5")
    scaler_path = os.path.join(scaler_dir, f"scaler{ts}.pkl")
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

history = train_model(
    model,
    X_train_resampled,
    y_train_resampled,
    validation_data=(X_validation_scaled, y_validation)
)

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
    r'models/thresholds',
    f"threshold{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
)
os.makedirs(os.path.dirname(threshold_path), exist_ok=True)
with open(threshold_path, 'w') as _f:
    _f.write(str(best_threshold_stats['threshold']))
print(f"Threshold αποθηκεύτηκε: {threshold_path}")