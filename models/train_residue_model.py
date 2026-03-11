import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import os
import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score

log_dir = f"models/logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

file_name = r'C:\Main Directory\ACT_SITE_PROTEIN_CLASSIFICATION_MODEL\data\preprocessed\features_dataset.csv'
data = pd.read_csv(file_name)
# Traning dataset: Συνολο δεδομενων που χρησιμοποιειται για την εκπαιδευση του μοντελου / αξιοποιειται για την προσαρμογη του μοντελου / Το μοντελο τα βλεπει, μαθαινει απο αυτα και αναπροσαρμοζει τις παραμετρους του
# Test set: Υποσυνολο του training dataset (το μοντελο δεν εχει δει το υποσυνολο αυτο), το οποιο χρησιμοποιειται για την τελικη αξιολογηση αποδοσης του μοντελου / Σκοπος ειναι να παρεχει μια αμεροληπτη εκτιμηση ικανοτητας γενικευσης σε νεα αγνωστα δεδομενα
# Γιατι χωρις stratify μπορει η αναλογια των labels στο τεστ σετ να αλλαξει? Διοτι το split γινεται τυχαια σε επιπεδο δειγματων και οχι label/κατηγοριων
X = data.drop(labels = [
    'pdb_id',
    'chain',
    'residue_number',
    'residue_name',
    'label'
], axis = 1)
Y = data['label']

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, train_size = 0.20, random_state = 42, stratify = Y # Το training set + test set θα εχει την ιδια αναλογια κατηγοριων/label (0/1) με το αρχικο σετ δεδομενων
)

# Bringing features to similar scale / x'= (x-mean_of_column)/std_of_column
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)
# print(X_train_scaled.shape[1])


weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: weights[0], 1: weights[1]}
# print(f"Calculated Weights: {class_weight_dict}")

model = Sequential()
model.add(keras.Input(shape = (X_train_scaled.shape[1],)))
model.add(Dense(64, activation = 'relu'))
model.add(layer = Dropout(0.2))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam',
                loss = 'binary_crossentropy',
                metrics = ['accuracy', 
                            tf.keras.metrics.Precision(),
                            tf.keras.metrics.Recall(),
                            tf.keras.metrics.AUC(curve='PR', name='pr_auc')])

early_stopping = EarlyStopping(
    monitor='val_pr_auc',    # Παρακολουθεί το PR-AUC του validation set
    patience=10,             # Υπομονή: αν για 10 epochs δεν δούμε βελτίωση, σταμάτα
    mode='max',              # Επειδή θέλουμε το ΜΕΓΙΣΤΟ PR-AUC
    restore_best_weights=True # ΠΟΛΥ ΣΗΜΑΝΤΙΚΟ: Επαναφέρει τα βάρη του καλύτερου epoch
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir = log_dir, histogram_freq = 1
)

def train_model(model, X, y, weights):
    print("\nΞεκινάει η εκπαίδευση...")
    history = model.fit(
        X, y,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        class_weight=weights,
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

history = train_model(model, X_train_scaled, y_train, class_weight_dict)

y_test_prob = model.predict(X_test_scaled, verbose=0).ravel()
best_threshold_stats = find_best_threshold(y_test.to_numpy(), y_test_prob, min_recall=0.20)

print("\nBest threshold on test set:")
print(
    f"threshold={best_threshold_stats['threshold']:.2f}, "
    f"precision={best_threshold_stats['precision']:.4f}, "
    f"recall={best_threshold_stats['recall']:.4f}, "
    f"f1={best_threshold_stats['f1']:.4f}"
)

# Αποθήκευση threshold για χρήση στο predict.py
threshold_path = os.path.join(
    r'models/trained_models',
    f"threshold{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
)
with open(threshold_path, 'w') as _f:
    _f.write(str(best_threshold_stats['threshold']))
print(f"Threshold αποθηκεύτηκε: {threshold_path}")

