import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

file_name = r'C:\Main Directory\ACT_SITE_PROTEIN_CLASSIFICATION_MODEL\data\preprocessed\protein_data_cleaned.csv'
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
                            tf.keras.metrics.Recall()])

early_stopping = EarlyStopping(
    monitor='val_recall',    # Παρακολουθεί το recall του validation set
    patience=10,             # Υπομονή: αν για 10 epochs δεν δούμε βελτίωση, σταμάτα
    mode='max',              # Επειδή θέλουμε το ΜΕΓΙΣΤΟ recall
    restore_best_weights=True # ΠΟΛΥ ΣΗΜΑΝΤΙΚΟ: Επαναφέρει τα βάρη του καλύτερου epoch
)

def train_model(model, X, y, weights):
    print("\nΞεκινάει η εκπαίδευση...")
    history = model.fit(
        X, y,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        class_weight=weights,
        callbacks = [early_stopping],
        verbose=1)
    print("\nΤο μοντέλο εκπαιδεύτηκε και αποθηκεύτηκε!")
    model.save('protein_model_final.h5')

    return history


if os.path.exists('protein_model_final.h5') and os.path.exists('training_history.csv'):
    print('Model already trained')
    model = load_model('protein_model_final.h5')
    history_model = pd.read_csv('training_history.csv')

    hist_dict = history_model.to_dict(orient='list') # Convert data frame to dictonary
else: 
    history = train_model(model, X_train_scaled, y_train, class_weight_dict)
    
    pd.DataFrame(history.history).to_csv('training_history.csv', index=False)
    hist_dict = history.history

plt.figure(figsize=(12, 4))
plt.plot(hist_dict['loss'], label='Train Loss')
plt.plot(hist_dict['val_loss'], label='Val Loss')
plt.title('Loss History')
plt.xlabel('Number Of Epochs')
plt.ylabel('Loss Score')
plt.legend()
plt.show()