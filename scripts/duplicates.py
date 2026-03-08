import pandas as pd

file_path = r'C:\Main Directory\ACT_SITE_PROTEIN_CLASSIFICATION_MODEL\data\preprocessed\protein_data_cleaned.csv'
df = pd.read_csv(file_path)

print(f'Lines before clean: {len(df)}')

duplicates = df.duplicated().sum()
print(f'Found {duplicates} duplicated rows')


df_final = df.drop_duplicates(subset=['pdb_id', 'residue_number', 'chain'], keep='first') # Χρηση composite key - συνδυαστικο κλειδι για να αφαιρεσουμε duplicates με βαση τα 3 πεδια, κρατωντας το πρωτο element
# df_final.to_csv('protein_data_cleaned.csv', index=False)

print(f"Labels {df_final['label'].value_counts()}")