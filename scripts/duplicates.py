import pandas as pd

file_path = r'C:\Main Directory\ACT_SITE_PROTEIN_CLASSIFICATION_MODEL\data\preprocessed\features_dataset.csv'

def clean_dublicates():
    df = pd.read_csv(file_path)
    print(f'Lines before removal of duplicates: {len(df)}')

    duplicates = df.duplicated().sum()
    print(f'Found {duplicates} duplicates')
    
    df_clean = df.drop_duplicates(subset = [
        'pdb_id', 'residue_number', 'chain'
    ], keep = 'first')
    return df_clean