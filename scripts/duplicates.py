import pandas as pd
 
def clean_dublicates(file_path: str):
    df = pd.read_csv(file_path)
    print(f'Lines before removal of duplicates: {len(df)}')

    duplicates = df.duplicated().sum()
    print(f'Found {duplicates} duplicates')
    
    df_clean = df.drop_duplicates(subset = [
        'pdb_id', 'residue_number', 'chain'
    ], keep = 'first')
    return df_clean

def view_class_labels(file_path: str):
    df = pd.read_csv(file_path)

    class_labels = df['label'].value_counts()
    # print(class_labels)
    return class_labels

test = view_class_labels(r'data\preprocessed\features_dataset.csv')
print(test)