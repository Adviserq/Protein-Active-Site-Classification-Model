import pandas as pd
import matplotlib.pyplot as plt
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
    fig, ax = plt.subplots()
    ax.bar(class_labels.index, class_labels.values)
    ax.set_xticks(class_labels.index, ['Negative', 'Positive'])
    ax.set_xlabel('Class Label')
    ax.set_ylabel('Count')
    ax.set_title(
        f'Distribution of Class Labels / Positive Counts {class_labels[1]} '
        f'- Negative Counts {class_labels[0]}'
    )
    fig.tight_layout()
    fig.savefig('data/preprocessed/class_label_distribution.png', dpi=300)
    
    plt.show()
    plt.close(fig)
    return class_labels