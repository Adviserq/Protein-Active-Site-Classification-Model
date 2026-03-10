import os
import csv
import threading
from Bio.PDB.MMCIFParser import MMCIFParser
from scripts.feature_extractor import extract_residue_features
from scripts.duplicates import clean_dublicates

# Global lock για το γράψιμο στο CSV από πολλά threads
csv_lock = threading.Lock()

def build_dataset(pdb_directory, uniprot_label_dict, output_csv, radius=8.0, append=False):
    parser = MMCIFParser(QUIET=True)
    mode = 'a' if append else 'w'

    for filename in os.listdir(pdb_directory):
        if not filename.lower().endswith(('.cif', '.mmcif')):
            continue

        pdb_id = filename.split('.')[0].upper()
        # Τα BioPython downloads συχνά έχουν όνομα pdbXXXX.ent ή XXXX.cif
        if pdb_id.startswith('PDB'): pdb_id = pdb_id[3:]
        
        file_path = os.path.join(pdb_directory, filename)
        structure = parser.get_structure(pdb_id, file_path)
        model = structure[0]

        # Τα active residues είναι ένα set από θέσεις (residue numbers)
        active_residue_positions = uniprot_label_dict.get(pdb_id, set())

        rows_to_write = []
        for chain in model:
            for residue in chain:
                if not residue.has_id('CA'):
                    continue

                features = extract_residue_features(model, chain, residue, radius)
                if features is None:
                    continue

                residue_number = residue.id[1]
                # Mapping: Ελέγχουμε αν ο αριθμός θέσης είναι στο set του UniProt
                label = 1 if residue_number in active_residue_positions else 0

                rows_to_write.append([
                    pdb_id, chain.id, residue_number, residue.resname,
                    *features.tolist(), label
                ])

        # Thread-safe εγγραφή στο αρχείο
        with csv_lock:
            file_exists = os.path.isfile(output_csv)
            with open(output_csv, mode, newline='') as f:
                writer = csv.writer(f)
                
                # Γράψε header μόνο αν το αρχείο είναι καινούργιο
                if not file_exists or os.stat(output_csv).st_size == 0:
                    header = ['pdb_id','chain','residue_number','residue_name'] + \
                             [f'f{i}' for i in range(len(rows_to_write[0])-5)] + ['label']
                    writer.writerow(header)
                
                writer.writerows(rows_to_write)

    cleaned_csv = clean_dublicates()
    cleaned_csv.to_csv('features_dataset.csv', index = False)