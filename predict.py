"""
predict.py — Πρόβλεψη ενεργού κέντρου σε άγνωστη πρωτεΐνη.

Χρήση:
    python predict.py --input path/to/protein.cif
    python predict.py --input path/to/protein.cif --model models/trained_models/trained_modelXXXX.h5 --scaler models/trained_models/scalerXXXX.pkl --threshold 0.92
    python predict.py --input path/to/protein.cif --top 10   # εμφάνισε μόνο top-10 πιο πιθανά residues

Αν δεν συμπληρωθούν --model / --scaler / --threshold, διαλέγεται αυτόματα το πιο πρόσφατο αρχείο
από τον φάκελο models/trained_models/.
"""

import argparse
import glob
import os
import sys
import requests

import joblib
import numpy as np
import pandas as pd
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBParser
from keras.models import load_model

# Για να βρεθεί το feature_extractor ανεξαρτήτως από πού τρέχεις το script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.feature_extractor import extract_residue_features


# ─── helpers ────────────────────────────────────────────────────────────────

def _latest_file(directory: str, pattern: str) -> str | None:
    """Επιστρέφει το πιο πρόσφατο αρχείο που ταιριάζει στο pattern."""
    matches = sorted(glob.glob(os.path.join(directory, pattern)), reverse=True)
    return matches[0] if matches else None


def infer_pdb_id_from_path(file_path: str) -> str | None:
    stem = os.path.splitext(os.path.basename(file_path))[0].upper()
    if len(stem) >= 4 and stem[:4].isalnum():
        return stem[:4]
    return None


def fetch_sifts_uniprot_mappings(pdb_id: str) -> dict:
    """Fetch SIFTS mappings from PDBe for a PDB id."""
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        print(f"[WARN] SIFTS mapping failed for {pdb_id}: {exc}")
        return {}


def build_sifts_lookup(pdb_id: str) -> list[dict]:
    """
    Build interval-based lookup records from SIFTS JSON.
    Each entry maps a chain/auth residue interval to UniProt interval.
    """
    data = fetch_sifts_uniprot_mappings(pdb_id)
    root = data.get(pdb_id.lower(), {})
    uniprot = root.get('UniProt', {})
    entries = []

    for unp_acc, payload in uniprot.items():
        for m in payload.get('mappings', []):
            chain_id = m.get('chain_id')
            start = m.get('start', {})
            end = m.get('end', {})

            pdb_start = start.get('residue_number')
            pdb_end = end.get('residue_number')
            unp_start = m.get('unp_start')
            unp_end = m.get('unp_end')

            if None in (chain_id, pdb_start, pdb_end, unp_start, unp_end):
                continue

            entries.append({
                'chain': str(chain_id),
                'pdb_start': int(pdb_start),
                'pdb_end': int(pdb_end),
                'unp_start': int(unp_start),
                'unp_end': int(unp_end),
                'unp_acc': unp_acc
            })

    return entries


def map_residue_to_uniprot(chain: str, residue_number: int, sifts_entries: list[dict]):
    """Map a PDB residue (auth chain + auth seq id) to UniProt position using SIFTS ranges."""
    for e in sifts_entries:
        if e['chain'] != chain:
            continue
        if e['pdb_start'] <= residue_number <= e['pdb_end']:
            pdb_span = e['pdb_end'] - e['pdb_start']
            unp_span = e['unp_end'] - e['unp_start']
            # Use only strict 1:1 interval mappings to avoid wrong UniProt positions.
            if abs(pdb_span) != abs(unp_span):
                continue
            offset = residue_number - e['pdb_start']
            if pdb_span == 0:
                unp_pos = e['unp_start']
            else:
                unp_pos = e['unp_start'] + int(round(offset * (unp_span / pdb_span)))
            return e['unp_acc'], int(unp_pos)
    return None, None


def load_structure(file_path: str):
    """Φορτώνει δομή από .cif/.mmcif ή .pdb/.ent αρχείο."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ('.cif', '.mmcif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure_id = os.path.splitext(os.path.basename(file_path))[0]
    return parser.get_structure(structure_id, file_path)


def extract_all_features(structure) -> list[dict]:
    """
    Εξάγει features για κάθε residue που έχει Cα.
    Επιστρέφει list από dicts με μεταδεδομένα + feature vector.
    """
    model = structure[0]
    records = []

    for chain in model:
        residue_index_in_chain = 0
        for residue in chain:
            if not residue.has_id('CA'):
                continue
            feat = extract_residue_features(model, chain, residue)
            if feat is None:
                continue
            residue_index_in_chain += 1
            insertion_code = residue.id[2].strip() if residue.id[2] != ' ' else ''
            records.append({
                'chain':          chain.id,
                'residue_number': residue.id[1],
                'insertion_code': insertion_code,
                'residue_index_in_chain': residue_index_in_chain,
                'residue_name':   residue.resname,
                'features':       feat
            })

    return records


def predict(records: list[dict], model, scaler, threshold: float, sifts_entries: list[dict] | None = None) -> pd.DataFrame:
    X = np.array([r['features'] for r in records])
    X_scaled = scaler.transform(X)
    probs = model.predict(X_scaled, verbose=0).ravel()

    rows = []
    for rec, prob in zip(records, probs):
        unp_acc, unp_pos = (None, None)
        if sifts_entries:
            unp_acc, unp_pos = map_residue_to_uniprot(
                chain=rec['chain'],
                residue_number=rec['residue_number'],
                sifts_entries=sifts_entries
            )

        rows.append({
            'chain':            rec['chain'],
            'residue_number':   rec['residue_number'],
            'insertion_code':   rec['insertion_code'],
            'residue_index_in_chain': rec['residue_index_in_chain'],
            'residue_name':     rec['residue_name'],
            'uniprot_acc':      unp_acc,
            'uniprot_pos':      unp_pos,
            'probability':      round(float(prob), 4),
            'predicted_active': int(prob >= threshold)
        })

    df = pd.DataFrame(rows)
    df.sort_values('probability', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Πρόβλεψη ενεργού κέντρου πρωτεΐνης από mmCIF/PDB δομή.'
    )
    parser.add_argument('--input',     required=True,  help='Αρχείο δομής (.cif ή .pdb)')
    parser.add_argument('--model',     default=None,   help='Μοντέλο .h5 (default: πιο πρόσφατο)')
    parser.add_argument('--scaler',    default=None,   help='Scaler .pkl (default: πιο πρόσφατο)')
    parser.add_argument('--threshold', default=None,   type=float,
                        help='Threshold πιθανότητας (default: πιο πρόσφατο .txt ή 0.92)')
    parser.add_argument('--top',       default=None,   type=int,
                        help='Εμφάνισε μόνο τα N residues με τη μεγαλύτερη πιθανότητα')
    parser.add_argument('--pdb-id',    default=None,
                        help='PDB ID για SIFTS mapping (default: inference από file name)')
    parser.add_argument('--no-sifts',  action='store_true',
                        help='Απενεργοποίηση SIFTS mapping')
    parser.add_argument('--output',    default=None,
                        help='Αποθήκευση αποτελεσμάτων σε CSV (προαιρετικό)')
    args = parser.parse_args()

    trained_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'models', 'trained_models')

    # ── Φόρτωση μοντέλου ──
    model_path = args.model or _latest_file(trained_dir, 'trained_model*.h5')
    if not model_path or not os.path.exists(model_path):
        sys.exit('[ERROR] Δεν βρέθηκε trained model. Τρέξε πρώτα train_residue_model.py.')
    print(f'[*] Model: {model_path}')
    nn_model = load_model(model_path, compile=False)

    # ── Φόρτωση scaler ──
    scaler_path = args.scaler or _latest_file(trained_dir, 'scaler*.pkl')
    if not scaler_path or not os.path.exists(scaler_path):
        sys.exit('[ERROR] Δεν βρέθηκε scaler. Τρέξε ξανά train_residue_model.py (αποθηκεύει scaler).')
    print(f'[*] Scaler: {scaler_path}')
    scaler = joblib.load(scaler_path)

    # ── Threshold ──
    if args.threshold is not None:
        threshold = args.threshold
    else:
        thresh_file = _latest_file(trained_dir, 'threshold*.txt')
        if thresh_file and os.path.exists(thresh_file):
            with open(thresh_file) as f:
                threshold = float(f.read().strip())
        else:
            threshold = 0.92
    print(f'[*] Threshold: {threshold}')

    # ── Φόρτωση δομής & εξαγωγή features ──
    print(f'[*] Φόρτωση δομής: {args.input}')
    structure = load_structure(args.input)
    print('[*] Εξαγωγή features...')
    records = extract_all_features(structure)
    if not records:
        sys.exit('[ERROR] Δεν βρέθηκαν residues με Cα άτομο στη δομή.')
    print(f'[*] Residues που αναλύθηκαν: {len(records)}')

    # ── SIFTS mapping (PDB -> UniProt) ──
    sifts_entries = None
    if not args.no_sifts:
        pdb_id = args.pdb_id or infer_pdb_id_from_path(args.input)
        if pdb_id:
            print(f'[*] SIFTS mapping για PDB: {pdb_id}')
            sifts_entries = build_sifts_lookup(pdb_id)
            print(f'[*] SIFTS intervals loaded: {len(sifts_entries)}')
        else:
            print('[WARN] Δεν μπόρεσα να συμπεράνω PDB id από το filename. Χρησιμοποίησε --pdb-id XXXX')

    # ── Πρόβλεψη ──
    results = predict(records, nn_model, scaler, threshold, sifts_entries=sifts_entries)

    # ── Εμφάνιση ──
    predicted_active = results[results['predicted_active'] == 1]
    print(f'\n{"─"*60}')
    print(f'  Προβλεπόμενα residues ενεργού κέντρου  (threshold={threshold})')
    print(f'{"─"*60}')

    if predicted_active.empty:
        print('  (Κανένα residue πάνω από το threshold)')
    else:
        print(predicted_active.to_string(index=False))

    # Top-N αν ζητήθηκε
    if args.top:
        print(f'\n{"─"*60}')
        print(f'  Top-{args.top} residues με τη μεγαλύτερη πιθανότητα')
        print(f'{"─"*60}')
        print(results.head(args.top).to_string(index=False))

    # ── Αποθήκευση ──
    if args.output:
        results.to_csv(args.output, index=False)
        print(f'\n[*] Αποτελέσματα αποθηκεύτηκαν: {args.output}')


if __name__ == '__main__':
    main()
