import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from scripts.extract_pdb_proteins import UniProtClient
from scripts.build_residue_dataset import build_dataset
from scripts.read_mmCIFs import download_mmCIFs
from scripts.duplicates import clean_dublicates

class ActiveSitePipeline:
    def __init__(self, 
                 resolution_cutoff: float = 2.0,
                 raw_mmcif_dir: str = "data/raw_mmcifs",
                 output_csv: str = "data/preprocessed/features_dataset.csv"):
        
        self.resolution = resolution_cutoff
        self.raw_mmcif_dir = raw_mmcif_dir
        self.output_csv = output_csv
        self.processed_accessions_file = os.path.join(
            os.path.dirname(output_csv), "processed_accessions.txt"
        )
        self._accessions_lock = threading.Lock()
        
        os.makedirs(self.raw_mmcif_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)

        # Load already-preprocessed accessions to skip them on re-runs
        if os.path.exists(self.processed_accessions_file):
            with open(self.processed_accessions_file, 'r') as f:
                self.processed_accessions = set(line.strip() for line in f if line.strip())
        else:
            self.processed_accessions = set()
            
        self.uniprot_client = UniProtClient()

    def get_best_pdb_id(self, pdb_list):
        """Επιλογή PDB με το καλύτερο resolution."""
        valid_pdbs = [p for p in pdb_list if p.get('resolution') is not None]
        if not valid_pdbs:
            return pdb_list[0]['pdb_id'] if pdb_list else None
        return min(valid_pdbs, key=lambda x: x['resolution'])['pdb_id']

    def process_accession(self, accession):
        """Η κύρια μονάδα εργασίας για κάθε thread."""
        # Μόνιμος φάκελος για τα raw mmCIF αρχεία ανά accession
        accession_dir = os.path.join(self.raw_mmcif_dir, accession)
        os.makedirs(accession_dir, exist_ok=True)
        
        try:
            # 1. Fetch Metadata
            protein_data = self.uniprot_client.get_active_sites_and_pdbs(accession, self.resolution)

            if protein_data.get('skipped'):
                return (f"SKIP: {accession} — mutagenesis overlaps active site "
                        f"at position {protein_data.get('conflict_pos')}")

            best_pdb_id = self.get_best_pdb_id(protein_data.get('pdbs', []))
            
            if not best_pdb_id:
                return f"SKIP: {accession} (No PDB found)"

            # 2. Download to permanent raw dir (skipped if file already exists)
            download_mmCIFs([{'accession': accession, 'pdbs': [best_pdb_id]}], accession_dir)

            # 3. Build Dataset (Append Mode)
            active_pos_set = {site['Active Site Pos'] for site in protein_data.get('active_sites', [])}
            uniprot_label_dict = {best_pdb_id.upper(): active_pos_set}

            build_dataset(
                pdb_directory=accession_dir,
                uniprot_label_dict=uniprot_label_dict,
                output_csv=self.output_csv,
                append=True
            )

            # 4. Mark accession as preprocessed
            with self._accessions_lock:
                with open(self.processed_accessions_file, 'a') as f:
                    f.write(f"{accession}\n")
                self.processed_accessions.add(accession)

            return f"SUCCESS: {accession} processed with {best_pdb_id}"

        except Exception as e:
            return f"ERROR: {accession} failed: {str(e)}"

def main():
    accessions = UniProtClient.search_homo_sapiens_with_active_site(UniProtClient, size=613)
    
    pipeline = ActiveSitePipeline()

    remaining = [acc for acc in accessions if acc not in pipeline.processed_accessions]
    if len(remaining) < len(accessions):
        print(f"[*] Skipping {len(accessions) - len(remaining)} already-processed accessions.")

    if not remaining:
        print("[*] All accessions already processed. Dataset is up to date.")
        return
    
    print(f"[*] Starting Parallel Processing with 8 workers...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(pipeline.process_accession, acc): acc for acc in remaining}

        with tqdm(total=len(remaining), desc='Processing accessions', unit='acc', colour='green') as pbar:
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)
                pbar.write(result)

    cleaned_dataframe = clean_dublicates(file_path = pipeline.output_csv)
    cleaned_dataframe.to_csv(pipeline.output_csv, index = False)

if __name__ == "__main__":
    main()