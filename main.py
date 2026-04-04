import os
import shutil
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts.extract_pdb_proteins import UniProtClient
from scripts.build_residue_dataset import build_dataset
from scripts.read_mmCIFs import download_mmCIFs
from scripts.duplicates import clean_dublicates

class ActiveSitePipeline:
    def __init__(self, 
                 resolution_cutoff: float = 2.0,
                 temp_pdb_dir: str = "data/temp_pdb",
                 output_csv: str = "data/preprocessed/features_dataset.csv"):
        
        self.resolution = resolution_cutoff
        self.temp_pdb_dir = temp_pdb_dir
        self.output_csv = output_csv
        
        os.makedirs(self.temp_pdb_dir, exist_ok=True)
        # Διαγραφή παλιού αρχείου αν υπάρχει
        if os.path.exists(self.output_csv):
            os.remove(self.output_csv)
            
        self.uniprot_client = UniProtClient()

    def get_best_pdb_id(self, pdb_list):
        """Επιλογή PDB με το καλύτερο resolution."""
        valid_pdbs = [p for p in pdb_list if p.get('resolution') is not None]
        if not valid_pdbs:
            return pdb_list[0]['pdb_id'] if pdb_list else None
        return min(valid_pdbs, key=lambda x: x['resolution'])['pdb_id']

    def process_accession(self, accession):
        """Η κύρια μονάδα εργασίας για κάθε thread."""
        # Δημιουργία μοναδικού temp φακέλου για το thread για αποφυγή conflicts
        thread_dir = os.path.join(self.temp_pdb_dir, accession)
        os.makedirs(thread_dir, exist_ok=True)
        
        try:
            # 1. Fetch Metadata
            protein_data = self.uniprot_client.get_active_sites_and_pdbs(accession, self.resolution)

            if protein_data.get('skipped'):
                shutil.rmtree(thread_dir)
                return (f"SKIP: {accession} — mutagenesis overlaps active site "
                        f"at position {protein_data.get('conflict_pos')}")

            best_pdb_id = self.get_best_pdb_id(protein_data.get('pdbs', []))
            
            if not best_pdb_id:
                shutil.rmtree(thread_dir)
                return f"SKIP: {accession} (No PDB found)"

            # 2. Download (Best PDB only)
            download_mmCIFs([{'accession': accession, 'pdbs': [best_pdb_id]}], thread_dir)

            # 3. Build Dataset (Append Mode)
            active_pos_set = {site['Active Site Pos'] for site in protein_data.get('active_sites', [])}
            uniprot_label_dict = {best_pdb_id.upper(): active_pos_set}

            build_dataset(
                pdb_directory=thread_dir,
                uniprot_label_dict=uniprot_label_dict,
                output_csv=self.output_csv,
                append=True # Νέα παράμετρος για το thread-safety
            )
            
            # 4. Cleanup: Διαγραφή του mmCIF αμέσως μετά την επεξεργασία
            shutil.rmtree(thread_dir)
            return f"SUCCESS: {accession} processed with {best_pdb_id}"

        except Exception as e:
            if os.path.exists(thread_dir):
                shutil.rmtree(thread_dir)
            return f"ERROR: {accession} failed: {str(e)}"

def main():
    accessions = UniProtClient.search_homo_sapiens_with_active_site(UniProtClient, size=613)
    
    pipeline = ActiveSitePipeline()
    
    print(f"[*] Starting Parallel Processing with 8 workers...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(pipeline.process_accession, acc): acc for acc in accessions}
        
        for future in as_completed(futures):
            print(future.result())

    cleaned_dataframe = clean_dublicates(file_path = pipeline.output_csv)
    cleaned_dataframe.to_csv(pipeline.output_csv, index = False)

if __name__ == "__main__":
    main()