import os
from tqdm import tqdm
from scripts.extract_pdb_proteins import UniProtClient
from scripts.build_residue_dataset import build_dataset
from scripts.duplicates import clean_dublicates


class DatasetRebuilder:
    def __init__(self,
                 resolution_cutoff: float = 2.0,
                 raw_mmcif_dir: str = "data/raw_mmcifs",
                 output_csv: str = "data/preprocessed/features_dataset_wcn.csv"):
        self.resolution = resolution_cutoff
        self.raw_mmcif_dir = raw_mmcif_dir
        self.output_csv = output_csv
        self.processed_accessions_file = os.path.join(
            os.path.dirname(output_csv), "processed_accessions.txt"
        )

        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        self.uniprot_client = UniProtClient()

    def get_best_pdb_id(self, pdb_list):
        """Επιλογή PDB με το καλύτερο resolution."""
        valid_pdbs = [p for p in pdb_list if p.get('resolution') is not None]
        if not valid_pdbs:
            return pdb_list[0]['pdb_id'] if pdb_list else None
        return min(valid_pdbs, key=lambda x: x['resolution'])['pdb_id']

    def find_cif_file(self, accession, pdb_id):
        accession_dir = os.path.join(self.raw_mmcif_dir, accession)
        if not os.path.isdir(accession_dir):
            return None

        expected_pdb_id = pdb_id.lower()
        for filename in os.listdir(accession_dir):
            if not filename.lower().endswith(('.cif', '.mmcif')):
                continue
            file_pdb_id = filename.rsplit('.', 1)[0].lower()
            if file_pdb_id.removeprefix('pdb') == expected_pdb_id:
                return os.path.join(accession_dir, filename)
        return None

    def process_accession(self, accession):
        """Η κύρια μονάδα εργασίας για κάθε accession."""
        try:
            # 1. Fetch Metadata
            protein_data = self.uniprot_client.get_active_sites_and_pdbs(accession, self.resolution)

            if protein_data.get('skipped'):
                return (f"SKIP: {accession} — mutagenesis overlaps active site "
                        f"at position {protein_data.get('conflict_pos')}")

            best_pdb_id = self.get_best_pdb_id(protein_data.get('pdbs', []))
            if not best_pdb_id:
                return f"SKIP: {accession} (No PDB found)"

            # 2. Find the existing raw mmCIF file
            cif_file = self.find_cif_file(accession, best_pdb_id)
            if not cif_file:
                return f"SKIP: {accession} (No local CIF found for {best_pdb_id})"

            # 3. Build Dataset (Append Mode)
            active_pos_set = {site['Active Site Pos'] for site in protein_data.get('active_sites', [])}
            uniprot_label_dict = {best_pdb_id.upper(): active_pos_set}

            build_dataset(
                pdb_directory=os.path.dirname(cif_file),
                uniprot_label_dict=uniprot_label_dict,
                output_csv=self.output_csv,
                append=True
            )

            return f"SUCCESS: {accession} processed with {best_pdb_id}"

        except Exception as e:
            return f"ERROR: {accession} failed: {str(e)}"


def main():
    pipeline = DatasetRebuilder()

    with open(pipeline.processed_accessions_file, 'r') as f:
        accessions = [line.strip() for line in f if line.strip()]

    print(f"[*] Starting Dataset Rebuild with {len(accessions)} accessions...")
    with tqdm(total=len(accessions), desc='Processing accessions', unit='acc', colour='green') as pbar:
        for accession in accessions:
            result = pipeline.process_accession(accession)
            pbar.update(1)
            pbar.write(result)

    cleaned_dataframe = clean_dublicates(file_path=pipeline.output_csv)
    cleaned_dataframe.to_csv(pipeline.output_csv, index=False)


if __name__ == "__main__":
    main()