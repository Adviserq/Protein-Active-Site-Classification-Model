import os

from Bio.PDB import PDBList
try:
    from Bio.PDB.MMCIFParser import MMCIFParser
except Exception:
    MMCIFParser = None


output_dir = r'C:\Main Directory\ACT_SITE_PROTEIN_CLASSIFICATION_MODEL\data\raw\pdb'

# residue-centric helpers are available in the scripts package.  Importing
# them here is not required by the download logic but provides a convenient
# place to remind users of the extended functionality.
# from scripts.build_residue_dataset import build_dataset


def get_pdb_entries(extractor, size: int = 100) -> list[dict]:
    '''
    Given an extractor (instance of `ExtractProteins` or compatible
    object), list PDB cross-references for a set of accessions.

    The function expects `extractor` to implement `list_accessions(size)`
    and `fetch_entry(accession)`.
    '''
    if extractor is None:
        raise ValueError('extractor instance is required')

    accessions = extractor.list_accessions(size=size)
    entries = []

    for accession in accessions:
        if not accession:
            continue
        entry_data = extractor.fetch_entry(accession=accession)
        pdbs = entry_data.get('pdbs', []) if isinstance(entry_data, dict) else []

        if isinstance(pdbs, list):
            pdb_ids = []
            for element in pdbs:
                if not element or not isinstance(element, dict):
                    continue
                pdb_id = element.get('pdb_id')
                resolution = element.get('resolution')
                # keep only entries with an id and a resolution value
                if pdb_id and resolution is not None:
                    pdb_ids.append(pdb_id)

            if pdb_ids:
                entries.append({'accession': accession, 'pdbs': pdb_ids})

    return entries


def download_mmCIFs(entries, output_directory: str = None):
    '''
    Download mmCIF files for given `entries` list produced by
    `get_pdb_entries`.
    '''
    if output_directory is None:
        output_directory = output_dir

    pdb_obj = PDBList()
    for element in entries:
        if not element or not isinstance(element, dict):
            continue
        pdb_ids = element.get('pdbs', [])
        for pdb in pdb_ids:
            try:
                pdb_obj.retrieve_pdb_file(
                    pdb_code=pdb,
                    pdir=output_directory,
                    file_format='mmCif',
                    overwrite=True
                )
                print(f'Downloaded {pdb}')
            except Exception as e:
                print(f"Error downloading {pdb}: {e}")