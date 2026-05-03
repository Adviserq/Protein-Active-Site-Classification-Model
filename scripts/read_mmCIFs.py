import os
import requests
from Bio.PDB import PDBList


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


RCSB_MMCIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"


def _download_single(pdb_id: str, output_directory: str) -> None:
    dest = os.path.join(output_directory, f"{pdb_id.lower()}.cif")
    if os.path.exists(dest):
        return  # already downloaded, skip
    url = RCSB_MMCIF_URL.format(pdb_id=pdb_id.upper())
    r = requests.get(url, timeout=60, stream=True)
    r.raise_for_status()
    with open(dest, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1 << 16):
            f.write(chunk)


def download_mmCIFs(entries, output_directory: str, method: str = "BioPython"):
    '''
    Download mmCIF files for given `entries` list produced by
    `get_pdb_entries`.
    '''
    if output_directory is None or output_directory.strip() == '':
        raise ValueError('An output directory is required')

    for element in entries:
        if not element or not isinstance(element, dict):
            continue
        pdb_ids = element.get('pdbs', [])
        for pdb in pdb_ids:
            try:
                if method == "HTTP":
                    _download_single(pdb, output_directory)
                elif method == "BioPython":
                    pdb_obj = PDBList()
                    pdb_obj.retrieve_pdb_file(
                        pdb_code=pdb,
                        pdir=output_directory,
                        file_format='mmCif',
                        overwrite=True
                    )
                else:
                    raise ValueError(f"Unknown download method: {method}")
                print(f'Downloaded {pdb}')
            except Exception as e:
                print(f"Error downloading {pdb}: {e}")
