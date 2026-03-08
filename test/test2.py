import requests
from Bio.PDB import MMCIFParser
import os

UNIPROT_SEARCH_URL = 'https://rest.uniprot.org/uniprotkb/search'
UNIPROT_SEARCH_ACCESS = 'https://rest.uniprot.org/uniprotkb/{}.json'


accessions = []
search_query = 'organism_id:9606 AND ft_act_site:* AND structure_3d:true'

params = {
    'query': search_query,
    'fields': 'accession',
    'format': 'json'
}

def first_accession():
    r = requests.get(
        url = UNIPROT_SEARCH_URL,
        params = params
    )
    r.raise_for_status()
    data = r.json()

    result_list = data.get('results') or []
    for item in result_list:
        if item and isinstance(item, dict):
            primaryAccession = item.get('primaryAccession') or ''

            if primaryAccession == 'A1L3X0':
                request_entry = requests.get(
                    url = UNIPROT_SEARCH_ACCESS.format(primaryAccession),
                    timeout = 30
                ); r.raise_for_status()
                entry_data = request_entry.json()
                future_results = entry_data.get('features') or []
                protein_sequence = entry_data.get('sequence') or {}
                for future in future_results:
                    if isinstance(future, dict):
                        ftype = (future.get('type') or '').lower()
                        if ftype.startswith('Active') or ftype.startswith('active') or ftype.startswith('site'):
                            location_act_site = future.get('location') or {}
                            pos = None
                            location_act_site_val = location_act_site.get('start') or {}
                            start_val = location_act_site_val.get('value') if isinstance(location_act_site_val, dict) else None

                            if start_val:
                                try:
                                    pos = int(start_val)
                                except Exception:
                                    pos = None

                                # print(pos)

                protein_seq_str = protein_sequence.get('value') or ''
                return (pos, protein_seq_str[pos])
            


aa_pos = first_accession()

print(aa_pos)