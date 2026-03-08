import requests
from typing import List, Dict, Tuple
import re
import random
from time import perf_counter

UNIPROT_SEARCH_URL = 'https://rest.uniprot.org/uniprotkb/search'
UNIPROT_SEARCH_ACCESS = 'https://rest.uniprot.org/uniprotkb/{}.json'


class UniProtClient:
    def __init__(self):
        pass

    def __str__(self):
        return f'Uniprot API Client'
        
    def search_homo_sapiens_with_active_site(self, size: int = 200):
        accessions = []
        search_query = 'organism_id:9606 AND ft_act_site:* AND structure_3d:true AND cc_disease:*'
        params = {
            'query': search_query,
            'format': 'json',
            'fields': 'accession'
        }

        start_time = perf_counter()
        next_url = None
        while True:
            if next_url:
                r = requests.get(next_url, timeout=30)
            else:
                r = requests.get(url=UNIPROT_SEARCH_URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            results = data.get('results') or []
            for item in results:
                primaryAccession = item.get('primaryAccession') or item.get('accession')
                if primaryAccession:
                    accessions.append(primaryAccession)

            next_url = None
            response_header_link = r.headers.get('Link')
            if response_header_link:
                link_m = re.search(r'<([^>]+)>\s*;\s*rel="next"', response_header_link)
                if link_m:
                    next_url = link_m.group(1)

            if not next_url:
                if isinstance(data, dict):
                    if data.get('next'):
                        next_url = data.get('next')
                    else:
                        links = data.get('links') or []
                        if isinstance(links, List):
                            for l in links:
                                if isinstance(l, dict) and (l.get('rel') == 'next' or l.get('relation') == 'next'):
                                    next_url = l.get('href') or l.get('url')
                                    break

            if not next_url:
                break
        end_time = perf_counter()
        result = accessions[:size]
        random.shuffle(result)
        return result

    def get_active_sites_and_pdbs(self, accession: str, resolution: float = 2.0) -> Dict:
        act_sites = []
        pdb_xrefs = []
        mutagenesis = []

        r = requests.get(url = UNIPROT_SEARCH_ACCESS.format(accession), timeout=30)
        r.raise_for_status()
        data = r.json()
        results = data.get('features') or []

        # Active sites and mutagenesis
        for feature in results:
            ftype = (feature.get('type') or '').lower()
            if 'active site' in ftype or ('active' in ftype and 'site' in ftype):
                location_act_site = feature.get('location') or {}
                pos = None
                location_val = location_act_site.get('start') or {}
                start_val = location_val.get('value') if isinstance(location_val, Dict) else None
                if start_val:
                    try:
                        pos = int(start_val)
                    except Exception:
                        pos = None
                description_field = feature.get('description', '') or ''
                act_sites.append({'Active Site Pos': pos, 'Active Site Desc': description_field})
            if 'mutagenesis' in ftype or 'mutagen' in ftype:
                loc = feature.get('location') or {}
                start = loc.get('start') or {}
                mpos = start.get('value') if isinstance(start, Dict) else None
                mutagenesis.append({'pos': mpos, 'description': feature.get('description')})

        # PDB xrefs
        xrefs = data.get('uniProtKBCrossReferences') or []
        for x_ref in xrefs:
            if x_ref.get('database') == 'PDB' or x_ref.get('type') == 'PDB':
                pdb_id = x_ref.get('id') or ''
                properties = x_ref.get('properties') or []
                prop_map = {}
                for prop in properties:
                    if not isinstance(prop, dict):
                        continue
                    key = (prop.get('key') or prop.get('name') or '').strip().lower()
                    val = prop.get('value') or ''
                    if key:
                        prop_map[key] = val

                res_value = None
                res_str = prop_map.get('resolution') or ''
                if res_str:
                    res_m = re.search(r"\d+(?:\.\d+)?", str(res_str))
                    if res_m:
                        try:
                            res_value = float(res_m.group())
                        except Exception:
                            res_value = None
                    if res_value is not None and res_value > resolution:
                        continue

                method = prop_map.get('method')
                pdb_id_clean = (pdb_id or '').strip()
                if pdb_id_clean:
                    pdb_entry = {'pdb_id': pdb_id_clean.upper(), 'resolution': res_value, 'method': method}
                    pdb_xrefs.append(pdb_entry)

        pdb_xrefs = [p for p in pdb_xrefs if p and p.get('pdb_id')]
        return {'active_sites': act_sites, 'pdbs': pdb_xrefs, 'mutagenesis': mutagenesis}


if __name__ == '__main__':
   
    client = UniProtClient()
    print(client)
    try:
        sample = client.search_homo_sapiens_with_active_site(size=10)
        # print('Sample accessions:', sample)
        if sample:
            # print('Fetching details for', sample[0])
            print(client.get_active_sites_and_pdbs('P00519'))
    except Exception as e:
        print(e)