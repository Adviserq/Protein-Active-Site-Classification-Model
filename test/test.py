# import requests
# import time
# import aaindex
# LINK = 'https://www.ebi.ac.uk/thornton-srv/m-csa/api/residues/?format=json'

# index = aaindex.AAIndex1().amino_acids()
# index_2 = aaindex.aaindex1.get_all_categories()

# if isinstance(index_2, dict):
#     for k, v in index_2.items():
#         time.sleep(5)
#         print(k, v)
# print(index_2)
# r = requests.get(url = LINK, timeout=30)
# r.raise_for_status()

# data = r.json() or []
# print(type(data))

# result = {
#     'aa_name': None,
#     'function': None
#     }
# for element in data:
#     if element and isinstance(element, dict) and element.get('mcsa_id', 0):
#         first_element = next(iter(data))

#         if first_element:

#             for key, val in first_element.items():
#                 time.sleep(5)
#                 print(key, val, sep='\n')


AA_LIST = [
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'
]

AA_TO_INDEX = {aa:i for i,aa in enumerate(AA_LIST)}


print(AA_TO_INDEX)