from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
import requests


class CompareExecution:
    UNIPROT_SEARCH_URL = 'https://rest.uniprot.org/uniprotkb/search'
    UNIPROT_SEARCH_ACCESS = 'https://rest.uniprot.org/uniprotkb/{}.json'


    def __init__(self, accession: str):
        self.accession = accession

    def search_uniprot_entries_homo_sapiens_with_ative_site(self, size: int = 10) -> list:
        pass
    
    def get_protein_details(self, accession: str) -> dict:
        try:
            r = requests.get(url = self.UNIPROT_SEARCH_ACCESS.format(accession))
            r.raise_for_status()
            data = r.json()
            return data
        except Exception as e:
            print(e)
            return None
        
        
    def fetch_multiple_proteins_sequential(self, accessions: list) -> list:
        results = []

        for accession in accessions:
            result = self.get_protein_details(accession=accession)

            if result:
                results.append(result)

    def fetch_multiple_proteins_parallel(self, accessions: list) -> list:
        pass
    
    def __get__(self):
        return self.accession
    


my_obj = CompareExecution(accession='dsds')

print(my_obj)

#     class Protein:
#         def __init__(self, name):
#             self.name = name

#         def __str__(self): # Πως εμφανιζω το Object σαν κειμενο για τον ανθρωπο / Ορισμος δημοσιας αναπαραστασης object
#             return f"Protein: {self.name}"


#     class A:
#     def __str__(self):
#         return f"CLass name {self.__class__.__qualname__}"
    
# a = A()

# # print(str(a))
# # print(isinstance(a, object))
# # print(type(object))
# class RandomNumbers:
#     def __init__(self, a, b):
#         self.a = a
#         self.b = b

#     def __add__(self, other):
#         if isinstance(other, RandomNumbers):
#             return (self.a + self.b + other.a + other.b)

#     def __mul__(self, other):
#         if not isinstance(other, int): 
#             return NotImplemented 
        
#         return (self.a * other, self.b * other)

#     def __repr__(self):
#         return f'{self.__class__}'
# set_a = RandomNumbers(2, 4)
# set_b = RandomNumbers(3, 5)

# print(set_a)
'''
Η object είναι η θεμελιώδης βάση που δημιουργεί το instance μέσω __new__ - το κανει allocate στην μνημη, αλλά δεν δίνει αριθμητική συμπεριφορά — αυτή πρέπει να την ορίσεις εσύ με __add__.
__new__ = static method = δεν δουλευει σε υπαρχουν αντικεμενο αλλα το δημιουργει = 
'''