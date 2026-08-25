import numpy as np
from Bio.PDB import NeighborSearch
from Bio.PDB.SASA import ShrakeRupley 

AA_LIST = [
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'
]

AA_TO_INDEX = {aa:i for i,aa in enumerate(AA_LIST)}

CHARGED = {'ASP','GLU','LYS','ARG','HIS'}
POLAR = {'SER','THR','ASN','GLN','HIS'}
HYDROPHOBIC = {'ALA','VAL','ILE','LEU','MET','PHE','TRP','PRO'}

_SASA_CACHE = {}


def one_hot_amino_acid(resname):
    vec = np.zeros(len(AA_LIST))
    if resname in AA_TO_INDEX:
        vec[AA_TO_INDEX[resname]] = 1.0
    return vec


# def compute_protein_centroid(model):
#     coords = []
#     for chain in model:
#         for residue in chain:
#             if residue.has_id('CA'):
#                 coords.append(residue['CA'].coord)
#     return np.mean(coords, axis=0)


def get_residue_sasa(model, residue):
    model_id = id(model)
    if model_id not in _SASA_CACHE:
        sr = ShrakeRupley()
        sr.compute(model, level='R')
        _SASA_CACHE[model_id] = True
    return float(getattr(residue, 'sasa', 0.0))


def extract_residue_features(model, chain, residue, radius=8.0):
    if not residue.has_id('CA'):
        return None

    central_ca = residue['CA']

    ca_atoms = []
    residue_map = {}

    for ch in model:
        for res in ch:
            if res.has_id('CA'):
                ca_atoms.append(res['CA'])
                residue_map[res['CA']] = (res, ch)

    ns = NeighborSearch(ca_atoms)
    neighbors = ns.search(central_ca.coord, radius)

    distances = []
    charged = 0
    polar = 0
    hydrophobic = 0

    for atom in neighbors:
        res, ch = residue_map[atom]
        if res == residue and ch.id == chain.id:
            continue
        dist = np.linalg.norm(atom.coord - central_ca.coord)
        distances.append(dist)
        if res.resname in CHARGED:
            charged += 1
        if res.resname in POLAR:
            polar += 1
        if res.resname in HYDROPHOBIC:
            hydrophobic += 1

    total_neighbors = len(distances)
    mean_dist = np.mean(distances) if distances else 0.0
    # std_dist = np.std(distances) if distances else 0.0

    # Disabled on request:
    # centroid = compute_protein_centroid(model)
    # dist_from_centroid = np.linalg.norm(central_ca.coord - centroid)

    sasa = get_residue_sasa(model, residue)

    one_hot = one_hot_amino_acid(residue.resname)

    numeric_features = np.array([
        total_neighbors,
        mean_dist,
        sasa
    ])

    return np.concatenate([one_hot, numeric_features])