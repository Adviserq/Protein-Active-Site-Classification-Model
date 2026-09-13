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

FUNCTIONAL_ATOM = {
    'SER': 'OG', 'THR': 'OG1', 'CYS': 'SG', 'TYR': 'OH',
    'HIS': 'NE2', 'LYS': 'NZ', 'ARG': 'CZ', 'ASN': 'ND2',
    'GLN': 'NE2', 'MET': 'SD', 'TRP': 'NE1',
    # Για ASP/GLU το CG είναι προσέγγιση του centroid μεταξύ OD1/OD2 ή OE1/OE2.
    'ASP': 'CG', 'GLU': 'CG'
}

_SASA_CACHE = {}


def one_hot_amino_acid(resname):
    vec = np.zeros(len(AA_LIST))
    if resname in AA_TO_INDEX:
        vec[AA_TO_INDEX[resname]] = 1.0
    return vec


def compute_wcn(central_ca, ca_atoms):
    """Υπολογίζει το WCN σύμφωνα με τους Chien & Huang (2012), εξίσωση (5)."""
    wcn = 0.0
    for ca_atom in ca_atoms:
        if ca_atom is central_ca:
            continue
        distance = np.linalg.norm(ca_atom.coord - central_ca.coord)
        if distance > 0:
            wcn += 1 / distance ** 2
    return wcn


def compute_sidechain_orientation_angle(residue, central_ca, neighbors, residue_map):
    """Υπολογίζει τη γωνία μεταξύ side-chain vector και neighbor-centroid vector.

    Βασίζεται στους Chien & Huang (2012), ενότητα "Definition of Side Chain
    Vector", με θ = arccos(v · u / (|v| |u|)) σε μοίρες.
    """
    functional_atom_name = FUNCTIONAL_ATOM.get(residue.resname)
    if not functional_atom_name or not residue.has_id(functional_atom_name):
        # ALA, VAL, LEU, ILE, PHE, PRO και GLY δεν έχουν σαφές functional atom
        # πλευρικής αλυσίδας στο άρθρο (διάκριση Phase 1 έναντι Phase 2).
        return -1.0

    sidechain_vector = residue[functional_atom_name].coord - central_ca.coord
    neighbor_ca_coords = [
        atom.coord for atom in neighbors
        if residue_map[atom][0] != residue
    ]
    if not neighbor_ca_coords:
        return -1.0

    neighbor_centroid = np.mean(neighbor_ca_coords, axis=0)
    neighbor_vector = neighbor_centroid - central_ca.coord
    sidechain_norm = np.linalg.norm(sidechain_vector)
    neighbor_norm = np.linalg.norm(neighbor_vector)
    if sidechain_norm == 0 or neighbor_norm == 0:
        return -1.0

    cosine = np.dot(sidechain_vector, neighbor_vector) / (sidechain_norm * neighbor_norm)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


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

    wcn = compute_wcn(central_ca, ca_atoms)
    mean_dist = np.mean(distances) if distances else 0.0
    # std_dist = np.std(distances) if distances else 0.0

    # Disabled on request:
    # centroid = compute_protein_centroid(model)
    # dist_from_centroid = np.linalg.norm(central_ca.coord - centroid)

    sasa = get_residue_sasa(model, residue)
    sidechain_orientation_angle = compute_sidechain_orientation_angle(
        residue, central_ca, neighbors, residue_map
    )

    one_hot = one_hot_amino_acid(residue.resname)

    numeric_features = np.array([
        wcn,
        mean_dist,
        sasa,
        sidechain_orientation_angle
    ])

    return np.concatenate([one_hot, numeric_features])