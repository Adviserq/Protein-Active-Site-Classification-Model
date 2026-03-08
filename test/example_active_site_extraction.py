#!/usr/bin/env python3
'''
Example script: Extract active site neighbors from a single mmCIF file.

This demonstrates how to use the extract_active_site_neighbors function
to get neighboring amino acids around the active site of a protein.
'''

import json
from scripts.read_mmCIFs import (
    extract_active_site_neighbors,
    extract_active_site_with_uniprot_validation
)


def example_basic_extraction():
    '''Basic example: Extract neighbors from a single mmCIF file.'''
    
    print("="*70)
    print("Example 1: Basic Active Site Extraction")
    print("="*70)
    
    mmcif_file = r'C:\Main Directory\ACT_SITE_PROTEIN_CLASSIFICATION_MODEL\data\raw\pdb\1bbz.cif'
    
    print(f"\nExtracting active site neighbors from: {mmcif_file}")
    
    result = extract_active_site_neighbors(
        mmcif_file_path=mmcif_file,
        radius=5.0  # 5 angstroms
    )
    
    print(f"\nStatus: {result['status']}")
    print(f"Message: {result['message']}")
    
    if result['status'] == 'success':
        print(f"\nPDB ID: {result['pdb_id']}")
        
        # Central residue info
        central = result['central_residue']
        print(f"\nCentral Residue:")
        print(f"  Chain: {central['chain']}")
        print(f"  Residue ID: {central['residue_id']}")
        print(f"  Name: {central['residue_name']}")
        print(f"  Coordinates: {central['coordinates']}")
        
        # Neighbors info
        print(f"\nNumber of neighboring residues: {len(result['neighbors'])}")
        
        for i, neighbor in enumerate(result['neighbors'][:5], 1):  # Show first 5
            print(f"\n  Neighbor {i}:")
            print(f"    Chain: {neighbor['chain']}")
            print(f"    Residue: {neighbor['residue_name']} {neighbor['residue_id']}")
            print(f"    Distance to center: {neighbor['distance_to_center']:.2f} Å")
            print(f"    Atoms: {len(neighbor['atom_coordinates'])}")
            
            # Show atom coordinates
            for atom_name, atom_info in list(neighbor['atom_coordinates'].items())[:3]:
                print(f"      {atom_name}: {atom_info['coordinates']}")


def example_with_uniprot_validation():
    '''Example with UniProt validation.'''
    
    print("\n" + "="*70)
    print("Example 2: Active Site Extraction with UniProt Validation")
    print("="*70)
    
    mmcif_file = r'C:\Main Directory\ACT_SITE_PROTEIN_CLASSIFICATION_MODEL\data\raw\pdb\1bbz.cif'
    uniprot_id = 'P00439'  # Example UniProt ID
    
    print(f"\nExtracting with UniProt validation:")
    print(f"  mmCIF: {mmcif_file}")
    print(f"  UniProt ID: {uniprot_id}")
    
    result = extract_active_site_with_uniprot_validation(
        mmcif_file_path=mmcif_file,
        uniprot_id=uniprot_id,
        radius=5.0
    )
    
    print(f"\nStatus: {result['status']}")
    
    if 'uniprot_validation' in result:
        validation = result['uniprot_validation']
        print(f"\nUniProt Validation:")
        print(f"  Status: {validation.get('status')}")
        print(f"  Mapped position: {result.get('uniprot_mapped_position')}")
        print(f"  Is active site: {result.get('is_valid_active_site')}")
        
        if validation.get('status') == 'error':
            print(f"  Error: {validation.get('message')}")


def example_extract_coordinates_for_ml():
    '''Example: Extract coordinates suitable for ML model.'''
    
    print("\n" + "="*70)
    print("Example 3: Extract Coordinates for ML Model")
    print("="*70)
    
    mmcif_file = r'C:\Main Directory\ACT_SITE_PROTEIN_CLASSIFICATION_MODEL\data\raw\pdb\1bbz.cif'
    
    result = extract_active_site_neighbors(mmcif_file, radius=5.0)
    
    if result['status'] == 'success':
        print(f"\nExtracting coordinates for ML model...")
        
        # Collect all atom coordinates
        all_coordinates = []
        
        for neighbor in result['neighbors']:
            for atom_name, atom_info in neighbor['atom_coordinates'].items():
                coord = atom_info['coordinates']
                element = atom_info['element']
                
                all_coordinates.append({
                    'atom': f"{neighbor['residue_name']}{neighbor['residue_id']}{atom_name}",
                    'element': element,
                    'x': coord[0],
                    'y': coord[1],
                    'z': coord[2]
                })
        
        print(f"\nTotal atoms extracted: {len(all_coordinates)}")
        
        # Convert to format suitable for tensors
        import numpy as np
        coords_array = np.array([
            [c['x'], c['y'], c['z']] for c in all_coordinates
        ])
        
        print(f"Coordinates array shape: {coords_array.shape}")
        print(f"Center of mass: {coords_array.mean(axis=0)}")
        print(f"Std dev: {coords_array.std(axis=0)}")


if __name__ == '__main__':
    # Check if mmCIF files exist
    import os
    pdb_dir = r'C:\Main Directory\ACT_SITE_PROTEIN_CLASSIFICATION_MODEL\data\raw\pdb'
    
    if not os.path.exists(pdb_dir):
        print(f"PDB directory not found: {pdb_dir}")
        print("Make sure to download mmCIF files first using main.py")
        exit(1)
    
    # Run examples
    try:
        example_basic_extraction()
        example_with_uniprot_validation()
        example_extract_coordinates_for_ml()
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)
