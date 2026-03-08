import os
import sys
import traceback

from scripts.read_mmCIFs import extract_near_and_far, _json_fallback


# Change this path if you want to test a different mmCIF file
mmcif_path = os.path.join('data', 'raw', 'pdb', '1bbz.cif')
out_dir = os.path.join('outputs', 'near_far')


def main():
    try:
        print(f"Running extract_near_and_far on: {mmcif_path}")
        res = extract_near_and_far(mmcif_path, near_radius=5.0, far_min_radius=10.0)
        print(res.get('status'), res.get('message'))

        # write full near+far to a JSON file for inspection
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(mmcif_path).split('.')[0] + '_near_far.json')
        import json
        with open(out_path, 'w', encoding='utf-8') as fh:
            json.dump(res, fh, default=_json_fallback)
        print(f"Wrote output to {out_path}")

    except Exception as e:
        traceback.print_exc()
        print('Error during extraction:', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
