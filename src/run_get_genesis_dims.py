"""
Script lance en sous-processus par ImageSceneWorker (c'est pour le mac again bare with me)
Genesis tourne dans le main thread de ce processus
run_get_genesis_dims.py <items_file> <orientations_file> <output_file>"
"""
import sys
import os
import json

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pipeline.sceneBuilding import get_genesis_dimensions


def main():
    items_file= sys.argv[1]
    orientations_file = sys.argv[2]
    output_file = sys.argv[3]
    with open(items_file)as f: items = json.load(f)
    with open(orientations_file) as f: orientations = json.load(f)
    result = get_genesis_dimensions(items, orientations)
    with open(output_file, 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    main()
