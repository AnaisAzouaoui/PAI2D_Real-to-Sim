"""
Script lance en sous-processus par PlacementWorker (MAAAAC)
Genesis tourne dans le main thread de ce processus
Usage: run_build_scene.py <items_file> <relations_file> <orientations_file> <output_file>"
"""
import sys
import os
import json

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pipeline.sceneBuilding import buildScene

def main():
    items_file  = sys.argv[1]
    relations_file = sys.argv[2]
    orientations_file = sys.argv[3]
    output_file = sys.argv[4]
    with open(items_file) as f: items = json.load(f)
    with open(relations_file) as f: relations = json.load(f)
    with open(orientations_file) as f: orientations = json.load(f)
    result = buildScene(items, relations, orientations)
    with open(output_file, 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    main()
