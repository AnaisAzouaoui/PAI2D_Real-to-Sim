import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation.simulationGenesis import create_scene
from pipeline.itemSpec import getFilePath

if __name__ == "__main__":
    scene_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "scenes", "scene_generee.json")
    with open(scene_path, 'r', encoding='utf-8') as f:
        objetsList = json.load(f)
    for obj in objetsList:
        if not obj.get('path'):
            obj['path'] = getFilePath(obj)
    create_scene(objetsList)
