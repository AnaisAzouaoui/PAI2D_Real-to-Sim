import os
from pipeline.sceneBuilding import buildScene
from pipeline.jsonParsing import readJSON
from boucle_validation.jsonToSim import create_scene

def main():
    path = os.path.join(os.path.dirname(__file__),'..', 'tests/test_json/test9.json')
    objects, relations, orientations = readJSON(path)
    print("Objects:", objects)
    print("Relations:", relations)
    print("Orientations:", orientations)
    items = buildScene(objects, relations, orientations)
    print("Items:", items)
    create_scene(items)

if __name__=="__main__":
    main()