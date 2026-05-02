import json
import os
from .jsonToSim import create_scene
from .validation_prompt import boucle_vlm_prompt
from .validation_img import boucle_vlm_img

def main():

    #test validation prompt:
    #path_json = os.path.join(os.path.dirname(__file__), 'exemple.json')
    #prompt = "Sur une table, je veux des ciseaux à droite d'un mug et une banane à sa gauche."
    #scene_finale = boucle_vlm_prompt(prompt, path_json)
    #create_scene(scene_finale)

    #test validation image:
    path_json = os.path.join(os.path.dirname(__file__), 'exemple_img.json')
    path_image = os.path.join(os.path.dirname(__file__), '..', '..', 'images/test1.png')
    scene_finale = boucle_vlm_img(path_image, path_json)
    create_scene(scene_finale)

    
if __name__=="__main__":
    main()