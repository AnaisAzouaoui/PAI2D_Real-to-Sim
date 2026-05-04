import os
import trimesh
import xml.etree.ElementTree as ET


def getFilePath(item):
    '''
    Donne le path pour l'item URDF.

    :param item: le dictionnaire d'item avec id et urdf
    :return path: le path pour l'item
    '''
    items_folder = os.path.join(os.path.dirname(__file__),'..', '..', 'objets')
    base = os.path.join(items_folder, item['urdf'])
    if not os.path.exists(base):
        raise FileNotFoundError(f"Pour {item['id']}, le fichier {item['urdf']} est introuvable.")
    #path = addMass(path)

    #TODO: question: pourquoi on cherche d'autres fichiers que mobility.urdf?
    # si c'est un dossier on cherche un fichier qui existe
    if os.path.isdir(base):
        # objets PartNet : fichiers directement dans le dossier
        for name in ["table.urdf","mobility.urdf", "kinbody.xml", "textured.obj", "nontextured.stl", "nontextured.ply"]:
            path = os.path.join(base, name)
            if os.path.exists(path):
                return path

        # objets YCB : fichiers dans le sous-dossier google_16k
        google_16k = os.path.join(base, "google_16k")
        if os.path.isdir(google_16k):
            for name in ["textured.obj", "nontextured.stl", "nontextured.ply"]:
                path = os.path.join(google_16k, name)
                if os.path.exists(path):
                    return path

        """# fallback : n'importe quel URDF dans le dossier
        for f in os.listdir(base):
            if f.endswith(".urdf"):
                return os.path.join(base, f)"""
        
        raise FileNotFoundError(f"Aucun fichier exploitable trouvé dans {base}")

    return base


def getOriginalDimensions(item):
    '''
    Extrait les dimensions de l'item depuis son fichier (URDF ou mesh direct).
    '''
    path = item['path']

    # mesh direct : pas de XML, on prend directement la bounding box
    if path.endswith(('.obj', '.stl', '.ply')):
        mesh = trimesh.load(path, force='mesh')
        bounds = mesh.bounding_box.extents
        item['dimensions'] = (float(bounds[0]), float(bounds[1]), float(bounds[2]))
        return item

    tree = ET.parse(path)
    root = tree.getroot()

    for geometry in root.iter('geometry'):
        box= geometry.find('box')
        if box is not None:
            size = box.attrib['size'].split()
            item['dimensions'] = (float(size[0]),float(size[1]), float(size[2]))
            return item
        
        cylinder= geometry.find('cylinder')
        if cylinder is not None:
            r = float(cylinder.attrib['radius'])
            length = float(cylinder.attrib['length'])
            item['dimensions'] = (r*2, r*2, length)
            return item
        
        sphere= geometry.find('sphere')
        if sphere is not None:
            r = float(sphere.attrib['radius'])
            item['dimensions'] = (r*2, r*2, r*2)
            return item
        
        mesh_tag = geometry.find('mesh') #quand les formes sont plus complexes c'est généralement mesh qui est utilisé
        if mesh_tag is not None:
            directory = os.path.dirname(item['path'])
            path_mesh = os.path.join(directory, mesh_tag.attrib['filename'])
            mesh = trimesh.load(path_mesh, force='mesh') #force='mesh' est pour ne pas avoir de scene, seulement un mesh unique
            bounds = mesh.bounding_box.extents
            item['dimensions'] = (float(bounds[0]),float(bounds[1]),float(bounds[2]))
            return item

    print("Dimensions non trouvées") #TODO: faire une meilleure erreur
    item['dimensions'] = (0.1,0.1,0.1)
    return item


def processScale(item): #TODO: is it going to be un indice de multiplication? are we adding a scale attribute to the item, that we can then give to the simulation?
    #TODO: check what scale actually does in the genesis simul: it seems like it makes it do weird stuff
    '''
    Gets the dimension of the item and compares them to the dimensions that the item needs to have in the simulation.
    Defines the scale at which the object will be displayed. Give new dimensions to the items so that positions can be processed accordingly.
    '''
    (item_w, item_d, item_h) = item['dimensions']
    if item['mesures']:
        (new_w, new_d, new_h) = item['mesures'] #TODO: 'mesures' à rajouter au prompt ou autre, c'est les mesures de l'objet ajoutées par l'utilisateur 
        item['scale'] # = scale such that item * scale = new
    return item