import genesis as gs
import os
import sys
from PIL import Image, ImageDraw, ImageFont #https://pillow.readthedocs.io/en/stable/
import copy

# rendre simulationGenesis importable depuis src/
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from simulation.simulationGenesis import mesh_to_urdf, patch_urdf


def prepare_path(obj):
    path = obj['path']
    if path.endswith('.urdf'):
        return patch_urdf(path)
    elif path.endswith('.xml'):
        xml_dir = os.path.dirname(path)
        for name in ["textured.obj", "nontextured.stl", "nontextured.ply"]:
            candidate = os.path.join(xml_dir, name)
            if os.path.exists(candidate):
                return mesh_to_urdf(candidate)
    else:
        return mesh_to_urdf(path)


def validation_physique(objetsList):
    steps = 150
    dt = 0.01
    corrected_objects = copy.deepcopy(objetsList)
    gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=dt))
    scene.add_entity(gs.morphs.Plane())
    cameras = {
        "perspective":scene.add_camera(res=(640, 480), pos=(3.5, 0.0, 2.5), lookat=(0, 0, 0.5), fov=30),
        "top":scene.add_camera(res=(640, 480), pos=(0.0, 0.0, 4.0), lookat=(0, 0, 0),   fov=40),
        "side":scene.add_camera(res=(640, 480), pos=(0.0, 3.5, 1.0), lookat=(0, 0, 0.5), fov=30),
        "side2": scene.add_camera(res=(640, 480),pos=(3.5, 0.0, 0.5),lookat=(0, 0, 0.5),fov=30)
    }
    entities = []
    for obj in corrected_objects:
        ent = scene.add_entity(
            gs.morphs.URDF(
                file=prepare_path(obj),
                pos=tuple(obj['pos']),
                quat=tuple(obj.get('quat', [0.0, 1.0, 1.0, 0.0])),
                scale=obj.get('scale', 1.0),
                fixed=False 
            ),
            material=gs.materials.Rigid(rho=1000)
        )
        entities.append(ent)
    scene.build()


    for i, ent in enumerate(entities):
        aabb_min, aabb_max = ent.get_AABB()
        corrected_objects[i]['lowest_point'] = float(aabb_min[2])
        corrected_objects[i]['highest_point'] = float(aabb_max[2])

        w = float(aabb_max[0] - aabb_min[0])
        d = float(aabb_max[1] - aabb_min[1])
        h = float(aabb_max[2] - aabb_min[2]) #les dimensions sont plus accurate
        corrected_objects[i]['dimensions'] = [w, d, h]
        z_min = float(aabb_min[2])
        if z_min < 0.0: #ça veut dire que l'objet est partiellement sous le sol
            correction = abs(z_min) + 0.001
            corrected_objects[i]['pos'][2] += correction
            corrected_objects[i]['lowest_point'] += correction
            corrected_objects[i]['highest_point'] += correction
            current_pos = ent.get_pos()
            ent.set_pos([current_pos[0], current_pos[1], current_pos[2] + correction])

    if len(entities) > 1:
        margin = 0.005
        aabb_mins = []
        aabb_maxs = []
        for ent in entities:
            aabb_min, aabb_max = ent.get_AABB()
            aabb_mins.append([float(aabb_min[0]), float(aabb_min[1]), float(aabb_min[2])])
            aabb_maxs.append([float(aabb_max[0]), float(aabb_max[1]), float(aabb_max[2])])

        def shift_object(index, axis, amount):
            corrected_objects[index]['pos'][axis] += amount
            aabb_mins[index][axis] += amount
            aabb_maxs[index][axis] += amount
            if axis == 2:
                corrected_objects[index]['lowest_point'] += amount
                corrected_objects[index]['highest_point'] += amount
            current_pos = entities[index].get_pos()
            new_pos = [float(current_pos[0]), float(current_pos[1]), float(current_pos[2])]
            new_pos[axis] += amount
            entities[index].set_pos(new_pos)

        max_passes = max(8, len(entities) * 3)
        for _ in range(max_passes):
            moved = False
            for i in range(len(entities) - 1):
                for j in range(i + 1, len(entities)):
                    overlaps = [
                        min(aabb_maxs[i][axis], aabb_maxs[j][axis]) -
                        max(aabb_mins[i][axis], aabb_mins[j][axis])
                        for axis in range(3)
                    ]
                    if overlaps[0] <= 0.0 or overlaps[1] <= 0.0 or overlaps[2] <= 0.0:
                        continue

                    axis = min(range(3), key=lambda k: overlaps[k])
                    centers_i = [(aabb_mins[i][k] + aabb_maxs[i][k]) * 0.5 for k in range(3)]
                    centers_j = [(aabb_mins[j][k] + aabb_maxs[j][k]) * 0.5 for k in range(3)]

                    if axis == 2:
                        target = j if centers_j[2] >= centers_i[2] else i
                        shift_object(target, 2, overlaps[2] + margin)
                    else:
                        direction = 1.0 if centers_j[axis] >= centers_i[axis] else -1.0
                        shift_object(j, axis, direction * (overlaps[axis] + margin))
                    moved = True
            if not moved:
                break

    z_init = [float(ent.get_pos()[2]) for ent in entities]
    for _ in range(steps):
        scene.step()

    z_final = [float(ent.get_pos()[2]) for ent in entities]
    for i, obj in enumerate(corrected_objects):
        delta_z = z_final[i] - z_init[i]
        v_z = delta_z / (steps * dt)
        if v_z > 0.5: #l'objet a été expulsé vers le haut, donc il était trop bas
            obj['pos'][2] += 0.05
            obj['lowest_point'] += 0.05
            obj['highest_point'] += 0.05
        elif delta_z < -0.001: #l'onbjet est tombe donc il etait trop haut
            obj['pos'][2] = z_final[i] #on met la hauteyr finale là où l'objet est retombé
            # final_quat = ent.get_quat()
            # obj['quat'] = [float(q) for q in final_quat]
  
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'images')
    os.makedirs(base_dir, exist_ok=True)

    views = ["perspective", "top", "side", "side2"]
    images = []

    try: 
        font = ImageFont.truetype("arial.ttf", 25)
    except: 
        font = ImageFont.load_default()
    
    for name in views:
        rgb, _, _, _ = cameras[name].render(rgb=True)
        images.append(Image.fromarray(rgb))
        draw = ImageDraw.Draw(images[-1])
        draw.text((10, 10), name, font=font, fill=(255, 255, 255))

    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    collage = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        collage.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    path_collage = os.path.abspath(os.path.join(base_dir, "collage_validation.png"))
    collage.save(path_collage)

    gs.destroy()
    return path_collage, corrected_objects



def create_scene(objetsList):
    '''
    Fonction qui permet de creer la scene sur genesis à partir des infos objenues précédemment.
    On commence par initialiser une scène vide, puis on rajoute les objets URDF.

    :param objets: une liste de dictionnaires des infos pour chaque objet (id, urdf, path, pos)
    '''
    gs.init(backend=gs.cpu)

    scene = gs.Scene(show_viewer=True) #,vis_options=gs.options.VisOptions(show_world_frame=True,show_link_frame=True))

    plane = scene.add_entity(gs.morphs.Plane()) #

    #ajout des objets: une boucle for qui prend l'objet, sa position, son filepath, et qui crée les objets un par un
    for obj in objetsList:
        entity = scene.add_entity(
            gs.morphs.URDF(
                file=obj['path'],
                pos=obj['pos'], 
                quat=obj.get('quat', [0.0, 1.0, 1.0, 0.0]),
                scale=obj.get('scale', 1.0),
                fixed=False
            ),
            material=gs.materials.Rigid(rho=1000),
        )

    scene.build()

    for i in range(1000):
        scene.step()
    gs.destroy()


# def create_scene_validation(objetsList, fixed=False):
#     '''
#     Fonction qui permet de creer la scene sur genesis à partir des infos objenues précédemment.
#     On commence par initialiser une scène vide, puis on rajoute les objets URDF.

#     :param objets: une liste de dictionnaires des infos pour chaque objet (id, urdf, path, pos, quat)
#     :return: le chemin vers le screenshot de la scène générée
#     '''

#     gs.init(backend=gs.cpu)

#     scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01), 
#                      viewer_options=gs.options.ViewerOptions(res=(640, 480)),
#                      vis_options=gs.options.VisOptions(show_world_frame=True,show_link_frame=True)
#                      )

#     plane = scene.add_entity(gs.morphs.Plane()) #la ground plaine

#     cameras = {
#         "perspective":scene.add_camera(res=(640, 480), pos=(3.5, 0.0, 2.5), lookat=(0, 0, 0.5), fov=30),
#         "top":scene.add_camera(res=(640, 480), pos=(0.0, 0.0, 4.0), lookat=(0, 0, 0),   fov=40),
#         "side":scene.add_camera(res=(640, 480), pos=(0.0, 3.5, 1.0), lookat=(0, 0, 0.5), fov=30),
#         "side2": scene.add_camera(res=(640, 480),pos=(3.5, 0.0, 0.5),lookat=(0, 0, 0.5),fov=30)
#     }


#     #ajout des objets: une boucle for qui prend l'objet, sa position, son filepath, et qui crée les objets un par un
#     for obj in objetsList:
#         entity = scene.add_entity(
#             gs.morphs.URDF(
#                 file=obj['path'],
#                 pos=tuple(obj['pos']),
#                 quat=tuple(obj['quat']),
#                 fixed=fixed
#             ),
#             material=gs.materials.Rigid(rho=1000)
#         )

#     scene.build()

#     #for i in range(150):
#     scene.step()

#     image_paths = []
#     base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'images')
#     os.makedirs(base_dir, exist_ok=True)

#     #for name, cam in cameras.items():
#     #    rgb, _, _, _ = cam.render(rgb=True)
#     #    img_path = os.path.abspath(os.path.join(base_dir, f'screenshot_{name}.png'))
#     #    Image.fromarray(rgb).save(img_path)
#     #    image_paths.append(img_path)

#     views = ["perspective", "top", "side", "side2"]
#     images = []


#     try: 
#         font = ImageFont.truetype("arial.ttf", 25)
#     except: 
#         font = ImageFont.load_default()

    
#     for name in views:
#         rgb, _, _, _ = cameras[name].render(rgb=True)
#         images.append(Image.fromarray(rgb))
#         draw = ImageDraw.Draw(images[-1])
#         draw.text((10, 10), name, font=font, fill=(255, 255, 255))

#     widths, heights = zip(*(i.size for i in images))
#     total_width = sum(widths)
#     max_height = max(heights)

#     collage = Image.new('RGB', (total_width, max_height))

#     x_offset = 0
#     for im in images:
#         collage.paste(im, (x_offset, 0))
#         x_offset += im.size[0]

#     path_collage = os.path.abspath(os.path.join(base_dir, "collage_validation.png"))
#     collage.save(path_collage)


#     gs.destroy()
#     return path_collage


# #get_pos, get_quat
# #regarder comment faire pour récupérer la velocité
# #q_pos: les positions angulaires, pour les joints
# # = pour vérifier si tous les trucs sont stables

# #entity.get_contact : donne les infos sur tous les contacts qui sont dans la scène
