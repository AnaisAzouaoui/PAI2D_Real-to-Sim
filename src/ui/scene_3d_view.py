import os
import numpy as np
import trimesh
import xml.etree.ElementTree as ET
import pyqtgraph.opengl as gl
from scipy.spatial.transform import Rotation


FALLBACK_COLORS = [
    (0.10, 0.82, 1.00, 1.0),
    (0.18, 0.95, 0.38, 1.0),
    (1.00, 0.72, 0.05, 1.0),
    (1.00, 0.18, 0.48, 1.0),
    (0.82, 0.22, 1.00, 1.0),
    (0.05, 0.90, 0.82, 1.0),
    (1.00, 0.96, 0.12, 1.0),
    (1.00, 0.48, 0.10, 1.0),
]


def _get_vertex_colors(m, fallback_rgba):
    """Retourne les couleurs par vertex du mesh, ou la couleur de repli."""
    try:
        vc = m.visual.to_color().vertex_colors  # (N, 4) uint8
        if vc is not None and len(vc) == len(m.vertices):
            return vc.astype(np.float32) / 255.0
    except Exception:
        pass
    return np.tile(fallback_rgba, (len(m.vertices), 1)).astype(np.float32)


def _compute_link_transforms(root_elem):
    """Calcule les transformations cumulatives de chaque link depuis la racine URDF."""
    joint_map = {}
    for joint in root_elem.findall('joint'):
        child_link = joint.find('child').get('link')
        parent_link = joint.find('parent').get('link')
        origin = joint.find('origin')
        T = np.eye(4)
        if origin is not None:
            xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()]
            rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()]
            T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
            T[:3, 3] = xyz
        joint_map[child_link] = (parent_link, T)

    all_links = {link.get('name') for link in root_elem.findall('link')}
    root_links = all_links - set(joint_map.keys())
    root_link = next(iter(root_links)) if root_links else None

    link_transforms = {root_link: np.eye(4)} if root_link else {}

    parent_to_children = {}
    for child, (parent, T) in joint_map.items():
        parent_to_children.setdefault(parent, []).append((child, T))

    queue = [root_link] if root_link else []
    while queue:
        current = queue.pop(0)
        current_T = link_transforms[current]
        for child, joint_T in parent_to_children.get(current, []):
            link_transforms[child] = current_T @ joint_T
            queue.append(child)

    return link_transforms


def load_meshes_from_urdf(urdf_path):
    """Charge les meshes du URDF en appliquant la chaine cinematique complete.
    Retourne une liste de (trimesh.Trimesh, rgba_or_None) ou rgba_or_None est la
    couleur URDF si specifiee, sinon None (utiliser les couleurs du mesh).
    """
    tree = ET.parse(urdf_path)
    root_elem = tree.getroot()
    urdf_dir = os.path.dirname(urdf_path)

    link_transforms = _compute_link_transforms(root_elem)
    results = []

    for link in root_elem.findall('link'):
        link_name = link.get('name')
        link_T = link_transforms.get(link_name, np.eye(4))

        for visual in link.findall('visual'):
            geom = visual.find('geometry')
            if geom is None:
                continue
            mesh_tag = geom.find('mesh')
            if mesh_tag is None:
                continue
            
            box_tag = geom.find('box')
            if box_tag is not None:
                size = [float(v) for v in box_tag.get('size').split()]
                m = trimesh.creation.box(extents=size)
            else:
                mesh_tag = geom.find('mesh')
                if mesh_tag is None:
                    continue
                mesh_path = os.path.join(urdf_dir, mesh_tag.get('filename', ''))
                if not os.path.exists(mesh_path):
                    continue
                m = trimesh.load(mesh_path, force='mesh')
            mesh_path = os.path.join(urdf_dir, mesh_tag.get('filename', ''))
            if not os.path.exists(mesh_path):
                continue

            # Couleur URDF optionnelle
            urdf_rgba = None
            material = visual.find('material')
            if material is not None:
                color_elem = material.find('color')
                if color_elem is not None:
                    vals = color_elem.get('rgba', '').split()
                    if len(vals) == 4:
                        urdf_rgba = tuple(float(v) for v in vals)

            try:
                m = trimesh.load(mesh_path, force='mesh')
                vis_T = np.eye(4)
                origin = visual.find('origin')
                if origin is not None:
                    xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()]
                    rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()]
                    vis_T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
                    vis_T[:3, 3] = xyz
                m.apply_transform(link_T @ vis_T)

                # Si URDF donne une couleur, l'injecter dans le visual du mesh
                if urdf_rgba is not None:
                    rgba_u8 = (np.array(urdf_rgba) * 255).clip(0, 255).astype(np.uint8)
                    m.visual = trimesh.visual.ColorVisuals(
                        mesh=m,
                        face_colors=np.tile(rgba_u8, (len(m.faces), 1))
                    )

                results.append(m)
            except Exception:
                continue

    return results


def _make_checkerboard(n_tiles=50, tile_size=1.0, c1=(220, 225, 245), c2=(128, 132, 150)):
    half = n_tiles * tile_size / 2.0
    verts, faces, colors = [], [], []
    for i in range(n_tiles):
        for j in range(n_tiles):
            x0, x1 = -half + i * tile_size, -half + (i + 1) * tile_size
            y0, y1 = -half + j * tile_size, -half + (j + 1) * tile_size
            base = len(verts)
            verts += [[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]]
            faces += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
            col = c1 if (i + j) % 2 == 0 else c2
            rgba = [col[0] / 255.0, col[1] / 255.0, col[2] / 255.0, 1.0]
            colors += [rgba, rgba, rgba, rgba]
    return (np.array(verts, dtype=np.float32),
            np.array(faces, dtype=np.uint32),
            np.array(colors, dtype=np.float32))


class SceneView3D(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._bg_color = (22, 22, 35, 255)
        self.setBackgroundColor(self._bg_color)
        self.setCameraPosition(distance=5, elevation=30, azimuth=45)
        self._add_ground()

    def set_background(self, bg_color):
        self._bg_color = bg_color
        self.setBackgroundColor(bg_color)
        saved = getattr(self, '_last_objetsList', None)
        self.clear()
        self._add_ground()
        if saved:
            self.update_scene(saved)

    def _add_ground(self):
        avg = sum(self._bg_color[:3]) / 3
        if avg > 128:
            c1, c2 = (230, 215, 222), (185, 170, 178)
        else:
            c1, c2 = (220, 225, 245), (128, 132, 150)
        verts, faces, colors = _make_checkerboard(c1=c1, c2=c2)
        md = gl.MeshData(vertexes=verts, faces=faces, vertexColors=colors)
        ground = gl.GLMeshItem(meshdata=md, smooth=False, drawEdges=False, shader='shaded')
        self.addItem(ground)

    def update_scene(self, objetsList):
        """Charge et affiche les objets 3D a partir de la liste d'objets"""
        self._last_objetsList = objetsList
        self.clear()
        self._add_ground()

        for i, obj in enumerate(objetsList):
            path = obj.get('path', '')
            pos = obj.get('pos', (0, 0, 0))
            fallback = FALLBACK_COLORS[i % len(FALLBACK_COLORS)]
            scale = obj.get('scale', 1.0)
            quat = obj.get('quat', None)

            if path.endswith('.urdf'):
                meshes = load_meshes_from_urdf(path)
            elif path.endswith(('.obj', '.stl', '.ply')):
                m = trimesh.load(path, force='mesh')
                meshes = [m] if m is not None else []
            else:
                continue

            if quat is not None:
                rot_matrix = Rotation.from_quat(quat).as_matrix().astype(np.float32)
            else:
                rot_matrix = None

            for m in meshes:
                vertex_colors = _get_vertex_colors(m, fallback)
                # ambient lift so shaded faces stay visible
                vertex_colors[:, :3] = np.clip(vertex_colors[:, :3] * 1.35 + 0.15, 0, 1)
                verts = np.array(m.vertices, dtype=np.float32) * scale
                if rot_matrix is not None:
                    verts = verts @ rot_matrix.T
                faces = np.array(m.faces, dtype=np.uint32)
                md = gl.MeshData(vertexes=verts, faces=faces, vertexColors=vertex_colors)
                mesh_item = gl.GLMeshItem(
                    meshdata=md,
                    smooth=True,
                    drawEdges=False,
                    shader='shaded',
                )
                mesh_item.translate(pos[0], -pos[1], pos[2])
                self.addItem(mesh_item)
