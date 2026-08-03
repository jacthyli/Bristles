#!/usr/bin/env python3
"""Generate a watertight ASCII STL of the bristle structure.

The STL coordinates are written in millimetres, matching the usual SolidWorks STL convention.
For OpenFOAM meshes stored in metres, scale the STL coordinates by 1e-3 before use.
"""
import math
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import triangulate

# Parameters extracted from makeblockMeshDict.py (micrometres before scaling)
RADIUS_UM = 1.0
NUM_BRISTLES = 7
BRISTLE_GAP_UM = RADIUS_UM * 2.0 * 6.0  # surface-to-surface gap = 12 um
BRISTLE_LENGTH_UM = 185.0
ROOT_HEIGHT_UM = 4.0
ROOT_WIDTH_UM = 6.0
CUBIC_WIDTH_UM = 106.0
CUBIC_LENGTH_UM = 300.0
ROOT_LENGTH_UM = (RADIUS_UM * 2.0 + BRISTLE_GAP_UM) * NUM_BRISTLES
SCALE = 1.0e-6  # micrometres -> metres
CIRCLE_SEGMENTS = 64


def bristle_centres_um():
    """Replicate the centre placement used by generate_vertices()."""
    centres = [(CUBIC_WIDTH_UM / 2.0, CUBIC_LENGTH_UM / 2.0)]
    count = (NUM_BRISTLES - 1) / 2.0
    pitch = BRISTLE_GAP_UM + 2.0 * RADIUS_UM
    for i in range(1, NUM_BRISTLES):
        offset = count * pitch
        direction = math.cos(i * math.pi)
        centres.append((CUBIC_WIDTH_UM / 2.0, CUBIC_LENGTH_UM / 2.0 + direction * offset))
        if i % 2 == 0:
            count -= 1.0
    return centres


def add_triangle(vertices, faces, pts):
    start = len(vertices)
    vertices.extend([list(p) for p in pts])
    faces.append([start, start + 1, start + 2])


def generate_mesh():
    centres_um = bristle_centres_um()

    x0_um = CUBIC_WIDTH_UM / 2.0 - ROOT_WIDTH_UM / 2.0
    x1_um = CUBIC_WIDTH_UM / 2.0 + ROOT_WIDTH_UM / 2.0
    y0_um = CUBIC_LENGTH_UM / 2.0 - ROOT_LENGTH_UM / 2.0
    y1_um = CUBIC_LENGTH_UM / 2.0 + ROOT_LENGTH_UM / 2.0

    x0, x1, y0, y1 = [v * SCALE for v in (x0_um, x1_um, y0_um, y1_um)]
    z0 = 0.0
    zb = ROOT_HEIGHT_UM * SCALE
    zt = (ROOT_HEIGHT_UM + BRISTLE_LENGTH_UM) * SCALE
    radius = RADIUS_UM * SCALE
    centres = [(x * SCALE, y * SCALE) for x, y in centres_um]

    # Exact shared circular boundary vertices for top plate holes and cylinder walls.
    circle_loops = []
    for cx, cy in centres:
        loop = [
            (cx + radius * math.cos(2.0 * math.pi * k / CIRCLE_SEGMENTS),
             cy + radius * math.sin(2.0 * math.pi * k / CIRCLE_SEGMENTS))
            for k in range(CIRCLE_SEGMENTS)
        ]
        circle_loops.append(loop)

    outer = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    # Shapely expects holes with opposite orientation; Polygon normalises them.
    plate_polygon = Polygon(shell=outer, holes=circle_loops)
    if not plate_polygon.is_valid:
        raise RuntimeError("Top plate polygon with bristle holes is invalid")

    vertices = []
    faces = []

    # Bottom face, normal -Z.
    add_triangle(vertices, faces, [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0)])
    add_triangle(vertices, faces, [(x0, y0, z0), (x1, y1, z0), (x1, y0, z0)])

    # Four vertical sides of root block, outward normals.
    side_quads = [
        # y = y0, outward -Y
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, zb), (x0, y0, zb)],
        # x = x1, outward +X
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, zb), (x1, y0, zb)],
        # y = y1, outward +Y
        [(x1, y1, z0), (x0, y1, z0), (x0, y1, zb), (x1, y1, zb)],
        # x = x0, outward -X
        [(x0, y1, z0), (x0, y0, z0), (x0, y0, zb), (x0, y1, zb)],
    ]
    for q in side_quads:
        add_triangle(vertices, faces, [q[0], q[1], q[2]])
        add_triangle(vertices, faces, [q[0], q[2], q[3]])

    # Root top surface with circular holes, normal +Z.
    for tri in triangulate(plate_polygon):
        # Keep only triangles fully contained in the polygon (excluding holes).
        if not plate_polygon.covers(tri):
            continue
        coords = list(tri.exterior.coords)[:3]
        p0, p1, p2 = [(float(x), float(y), zb) for x, y in coords]
        # Enforce +Z orientation.
        cross_z = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if cross_z < 0:
            p1, p2 = p2, p1
        add_triangle(vertices, faces, [p0, p1, p2])

    # Cylinders: side walls from base top to tip, plus top caps. No bottom caps.
    for (cx, cy), loop in zip(centres, circle_loops):
        for k in range(CIRCLE_SEGMENTS):
            k1 = (k + 1) % CIRCLE_SEGMENTS
            xk, yk = loop[k]
            xn, yn = loop[k1]
            # Outward-oriented side quad.
            q0 = (xk, yk, zb)
            q1 = (xn, yn, zb)
            q2 = (xn, yn, zt)
            q3 = (xk, yk, zt)
            add_triangle(vertices, faces, [q0, q1, q2])
            add_triangle(vertices, faces, [q0, q2, q3])

        # Top cap, normal +Z.
        centre_top = (cx, cy, zt)
        for k in range(CIRCLE_SEGMENTS):
            k1 = (k + 1) % CIRCLE_SEGMENTS
            p0 = centre_top
            p1 = (loop[k][0], loop[k][1], zt)
            p2 = (loop[k1][0], loop[k1][1], zt)
            add_triangle(vertices, faces, [p0, p1, p2])

    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=True)
    mesh.merge_vertices(digits_vertex=14)
    mesh.remove_unreferenced_vertices()
    # The faces above are constructed with explicit outward orientation.
    # multibody=True requires an optional graph backend (networkx or scipy)
    # in older trimesh versions, so avoid that dependency here.
    if not mesh.is_winding_consistent:
        mesh.fix_normals(multibody=False)

    if not mesh.is_watertight:
        raise RuntimeError(f"Generated mesh is not watertight; boundary edges: {len(mesh.edges_boundary)}")
    if not mesh.is_winding_consistent:
        raise RuntimeError("Generated mesh winding is inconsistent")
    return mesh




def move_base_centre_to_origin(mesh):
    """
    Shift the complete structure so that the geometric centre of the
    rectangular base is located at (0, 0, 0).

    Base dimensions and original position in millimetres:
        x: 0.050 to 0.056  -> centre 0.053
        y: 0.101 to 0.199  -> centre 0.150
        z: 0.000 to 0.004  -> centre 0.002
    """
    base_centre_m = np.array([53.0e-6, 150.0e-6, 2.0e-6], dtype=float)
    mesh.apply_translation(-base_centre_m)
    return mesh

def write_solidworks_ascii_stl(mesh, output_path, solid_name="Bristles"):
    """
    Write an ASCII STL using the same indentation and numeric style
    commonly produced by SolidWorks.

    Layout:
    solid Name
       facet normal x y z
          outer loop
             vertex x y z
             vertex x y z
             vertex x y z
          endloop
       endfacet
    endsolid
    """
    lines = ["solid {0}".format(solid_name)]

    normals = mesh.face_normals
    vertices = mesh.vertices
    faces = mesh.faces

    for face_index, face in enumerate(faces):
        normal = normals[face_index]
        lines.append(
            "   facet normal {0:.6e} {1:.6e} {2:.6e}".format(
                float(normal[0]), float(normal[1]), float(normal[2])
            )
        )
        lines.append("      outer loop")

        for vertex_index in face:
            vertex = vertices[int(vertex_index)]
            lines.append(
                "         vertex {0:.6e} {1:.6e} {2:.6e}".format(
                    float(vertex[0]), float(vertex[1]), float(vertex[2])
                )
            )

        lines.append("      endloop")
        lines.append("   endfacet")

    lines.append("endsolid")

    # Explicit CRLF line endings to closely match common SolidWorks output.
    with open(str(output_path), "wb") as stl_file:
        stl_file.write(("\r\n".join(lines) + "\r\n").encode("ascii"))

def main():
    output = Path('fluid/constant/triSurface/bristle.stl')
    mesh = generate_mesh()
    mesh = move_base_centre_to_origin(mesh)
    # Directly write SolidWorks-style ASCII STL.
    write_solidworks_ascii_stl(mesh, output, solid_name="Bristles")

    centres = bristle_centres_um()
    print(f"Wrote: {output}")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")
    print(f"Watertight: {mesh.is_watertight}")
    print(f"Winding consistent: {mesh.is_winding_consistent}")
    print(f"Euler number: {mesh.euler_number}")
    print("Volume [m^3]: {:.12e}".format(mesh.volume))
    print("Bounds [m]:\n{}".format(mesh.bounds))
    print("Bristle centres [um]:", sorted(centres, key=lambda p: p[1]))


if __name__ == '__main__':
    main()
