import cv2
import numpy as np

from transforms3d.euler import euler2mat
from pnp_test import draw_cuboid_2d, draw_axis_pose, get_rotate_mat, get_rotate


RED = (0,0,255)
BLUE = (255,0,0)
GREEN = (0,255,0)

def draw_cuboid_2d(img2, cuboid, color):
    assert len(cuboid) == 8
    points = [tuple(pt) for pt in cuboid]
    for ix in range(len(points)):
        pt = points[ix]
        cv2.putText(img2, "%d"%(ix), pt, cv2.FONT_HERSHEY_COMPLEX, 0.5, color)
        cv2.circle(img2, pt, 3, (0,255,0), -1)

    lines = [[0,1],[0,2],[1,3],[2,3],
         [4,5],[4,6],[5,7],[6,7],
         [0,4],[1,5],[2,6],[3,7]]
   
    for ix,line in enumerate(lines):


        pt1 = points[line[0]]
        pt2 = points[line[1]]
        if ix == 0:
            cv2.line(img2, pt1, pt2, (255,255,255))
        else:
            cv2.line(img2, pt1, pt2, color)

camera_matrix = np.array([[768.1605834960938, 0.0, 480.0], [0.0, 768.1605834960938, 270.0], [0.0, 0.0, 1.0]])
dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

im = np.zeros((540,960,3))

theeight = np.array([
    [1,1,1],  # top right, back
    [1,-1,1],  # bottom right, back
    [1,1,-1],  # top right, front
    [1,-1,-1],  # bottom right, front
    
    [-1,1,1],  # top left, back
    [-1,-1,1],  # bottom left, back
    [-1,1,-1],  # top left, front
    [-1,-1,-1],  # bottom left, front

    [0,0,0]  # centroid
], dtype=np.float32)

m = np.zeros(im.shape, dtype=np.uint8)
R45 = get_rotate([get_rotate_mat(20,0),get_rotate_mat(10,1)])#, get_rotate_mat(270,1)])#, get_rotate_mat(30,1)])
T = np.array([0,0,1.0])
imgpts, _ = cv2.projectPoints(theeight * 0.1, R45, T, camera_matrix, dist_coeffs)
m = draw_axis_pose(m, R45, T, camera_matrix, dist_coeffs)
# for ix, pt in enumerate(imgpts.squeeze()):
#     pt = tuple(pt)
#     cv2.circle(m, pt, 4, (0,0,255), -1)
#     cv2.putText(m, "%d"%(ix), pt, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255))
draw_cuboid_2d(m, imgpts.squeeze()[:-1], (255,0,0))
cv2.imshow("m", m)
# cv2.waitKey(0)

m = np.zeros(im.shape, dtype=np.uint8)
rotation_list = [get_rotate_mat(180,1)]#, get_rotate_mat(270,0)]
# rotation_list = [get_rotate_mat(180,0), get_rotate_mat(45,0)]
rotation = get_rotate(rotation_list, R45)
imgpts, _ = cv2.projectPoints(theeight * 0.1, rotation, T, camera_matrix, dist_coeffs)
m = draw_axis_pose(m, rotation, T, camera_matrix, dist_coeffs)
draw_cuboid_2d(m, imgpts.squeeze()[:-1], (0,0,255))

# cv2.imshow("m2", m)
# cv2.waitKey(0)

aa = {
    (1,2,3): [get_rotate_mat(0,0)], # default
    (1,-2,-3): [get_rotate_mat(180,0)],  # rotated 180 around x
    (1,-3,2): [get_rotate_mat(90,0)],   # rotated 90 clockwise around x
    (1,3,-2): [get_rotate_mat(270,0)],  # rotated 270 clockwise around x

    # start with -1 (rotate 180 around y from default)
    (-1,2,-3): [get_rotate_mat(180,1)],   # 
    (-1,-2,3): [get_rotate_mat(180,1), get_rotate_mat(180,0)],   # then rotate 180 around x
    (-1,3,2): [get_rotate_mat(180,1), get_rotate_mat(90,0)],   # then rotate 90 clockwise around x
    (-1,-3,-2): [get_rotate_mat(180,1), get_rotate_mat(270,0)],   # then rotate 270 clockwise around x

    # start with 2 (rotate 270 clockwise around z from default)
    (2,-1,3): [get_rotate_mat(270,2)],   # 
    (2,1,-3): [get_rotate_mat(270,2),get_rotate_mat(180,1)],   # then rotate 180 around y
    (2,3,1): [get_rotate_mat(270,2),get_rotate_mat(90,1)],   # then rotate 90 clockwise around y
    (2,-3,-1): [get_rotate_mat(270,2),get_rotate_mat(270,1)],   # then rotate 270 clockwise around y

    # start with -2 (rotate 90 clockwise around z from default)
    (-2,1,3): [get_rotate_mat(90,2)],   # 
    (-2,-1,-3): [get_rotate_mat(90,2),get_rotate_mat(180,1)],   # then rotate 180 around y
    (-2,-3,1): [get_rotate_mat(90,2),get_rotate_mat(90,1)],   # then rotate 90 clockwise around y
    (-2,3,-1): [get_rotate_mat(90,2),get_rotate_mat(270,1)],   # then rotate 270 clockwise around y

    # start with 3 (rotate 270 clockwise around y from default)
    (3,2,-1): [get_rotate_mat(270,1)],   # 
    (3,-2,1): [get_rotate_mat(270,1),get_rotate_mat(180,2)],   # then rotate 180 around z
    (3,-1,-2): [get_rotate_mat(270,1),get_rotate_mat(90,2)],   # then rotate 90 clockwise around z
    (3,1,2): [get_rotate_mat(270,1),get_rotate_mat(270,2)],   # then rotate 270 clockwise around z

    # start with -3 (rotate 90 clockwise around y from default)
    (-3,2,1): [get_rotate_mat(90,1)],   # 
    (-3,-2,-1): [get_rotate_mat(90,1),get_rotate_mat(180,2)],   # then rotate 180 around z
    (-3,-1,2): [get_rotate_mat(90,1),get_rotate_mat(90,2)],   # then rotate 90 clockwise around z
    (-3,1,-2): [get_rotate_mat(90,1),get_rotate_mat(270,2)],   # then rotate 270 clockwise around z
}


hashx = dict((tuple(v), ix) for ix, v in enumerate(theeight))
aa_order = {}
for k in aa:
    x = np.dot(get_rotate(aa[k]), theeight.T).T 
    x = np.round(x)[:-1]
    aa_order[k] = [hashx[tuple(v)] for v in x]
