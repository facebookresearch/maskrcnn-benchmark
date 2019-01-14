"""
https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
"""
import cv2
import numpy as np
import open3d

from conv_test import create_cloud, backproject_camera
from transforms3d.quaternions import quat2mat#, mat2quat

 
def draw_cuboid_2d(img2, cuboid, color):
    assert len(cuboid) == 8
    points = [tuple(pt) for pt in cuboid]
    for ix in range(len(points)):
        pt = points[ix]
        cv2.putText(img2, "%d"%(ix), pt, cv2.FONT_HERSHEY_COMPLEX, 0.4, color)
        cv2.circle(img2, pt, 3, (0,255,0), -1)

    lines = [[0,1],[0,2],[1,3],[2,3],
         [4,5],[4,6],[5,7],[6,7],
         [0,4],[1,5],[2,6],[3,7]]

    for line in lines:
        pt1 = points[line[0]]
        pt2 = points[line[1]]
        cv2.line(img2, pt1, pt2, color)


points_file = "./datasets/FAT/points_all_orig.npy"
points = np.load(points_file)

# Read Image
im_file = "./datasets/FAT/data/mixed/temple_0/000295.right.jpg"
depth_file = im_file.replace(".jpg",".depth.png")


intrinsics = np.array([[768.1605834960938, 0.0, 480.0], [0.0, 768.1605834960938, 270.0], [0.0, 0.0, 1.0]])
factor_depth=10000

im = cv2.imread(im_file)
# im = np.zeros((800,800,3), dtype=np.uint8) #cv2.imread("headPose.jpg")
H,W,_ = im.shape
cls = 9

# load depth and cloud
depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

rgb = im.copy()[:,:,::-1]
if rgb.dtype == np.uint8:
    rgb = rgb.astype(np.float32) / 255
X = backproject_camera(depth, intrinsics, factor_depth)
scene_cloud = create_cloud(X.T, colors=rgb.reshape((H*W,3)))

model_points = np.array([
    [-0.05080691, -0.041769,   -0.02880055],
    [-0.05080691,  0.04176 ,   -0.02880055],
    [-0.05080691, -0.041769,    0.02880038],
    [-0.05080691,  0.04176 ,    0.02880038],
    [ 0.0508229 , -0.041769,   -0.02880055],
    [ 0.0508229 ,  0.04176 ,   -0.02880055],
    [ 0.0508229 , -0.041769,    0.02880038],
    [ 0.0508229 ,  0.04176 ,    0.02880038],
    [ 0,           0,           0         ] # add centroid
])
# model_points += np.random.normal(0, scale=0.01, size=model_points.shape)
# model_points *= 0.9

# image_points = np.array([
#     [569, 299],
#     [526, 309],
#     [589, 329],
#     [546, 337],
#     [538, 348],
#     [495, 355],
#     [561, 379],
#     [517, 384]
# ], dtype=np.float32)
image_points = np.array([
    [575, 304],
    [530, 311],
    [580, 332],
    [540, 332],
    [533, 350],
    [496, 350],
    [564, 383],
    [514, 389],
    [540, 343]  # centroid
], dtype=np.float32)

camera_matrix = intrinsics

print("Camera Matrix :\n %s"%(camera_matrix))
 
dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
pnp_algorithm = cv2.SOLVEPNP_ITERATIVE 
success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=pnp_algorithm)
# success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(model_points, image_points, camera_matrix, dist_coeffs, flags=pnp_algorithm)#, iterationsCount=1000)
 
print("Rotation Vector:\n %s"%(rotation_vector))
print("Translation Vector:\n %s"%(translation_vector))
 
# Project a 3D point (0, 0, 1000.0) onto the image plane.
# We use this to draw a line sticking out of the nose
 
R, j = cv2.Rodrigues(rotation_vector)
# print(R)
# (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 0.1)]), R, translation_vector, camera_matrix, dist_coeffs)
 
draw_cuboid_2d(im, image_points[:8], (255,0,0)) 
cv2.circle(im, tuple(image_points[-1]), 4, (0,0,255), -1)
# p1 = ( int(image_points[0][0]), int(image_points[0][1]))
# p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
 
# cv2.line(im, p1, p2, (255,0,0), 2)
 
# Display image
cv2.imshow("Output", im)
cv2.waitKey(0)

object_points = points[cls]
M = np.identity(4)
M[:3,:3] = R
M[:3,-1] = translation_vector.squeeze()
object_cloud = create_cloud(object_points, T=M)

open3d.draw_geometries([scene_cloud, object_cloud])
