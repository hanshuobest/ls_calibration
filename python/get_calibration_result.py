
import numpy as np
import yaml
import argparse
import math
import cv2

# base_link坐标系                          # patten object坐标系            camera坐标系 Z
#           Z       X                          ------------>X                       ^
#           ^     ^                          / |                                  *
#           |    *                          /  |                                *
#           |  *                           /   |                              ------------> X
#   Y <-----^                             v    v                              |
#                                        Z     Y                              |
#                                                                             |
#                                                                             V
#                                                                             Y


def read_camera_2_lidar(yml_file):
    fs = cv2.FileStorage(yml_file, cv2.FILE_STORAGE_READ)
    fn = fs.getNode("CameraExtrinsicMat")
    return fn.mat()


def write_yaml(yaml_file, info):
    with open(yaml_file, "w") as f:
        yaml.dump(info, f, indent=4)


# 检查一个矩阵是否是一个有效的旋转矩阵
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# 将旋转矩阵转为欧拉角
def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yml", help="calibration result yal file",
                        default="/home/han/catkin_ws/2020-12-26-calibrate-7.yml")
    args = parser.parse_args()

    yaml_file = args.yml
    extrinsic_mat = read_camera_2_lidar(yaml_file)
    print("extrinsic_mat: ", extrinsic_mat)

    lidar_2_camera = np.mat(extrinsic_mat).I
    print(lidar_2_camera)

    calibration_euler_angle_info = {}

    Ang_cl = rotationMatrixToEulerAngles(lidar_2_camera[:3, :3])
    print("lidar to camera euler angles: {}".format(Ang_cl))

    # 相机到base
    T_bc = np.mat([[0, 0, 1, 0.2103], [-1, 0, 0, 0.03092],
                   [0, -1, 0, 0.05345], [0, 0, 0, 1]])

    Ang_bc = rotationMatrixToEulerAngles(T_bc[:3, :3]).tolist()
    print("Camera to base euler angles: {}".format(Ang_bc))

    calibration_bc = []
    calibration_bc.append(T_bc[0, 3].item())
    calibration_bc.append(T_bc[1, 3].item())
    calibration_bc.append(T_bc[2, 3].item())
    calibration_bc.append(Ang_bc[2])
    calibration_bc.append(Ang_bc[1])
    calibration_bc.append(Ang_bc[0])
    
    print("calibration_bc: {}".format(calibration_bc))

    # lidar到base
    T_bl = T_bc @ lidar_2_camera
    print("R_bl: ", T_bl)
    Ang_bl = rotationMatrixToEulerAngles(T_bl[:3, :3]).tolist()
    print("lidar to base euler angles: {}".format(Ang_bl))

    calibration_bl = []
    calibration_bl.append(T_bl[0, 3].item())
    calibration_bl.append(T_bl[1, 3].item())
    calibration_bl.append(T_bl[2, 3].item())
    calibration_bl.append(Ang_bl[2])
    calibration_bl.append(Ang_bl[1])
    calibration_bl.append(Ang_bl[0])
    print("calibration_bc: {}".format(calibration_bl))

    # tof->camera
    # T_ct = np.mat([[0.99995685, -0.001101203, -0.0092244782, 0.046744373],
    #                [0.0011235935, 0.99999642, 0.0024224618, 0.00042156503],
    #                [0.0092217773, -0.0024327217, 0.99995452, -0.0026839232],
    #                [0, 0, 0, 1]])

    T_ct = np.mat([[0.99994123, -0.0027064332, -0.010495939, 0.047117355],
                   [0.0026847674, 0.99999422, -0.0020777581, 0.0010146408],
                   [0.010501503, 0.0020494568, 0.99994278, -0.0017133131],
                   [0, 0, 0, 1]])

    # tof->base
    T_bt = T_bc @ T_ct
    Ang_bt = rotationMatrixToEulerAngles(T_bt[:3, :3]).tolist()
    print("tof to base euler angles: {}".format(Ang_bt))

    calibration_bt = []
    calibration_bt.append(T_bt[0, 3].item())
    calibration_bt.append(T_bt[1, 3].item())
    calibration_bt.append(T_bt[2, 3].item())
    calibration_bt.append(Ang_bt[2])
    calibration_bt.append(Ang_bt[1])
    calibration_bt.append(Ang_bt[0])
    print("calibration_bt: {}".format(calibration_bt))

    calib_info = {
        "lidar_to_base": T_bl.tolist(),
        "camera_to_base": T_bc.tolist(),
        "tof_to_base": T_bt.tolist()
    }
    write_yaml("calibration.yaml", calib_info)

    calib_info = {
        "lidar_to_base": calibration_bl,
        "camera_to_base": calibration_bc,
        "tof_to_base": calibration_bt
    }
    write_yaml("calibration_euler.yaml", calib_info)
