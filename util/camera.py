import cv2
import math
import time
import numpy as np
import scipy.io as scio
import skimage.transform as skt
import pybullet as p
import scipy.stats as ss
# 图像参数
HEIGHT = 480
WIDTH = 640

# 图像尺寸
IMAGEWIDTH = 640
IMAGEHEIGHT = 480
nearPlane = 0.01
farPlane = 10
fov = 60  #垂直视场 图像高tan(30) * 0.7 *2 = 0.8082903m
aspect = IMAGEWIDTH / IMAGEHEIGHT

# 相机参数
def imresize(image, size, interp="nearest"):
    skt_interp_map = {
        "nearest": 0,
        "bilinear": 1,
        "biquadratic": 2,
        "bicubic": 3,
        "biquartic": 4,
        "biquintic": 5
    }
    if interp in ("lanczos", "cubic"):
        raise ValueError("'lanczos' and 'cubic'"
                         " interpolation are no longer supported.")
    assert interp in skt_interp_map, ("Interpolation '{}' not"
                                      " supported.".format(interp))

    if isinstance(size, (tuple, list)):
        output_shape = size
    elif isinstance(size, (float)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size
        output_shape = tuple(np_shape.astype(int))
    elif isinstance(size, (int)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size / 100.0
        output_shape = tuple(np_shape.astype(int))
    else:
        raise ValueError("Invalid type for size '{}'.".format(type(size)))

    return skt.resize(image,
                      output_shape,
                      order=skt_interp_map[interp],
                      anti_aliasing=False,
                      mode="constant")

# 深度图像修复函数，参考dex-net代码
def inpaint(img, missing_value=0):
    """
    Inpaint missing values in depth image.
    :param missing_value: Value to fill in teh depth image.
    """
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)
    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    scale = np.abs(img).max()
    img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)
    # Back to original size and value range.
    img = img[1:-1, 1:-1]
    img = img * scale
    return img

def radians_TO_angle(radians):
    """
    弧度转角度
    """
    return 180 * radians / math.pi


def angle_TO_radians(angle):
    """
    角度转弧度
    """
    return math.pi * angle / 180


def eulerAnglesToRotationMatrix(theta):
    """
    欧拉角转旋转矩阵
    theta: [r, p, y]
    """
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def getTransfMat(offset, rotate):
    """
    将平移向量和旋转矩阵合并为变换矩阵
    offset: (x, y, z)
    rotate: 旋转矩阵
    """
    mat = np.array([
        [rotate[0, 0], rotate[0, 1], rotate[0, 2], offset[0]],
        [rotate[1, 0], rotate[1, 1], rotate[1, 2], offset[1]],
        [rotate[2, 0], rotate[2, 1], rotate[2, 2], offset[2]],
        [0, 0, 0, 1.]
    ])
    return mat


class Camera:
    def __init__(self):
        """
        初始化相机参数，计算相机内参
        """
        self.fov = 60  # 垂直视场
        self.length = 2  # 相机高度
        self.H = self.length * math.tan(angle_TO_radians(self.fov / 2))  # 图像第一行的中点到图像中心点的实际距离 m
        self.W = WIDTH * self.H / HEIGHT  # 图像右方点的到中心点的实际距离 m
        # 计算 fx 和 fy
        self.A = (HEIGHT / 2) * self.length / self.H
        self.fx = self.fy = self.A
        self.cx = WIDTH / 2.0
        self.cy = HEIGHT / 2.0
        # 计算内参
        self.InMatrix = np.array([[self.A, 0, WIDTH / 2 - 0.5], [0, self.A, HEIGHT / 2 - 0.5], [0, 0, 1]], dtype=float)

        # 计算世界坐标系->相机坐标系的转换矩阵 4*4
        # 欧拉角: (pi, 0, 0)
        rotMat = eulerAnglesToRotationMatrix([math.pi, 0, 0])
        self.transMat = getTransfMat([-0.4, -0.95, 2], rotMat)

        # 加载相机
        self.movecamera(-0.4, -0.95, 2)
        self.projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
        self.camera_pos = (-0.4, -0.95, 2)
        self.camera_orn = p.getQuaternionFromEuler([math.pi, 0, 0])

    # 移动相机到指定位置
    def movecamera(self, x, y, z=0.7):
        self.viewMatrix = p.computeViewMatrix([x, y, z], [x, y, 0], [0, 1, 0])

    def renderCameraRGBImage(self):
        """
        渲染并返回RGB图像
        """
        # 渲染图像
        img_camera = p.getCameraImage(IMAGEWIDTH, IMAGEHEIGHT, self.viewMatrix, self.projectionMatrix,
                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
        w = img_camera[0]  # width of the image, in pixels
        h = img_camera[1]  # height of the image, in pixels
        rgba = img_camera[2]  # color data RGBA

        # 获取RGB图像（去除alpha通道）
        rgb = np.reshape(rgba, (h, w, 4))[:, :, :3]  # 转换为(h,w,4)并只取前3个通道
        return rgb

    def renderCameraDepthImage(self):

        # 渲染图像
        img_camera = p.getCameraImage(IMAGEWIDTH, IMAGEHEIGHT, self.viewMatrix, self.projectionMatrix,
                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
        w = img_camera[0]  # width of the image, in pixels
        h = img_camera[1]  # height of the image, in pixels
        dep = img_camera[3]  # depth data
        # 获取深度图像
        depth = np.reshape(dep, (h, w))  # [40:440, 120:520]
        A = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * farPlane * nearPlane
        B = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * farPlane
        C = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * (farPlane - nearPlane)

        im_depthCamera = np.divide(A, (np.subtract(B, np.multiply(C, depth))))  # 单位 m
        return im_depthCamera

    def renderCameraMask(self):
        """
        渲染计算抓取配置所需的图像
        """
        # 渲染图像
        img_camera = p.getCameraImage(IMAGEWIDTH, IMAGEHEIGHT, self.viewMatrix, self.projectionMatrix,
                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
        w = img_camera[0]  # width of the image, in pixels
        h = img_camera[1]  # height of the image, in pixels
        # rgba = img_camera[2]    # color data RGB
        # dep = img_camera[3]    # depth data
        mask = img_camera[4]  # mask data

        # 获取分割图像
        im_mask = np.reshape(mask, (h, w)).astype(np.int32)
        return im_mask

    def gaussian_noise(self, im_depth):
        """
        在image上添加高斯噪声，参考dex-net代码
        im_depth: 浮点型深度图，单位为米
        """
        gamma_shape = 1000.00
        gamma_scale = 1 / gamma_shape
        gaussian_process_sigma = 0.002  # 0.002
        gaussian_process_scaling_factor = 8.0  # 8.0
        im_height, im_width = im_depth.shape
        gp_rescale_factor = gaussian_process_scaling_factor  # 4.0
        gp_sample_height = int(im_height / gp_rescale_factor)  # im_height / 4.0
        gp_sample_width = int(im_width / gp_rescale_factor)  # im_width / 4.0
        gp_num_pix = gp_sample_height * gp_sample_width  # im_height * im_width / 16.0
        gp_sigma = gaussian_process_sigma
        gp_noise = ss.norm.rvs(scale=gp_sigma, size=gp_num_pix).reshape(gp_sample_height,
                                                                        gp_sample_width)  # 生成(均值为0，方差为scale)的gp_num_pix个数，并reshape

        gp_noise = imresize(gp_noise, gp_rescale_factor, interp="bicubic")  # resize成图像尺寸，bicubic为双三次插值算法

        im_depth += gp_noise

        return im_depth

    def add_noise(self, img):
        """
        添加高斯噪声和缺失值
        """
        img = self.gaussian_noise(img)  # 添加高斯噪声
        return img

    def camera_height(self):
        return self.length

    def img2camera(self, pt, dep):
        """
        获取像素点pt在相机坐标系中的坐标
        pt: [x, y]
        dep: 深度值

        return: [x, y, z]
        """
        pt_in_img = np.array([[pt[0]], [pt[1]], [1]], dtype=float)
        ret = np.matmul(np.linalg.inv(self.InMatrix), pt_in_img) * dep
        return list(ret.reshape((3,)))
        # print('坐标 = ', ret)

    def camera2img(self, coord):
        """
        将相机坐标系中的点转换至图像
        coord: [x, y, z]

        return: [row, col]
        """
        z = coord[2]
        coord = np.array(coord).reshape((3, 1))
        rc = (np.matmul(self.InMatrix, coord) / z).reshape((3,))

        return list(rc)[:-1]

    def length_TO_pixels(self, l, dep):
        """
        与相机距离为dep的平面上 有条线，长l，获取这条线在图像中的像素长度
        l: m
        dep: m
        """
        return l * self.A / dep

    def pixels_TO_length(self, p, dep):
        """
        length_TO_pixels函数的反转
        """
        return p * dep / self.A

    def camera2world(self, coord):
        """
        获取相机坐标系中的点在世界坐标系中的坐标
        corrd: [x, y, z]

        return: [x, y, z]
        """
        coord.append(1.)
        coord = np.array(coord).reshape((4, 1))
        coord_new = np.matmul(self.transMat, coord).reshape((4,))
        return list(coord_new)[:-1]

    def world2camera(self, coord):
        """
        获取世界坐标系中的点在相机坐标系中的坐标
        corrd: [x, y, z]

        return: [x, y, z]
        """
        coord.append(1.)
        coord = np.array(coord).reshape((4, 1))
        coord_new = np.matmul(np.linalg.inv(self.transMat), coord).reshape((4,))
        return list(coord_new)[:-1]

    def world2img(self, coord):
        """
        获取世界坐标系中的点在图像中的坐标
        corrd: [x, y, z]

        return: [row, col]
        """
        # 转到相机坐标系
        coord = self.world2camera(coord)
        # 转到图像
        pt = self.camera2img(coord)  # [y, x]
        return [int(pt[1]), int(pt[0])]

    def img2world(self, pt, dep):
        """
        获取像素点的世界坐标
        pt: [x, y]
        dep: 深度值 m
        return: [x, y, z]
        """
        coordInCamera = self.img2camera(pt, dep)
        return self.camera2world(coordInCamera)


if __name__ == '__main__':
    camera = Camera()
    print(camera.InMatrix)
