# 后端配置（修复点云空缺问题）
import matplotlib

matplotlib.use('TkAgg')

import cv2 as cv
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def compute_multi_scale_features(patch):
    """多尺度锐度特征提取（保留低锐度区域）"""
    features = []
    scales = [3, 5, 7]
    for s in scales:
        pad = (s - 1) // 2
        patch_s = cv.copyMakeBorder(patch, pad, pad, pad, pad, cv.BORDER_REFLECT_101)
        patch_s = patch_s[pad:-pad, pad:-pad]

        lap = cv.Laplacian(patch_s, cv.CV_64F)
        f_lap = np.var(lap)

        p95 = np.percentile(patch_s, 95)
        p5 = np.percentile(patch_s, 5)
        f_con = p95 - p5

        sobel_x = cv.Sobel(patch_s, cv.CV_64F, 1, 0, ksize=3)
        sobel_y = cv.Sobel(patch_s, cv.CV_64F, 0, 1, ksize=3)
        f_tex = np.mean(np.sqrt(sobel_x ** 2 + sobel_y ** 2))

        features.append([f_lap, f_con, f_tex])
    return np.array(features)


def dynamic_attention_weight(features, patch, na, magnification):
    """动态权重计算（适配低锐度区域）"""
    # 特征权重：降低低锐度区域的权重衰减
    lap_std = np.std(features[:, 0])
    lap_weight = 0.2 + 0.4 * (lap_std / (lap_std + 1e-8))  # 扩大权重范围

    con_mean = np.mean(features[:, 1])
    con_weight = 0.2 + 0.4 * (con_mean / (np.max(features[:, 1]) + 1e-8))

    tex_vals = features[:, 2]
    tex_entropy = -np.sum(tex_vals * np.log2(tex_vals + 1e-8)) / len(tex_vals)
    tex_weight = 0.2 + 0.4 * (tex_entropy / (tex_entropy + 1e-8))

    chan_weight = np.array([lap_weight, con_weight, tex_weight])
    chan_weight = chan_weight / chan_weight.sum()

    # 尺度权重
    scales = [3, 5, 7]
    scale_energy = []
    for s in scales:
        pad = (s - 1) // 2
        patch_s = cv.copyMakeBorder(patch, pad, pad, pad, pad, cv.BORDER_REFLECT_101)
        patch_s = patch_s[pad:-pad, pad:-pad]
        grad = cv.Sobel(patch_s, cv.CV_64F, 1, 1, ksize=3)
        scale_energy.append(np.mean(np.abs(grad)))
    scale_weight = np.array(scale_energy) / (np.sum(scale_energy) + 1e-8)

    dynamic_weight = np.outer(scale_weight, chan_weight)

    # 物理约束（放宽低锐度区域的下限）
    con_min_weight = 0.05 + 0.15 * na  # 降低对比度权重下限
    dynamic_weight[:, 1] = np.maximum(dynamic_weight[:, 1], con_min_weight)
    tex_min_weight = 0.03 + 0.08 * np.log10(magnification)  # 降低纹理权重下限
    dynamic_weight[:, 2] = np.maximum(dynamic_weight[:, 2], tex_min_weight)

    return dynamic_weight / dynamic_weight.sum()


def compute_vhx_sharpness(patch, na, magnification):
    """锐度计算（保留低锐度值）"""
    multi_features = compute_multi_scale_features(patch)
    dynamic_weight = dynamic_attention_weight(multi_features, patch, na, magnification)
    sharpness = np.sum(multi_features * dynamic_weight)
    return sharpness if sharpness > 1e-8 else 1e-8  # 避免锐度为0


def calculate_vhx_dof(na, magnification, n=1.0):
    lambda_light = 0.55e-6
    e = 0.00025
    dof_diffraction = (lambda_light * n) / (na ** 2)
    dof_geometric = (n * e) / (na * magnification)
    return (dof_diffraction + dof_geometric) * 1000


def generate_depth_vhx(img, na, magnification, pixel_size, depth_scale=4.0):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    H, W = gray.shape
    print(f"基恩士图像尺寸: {H}x{W}，X{int(magnification)}倍率")

    sharpness_map = np.zeros((H, W), dtype=np.float32)
    gray_padded = cv.copyMakeBorder(gray, 3, 3, 3, 3, cv.BORDER_REFLECT_101)

    for y in range(H):
        for x in range(W):
            patch = gray_padded[y:y + 7, x:x + 7]
            sharpness_map[y, x] = compute_vhx_sharpness(patch, na, magnification)

    # 锐度增强（保留低锐度区域）
    sharpness_max, sharpness_min = np.max(sharpness_map), np.min(sharpness_map)
    if sharpness_max - sharpness_min > 1e-6:
        sharpness_map = (sharpness_map - sharpness_min) / (sharpness_max - sharpness_min + 1e-8)
        sharpness_map = np.power(sharpness_map, 0.6)  # 降低非线性拉伸强度，保留低锐度

    dof = calculate_vhx_dof(na, magnification)
    print(f"X{int(magnification)}倍率景深: {dof:.4f} μm，深度缩放系数: {depth_scale}")
    depth_map = - (sharpness_map - 0.5) * dof * depth_scale
    # 归一化时保留全范围值，不截断
    depth_map = cv.normalize(depth_map, None, alpha=np.min(depth_map), beta=np.max(depth_map), norm_type=cv.NORM_MINMAX)
    depth_map = cv.bilateralFilter(depth_map, d=5, sigmaColor=15, sigmaSpace=15)  # 减弱滤波，保留孔洞区域
    return depth_map


def generate_point_cloud_vhx(rgb_img, depth_map, pixel_size, z_scale=3.0, depth_stretch=10):
    H, W = depth_map.shape
    rgb_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2RGB)

    x = (np.arange(W) - W / 2) * pixel_size
    y = (np.arange(H) - H / 2) * pixel_size
    xx, yy = np.meshgrid(x, y)

    # 深度拉伸（保留低深度值）
    z_norm = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 1e-8)
    z_stretched = np.power(z_norm, 0.7) * depth_stretch  # 降低拉伸强度
    x_flat, y_flat, z_flat = xx.flatten(), yy.flatten(), z_stretched.flatten() * z_scale

    # 放宽异常点过滤（保留3倍标准差内的点）
    z_mean, z_std = np.mean(z_flat), np.std(z_flat)
    valid_mask = np.abs(z_flat - z_mean) <= 3 * z_std  # 从2→3倍标准差
    # 额外保留低锐度区域（即使偏离均值）
    sharpness_flat = (z_norm.flatten() < 0.2)  # 低锐度区域标记
    valid_mask = np.logical_or(valid_mask, sharpness_flat)

    x_flat, y_flat, z_flat = x_flat[valid_mask], y_flat[valid_mask], z_flat[valid_mask]
    colors = rgb_img.reshape(-1, 3)[valid_mask] / 255.0

    print(f"点云点数: {len(x_flat)}，Z轴范围: {np.min(z_flat):.4f}~{np.max(z_flat):.4f}")
    return np.column_stack((x_flat, y_flat, z_flat)), colors, z_norm


def visualize_vhx_results(img, depth_map, point_cloud, colors, pixel_size, z_norm, magnification):
    cmap = LinearSegmentedColormap.from_list(
        'depth_cmap',
        [(0, 0, 0.8), (0, 0.8, 0), (0.8, 0, 0), (0.9, 0.9, 0)],
        N=256
    )

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title(f"基恩士X{int(magnification)}倍率图像 (1Pixel={pixel_size}μm)")
    plt.subplot(122)
    im = plt.imshow(z_norm, cmap=cmap)
    plt.colorbar(im, label='Normalized Depth (0~1)')
    plt.tight_layout()
    plt.savefig("vhx_depth_enhanced.png", dpi=200)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    z_vals = np.array(pcd.points)[:, 2]
    z_vals_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-8)
    pcd.colors = o3d.utility.Vector3dVector(cmap(z_vals_norm)[:, :3])

    # 取消下采样，保留所有点
    if len(pcd.points) > 200000:
        pcd = pcd.voxel_down_sample(voxel_size=pixel_size * 1.0)  # 增大体素，减少过滤

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="无空缺点云")
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = 4  # 减小点大小，显示密集区域
    opt.background_color = [1, 1, 1]
    ctr = vis.get_view_control()
    ctr.set_front([-0.5, -0.5, 0.5])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 0, 1])
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5))
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    try:
        IMAGE_PATH = "1.jpg"
        MAGNIFICATION = 800
        NA = 0.12
        PIXEL_SIZE = 0.567

        DEPTH_SCALE = 6.0  # 降低深度缩放，减少空缺
        Z_SCALE = 4.0
        DEPTH_STRETCH = 12

        img = cv.imread(IMAGE_PATH)
        if img is None:
            raise FileNotFoundError(f"无法读取图像：{IMAGE_PATH}")

        depth_map = generate_depth_vhx(
            img, NA, MAGNIFICATION,
            pixel_size=PIXEL_SIZE,
            depth_scale=DEPTH_SCALE
        )

        point_cloud, colors, z_norm = generate_point_cloud_vhx(
            img, depth_map,
            pixel_size=PIXEL_SIZE,
            z_scale=Z_SCALE,
            depth_stretch=DEPTH_STRETCH
        )

        visualize_vhx_results(img, depth_map, point_cloud, colors, pixel_size=PIXEL_SIZE, z_norm=z_norm,
                              magnification=MAGNIFICATION)

    except Exception as e:
        print(f"错误：{str(e)}")