clc; clear; close all;

%% ========== 1. 输入拼接结果图像 ==========
imgPath = "D:\Image together\序列拼接\10.29\分割指标\stitched_result.png";   % ← 改成你的拼接结果图像
I = imread(imgPath);

%% ========== 2. 计算无参考拼接质量指标 ==========
metrics = noRefStitchMetrics(I);

%% ========== 3. 显示数值结果 ==========
disp("===== 无参考拼接质量评价指标 =====");
disp(metrics);

%% ========== 4. 可视化（可用于论文） ==========
figure('Color','w','Position',[200 200 1200 500]);

subplot(1,3,1);
imshow(I);
title('Stitched Result');

subplot(1,3,2);
imshow(metrics.gradMap,[]);
title('Gradient Magnitude Map');

subplot(1,3,3);
imshow(metrics.edgeMap);
title('Edge Map (Canny)');



function metrics = noRefStitchMetrics(I)
% =========================================================
% 无参考图像拼接质量评价函数
% Input:
%   I : 拼接结果图像 (RGB 或 Gray)
% Output:
%   metrics : 结构体，包含多种无参考评价指标
% =========================================================

%% ---------- 1. 预处理 ----------
if size(I,3) == 3
    Igray = rgb2gray(I);
else
    Igray = I;
end
Igray = im2double(Igray);

%% ---------- 2. 信息熵（Entropy） ----------
metrics.entropy = entropy(Igray);

%% ---------- 3. Laplacian 方差（清晰度） ----------
h = fspecial('laplacian', 0.2);
L = imfilter(Igray, h, 'replicate');
metrics.lap_var = var(L(:));

%% ---------- 4. Tenengrad 清晰度 ----------
[Gx, Gy] = imgradientxy(Igray, 'sobel');
metrics.tenengrad = mean(Gx(:).^2 + Gy(:).^2);

%% ---------- 5. 梯度幅值图 ----------
Gmag = sqrt(Gx.^2 + Gy.^2);
metrics.gradMap = Gmag;

%% ---------- 6. 梯度异常率（拼接痕迹） ----------
thr = mean(Gmag(:)) + 2 * std(Gmag(:));
metrics.grad_anomaly = sum(Gmag(:) > thr) / numel(Gmag);

%% ---------- 7. 边缘密度（结构完整性） ----------
edges = edge(Igray, 'canny');
metrics.edgeMap = edges;
metrics.edge_density = sum(edges(:)) / numel(edges);

%% ---------- 8. BRISQUE 无参考感知质量 ----------
% 需要 Computer Vision Toolbox
try
    metrics.brisque = brisque(I);
catch
    metrics.brisque = NaN;
    warning('BRISQUE 需要 Computer Vision Toolbox');
end

%% ---------- 9. 综合质量分（可选，用于排序） ----------
% 归一化（经验形式，论文中可不写）
metrics.quality_score = ...
    0.3 * normalizeValue(metrics.entropy) + ...
    0.3 * normalizeValue(metrics.tenengrad) + ...
    0.2 * normalizeValue(metrics.edge_density) - ...
    0.2 * normalizeValue(metrics.grad_anomaly);

end

%% ========== 子函数：归一化 ==========
function v = normalizeValue(x)
v = (x - min(x(:))) / (max(x(:)) - min(x(:)) + eps);
end
