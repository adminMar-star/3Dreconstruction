clc; clear; close all;

%% ================== 1. 实验路径 ==================
imgDir = "D:\Image together\序列拼接\10.29\分割指标";     % 拼接结果图像文件夹
imgFiles = dir(fullfile(imgDir, "*.png" ));

%% ================== 2. 参数设置（统一固定） ==================
params.grad_k   = 3.0;              % 梯度异常灵敏度
params.scales   = [0.5 1.0 2.0];     % 多尺度清晰度
params.edge_th  = [0.05 0.4];       % Canny 阈值

%% ================== 3. 批量评价 ==================
results = [];

for i = 1:length(imgFiles)
    I = imread(fullfile(imgDir, imgFiles(i).name));
    metrics = paramNoRefStitchMetrics(I, params);

    results(i).name          = imgFiles(i).name;
    results(i).entropy       = metrics.entropy;
    results(i).tenengrad_ms  = metrics.tenengrad_ms;
    results(i).edge_density  = metrics.edge_density;
    results(i).grad_anomaly  = metrics.grad_anomaly;
    results(i).brisque       = metrics.brisque;
end

%% ================== 4. 转表显示 ==================
T = struct2table(results);
disp(T);

%% ================== 5. 保存实验结果 ==================
writetable(T, "NoRef_Stitch_Metrics.csv");

function metrics = paramNoRefStitchMetrics(I, p)
% =========================================================
% 参数化无参考拼接质量评价
% =========================================================

%% ---------- 1. 预处理 ----------
if size(I,3)==3
    Igray = rgb2gray(I);
else
    Igray = I;
end
Igray = im2double(Igray);

%% ---------- 2. 信息熵 ----------
metrics.entropy = entropy(Igray);

%% ---------- 3. 多尺度 Tenengrad 清晰度 ----------
T = 0;
for s = p.scales
    Is = imgaussfilt(Igray, s);
    [Gx, Gy] = imgradientxy(Is, 'sobel');
    T = T + mean(Gx(:).^2 + Gy(:).^2);
end
metrics.tenengrad_ms = T / numel(p.scales);

%% ---------- 4. 梯度异常率（拼接痕迹） ----------
[Gx, Gy] = imgradientxy(Igray, 'sobel');
Gmag = sqrt(Gx.^2 + Gy.^2);
thr = mean(Gmag(:)) + p.grad_k * std(Gmag(:));
metrics.grad_anomaly = sum(Gmag(:) > thr) / numel(Gmag);

%% ---------- 5. 边缘密度（结构完整性） ----------
edges = edge(Igray, 'canny', p.edge_th);
metrics.edge_density = sum(edges(:)) / numel(edges);

%% ---------- 6. BRISQUE 感知质量 ----------
try
    metrics.brisque = brisque(I);
catch
    metrics.brisque = NaN;
end

end
