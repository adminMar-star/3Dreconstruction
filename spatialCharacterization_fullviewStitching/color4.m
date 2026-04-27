clc; clear; close all;

%% ====== 参数 ======
folder ="C:\Users\xiiamyyyy\Desktop\10-10\新建文件夹 (2)";   % 子图文件夹
save_result = true;
output_file = "stitched_result_improved.jpg";

% 色彩迁移参数
use_color_transfer = true;      
color_ref_index = 1;            
color_transfer_model = 'lightweight_cnn'; 

% 新增：色彩融合参数
smooth_color_transition = true;    % 启用色彩平滑过渡
color_consistency_strength = 5;  % 色彩一致性强度 (0-1)
overlap_color_blending = true;     % 重叠区域色彩混合

% 新增：可视化参数
show_feature_matches = true;       % 显示特征匹配连线
show_all_matches = false;          % 显示所有匹配点（true）或只显示内点（false）
max_matches_to_show = 50;          % 最多显示的匹配点数

%% ====== 读取图像并排序 ======
imgs_info = dir(fullfile(folder, '*.*g'));
imgs_info = imgs_info(~[imgs_info.isdir]);

% 按文件名中的数字排序（没有数字按原序）
names = {imgs_info.name};
nums = zeros(size(names));
for i = 1:numel(names)
    tok = regexp(names{i}, '\d+', 'match');
    if ~isempty(tok)
        nums(i) = str2double(tok{end});
    else
        nums(i) = i;
    end
end
[~, idx] = sort(nums);
imgs_info = imgs_info(idx);
n = numel(imgs_info);
fprintf("共读取 %d 张图像\n", n);

% 显示所有图像文件名用于调试
fprintf('图像文件列表:\n');
for i = 1:n
    fprintf('%d: %s\n', i, imgs_info(i).name);
end

%% ====== 改进的色彩预处理 ======
if use_color_transfer
    fprintf('正在进行色彩统一预处理...\n');
    
    % 读取参考图像
    ref_img = imread(fullfile(folder, imgs_info(color_ref_index).name));
    
    % 创建处理后的图像存储
    processed_imgs = cell(n, 1);
    
    for i = 1:n
        if i == color_ref_index
            processed_imgs{i} = ref_img;
            fprintf('参考图像 %d: 保持原图\n', i);
            continue;
        end
        
        current_img = imread(fullfile(folder, imgs_info(i).name));
        
        switch color_transfer_model
            case 'reinhard'
                processed_imgs{i} = smoothColorTransfer(ref_img, current_img, color_consistency_strength);
            case 'lightweight_cnn'
                processed_imgs{i} = smoothColorTransfer(ref_img, current_img, color_consistency_strength);
            otherwise
                processed_imgs{i} = current_img;
        end
        
        fprintf('完成图像 %d/%d 的色彩统一\n', i, n);
    end
else
    % 如果不使用色彩迁移，直接读取图像
    processed_imgs = cell(n, 1);
    for i = 1:n
        processed_imgs{i} = imread(fullfile(folder, imgs_info(i).name));
    end
end

%% ====== 读入第一张作为基准 ======
img0 = processed_imgs{1};
img_base = im2double(img0);                    
gray_base = rgb2gray(img_base);

% 保存每张图像的变换矩阵（第一张为单位矩阵）
tforms(n) = projective2d(eye(3));
imageSize = zeros(n,2);
imageSize(1,:) = size(gray_base);

% 调试：显示第一张图像
figure('Name', '基准图像');
imshow(img0);
title(sprintf('基准图像: %s', imgs_info(1).name));

%% ====== 逐张计算变换（添加可视化） ======
fprintf('开始计算图像变换...\n');

Nreset = 3;  
failed_matches = 0;

% 存储匹配信息用于可视化
match_info = cell(n-1, 1);

for k = 2:n
    fprintf('处理图像 %d/%d: %s\n', k, n, imgs_info(k).name);
    
    % 读取下一张图像
    Inext = processed_imgs{k};
    img_next = im2double(Inext);
    gray_next = rgb2gray(img_next);
    imageSize(k,:) = size(gray_next);

    % 特征检测与匹配
    try
        points_base = detectSIFTFeatures(gray_base);
        points_next = detectSIFTFeatures(gray_next);
        feature_method = 'SIFT';
    catch
        warning('SIFT不可用，使用SURF特征');
        points_base = detectSURFFeatures(gray_base);
        points_next = detectSURFFeatures(gray_next);
        feature_method = 'SURF';
    end

    [features_base, valid_base] = extractFeatures(gray_base, points_base);
    [features_next, valid_next] = extractFeatures(gray_next, points_next);

    indexPairs = matchFeatures(features_next, features_base, ...
                               'MaxRatio', 0.6, 'MatchThreshold', 50, ...
                               'Unique', true);

    % 存储匹配信息
    match_info{k-1}.points_base = valid_base;
    match_info{k-1}.points_next = valid_next;
    match_info{k-1}.indexPairs = indexPairs;
    match_info{k-1}.feature_method = feature_method;

    if isempty(indexPairs) || size(indexPairs,1) < 10
        warning("图像 %d 匹配点不足 (%d)，尝试调整参数", k, size(indexPairs,1));
        
        indexPairs = matchFeatures(features_next, features_base, ...
                                   'MaxRatio', 0.8, 'MatchThreshold', 80, ...
                                   'Unique', true);
        
        if isempty(indexPairs) || size(indexPairs,1) < 6
            warning("重新匹配仍失败，使用平移变换估计");
            failed_matches = failed_matches + 1;
            tform_est = affine2d([1 0 0; 0 1 0; 100 0 1]);
            tforms(k).T = tform_est.T * tforms(k-1).T;
            gray_base = gray_next;
            continue;
        end
    end

    matched_next = valid_next(indexPairs(:,1)); 
    matched_base = valid_base(indexPairs(:,2));

    fprintf('找到 %d 对匹配点\n', size(indexPairs,1));

    % 计算变换矩阵
    try
        [tform, inlierIdx] = estimateGeometricTransform2D(...
            matched_next, matched_base, 'affine', ...
            'Confidence', 99.5, 'MaxNumTrials', 3000, 'MaxDistance', 2.5);
    catch
        try
            [tform, inlierIdx] = estimateGeometricTransform2D(...
                matched_next, matched_base, 'similarity', ...
                'Confidence', 99.5, 'MaxNumTrials', 3000);
        catch
            [tform, inlierIdx] = estimateGeometricTransform2D(...
                matched_next, matched_base, 'translation', ...
                'Confidence', 99.5, 'MaxNumTrials', 3000);
        end
    end

    % 存储内点信息
    match_info{k-1}.inlierIdx = inlierIdx;
    match_info{k-1}.tform = tform;

    if sum(inlierIdx) < 8
        warning("图像 %d 有效内点过少 (%d)，使用上一张变换", k, sum(inlierIdx));
        tforms(k) = tforms(k-1);
        gray_base = gray_next;
        continue;
    end

    fprintf('有效内点: %d/%d\n', sum(inlierIdx), size(indexPairs,1));

    % ==== 特征匹配可视化 ====
    if show_feature_matches
        visualizeFeatureMatches(img_base, img_next, match_info{k-1}, k, ...
                               show_all_matches, max_matches_to_show);
    end

    % 重定位机制
    if mod(k, Nreset) == 0 || k == n
        fprintf('执行重定位到基准图像...\n');
        gray_ref = rgb2gray(im2double(processed_imgs{1}));
        
        try
            points_ref = detectSIFTFeatures(gray_ref);
        catch
            points_ref = detectSURFFeatures(gray_ref);
        end
        
        [features_ref, valid_ref] = extractFeatures(gray_ref, points_ref);
        
        indexPairs0 = matchFeatures(features_next, features_ref, ...
                                    'MaxRatio', 0.7, 'MatchThreshold', 60, ...
                                    'Unique', true);

        if size(indexPairs0,1) >= 10
            matched_next0 = valid_next(indexPairs0(:,1));
            matched_ref   = valid_ref(indexPairs0(:,2));
            try
                [tform0, inlierIdx0] = estimateGeometricTransform2D(...
                    matched_next0, matched_ref, 'affine', ...
                    'Confidence', 99.5, 'MaxNumTrials', 2000);
                
                if sum(inlierIdx0) >= 8 && rcond(tform0.T) > 1e-10
                    tforms(k).T = tform0.T;
                    fprintf('重定位成功，使用全局变换\n');
                else
                    tforms(k).T = tform.T * tforms(k-1).T;
                end
            catch
                tforms(k).T = tform.T * tforms(k-1).T;
            end
        else
            tforms(k).T = tform.T * tforms(k-1).T;
        end
    else
        if rcond(tform.T) < 1e-12
            tforms(k) = tforms(k-1);
        else
            tforms(k).T = tform.T * tforms(k-1).T;
        end
    end

    gray_base = gray_next;
end

fprintf('变换计算完成，失败匹配数: %d/%d\n', failed_matches, n-1);

%% ====== 计算全景输出范围 ======
fprintf('计算全景图范围...\n');
xLimits = zeros(n,2);
yLimits = zeros(n,2);
for i = 1:n
    [xlim, ylim] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
    xLimits(i,:) = xlim;
    yLimits(i,:) = ylim;
end

xMin = min(xLimits(:));
xMax = max(xLimits(:));
yMin = min(yLimits(:));
yMax = max(yLimits(:));

fprintf('全景图范围: X[%.1f, %.1f], Y[%.1f, %.1f]\n', xMin, xMax, yMin, yMax);

width  = round(xMax - xMin);
height = round(yMax - yMin);
fprintf('全景图尺寸: %d x %d\n', width, height);

if width <= 0 || height <= 0
    error('计算到的画布尺寸无效：宽或高 <=0');
end

panoramaView = imref2d([height width], [xMin xMax], [yMin yMax]);

%% ====== 改进的色彩融合拼接 ======
fprintf('开始图像融合（改进的色彩融合）...\n');
panorama_accum = zeros([height width 3], 'double');
weight_accum = zeros([height width], 'double');

% 存储每张图像的变换后版本和掩码
warped_imgs = cell(n, 1);
warped_masks = cell(n, 1);

for i = 1:n
    fprintf('处理图像 %d/%d...\n', i, n);
    
    I = im2double(processed_imgs{i});
    warped = imwarp(I, tforms(i), 'OutputView', panoramaView);
    mask = imwarp(true(size(I,1), size(I,2)), tforms(i), 'OutputView', panoramaView);

    warped(isnan(warped)) = 0;
    mask(isnan(mask)) = false;

    if ~any(mask(:))
        warning('图像 %d 变换后无有效像素，跳过', i);
        continue;
    end

    warped_imgs{i} = warped;
    warped_masks{i} = mask;

    % 改进的权重计算：更平滑的过渡
    distMap = bwdist(~mask);
    if max(distMap(:)) > 0
        distMap = distMap / max(distMap(:));
    else
        distMap = double(mask);
    end
    
    % 使用更柔和的高斯平滑
    distMap = imgaussfilt(distMap, 5);  % 增加平滑半径
    distMap3 = repmat(distMap, [1,1,3]);

    panorama_accum = panorama_accum + warped .* distMap3;
    weight_accum = weight_accum + distMap;
end

% 基础融合
weight_accum(weight_accum == 0) = eps;
panorama_base = zeros(size(panorama_accum));
for c = 1:3
    panorama_base(:,:,c) = panorama_accum(:,:,c) ./ weight_accum;
end

%% ====== 色彩一致性后处理 ======
if smooth_color_transition
    fprintf('进行色彩一致性后处理...\n');
    panorama_final = colorConsistencyBlend(panorama_base, warped_imgs, warped_masks, weight_accum);
else
    panorama_final = panorama_base;
end

% 转 uint8
panorama_uint8 = im2uint8(panorama_final);

%% ====== 显示与保存 ======
figure('Name','全景拼接结果 - 改进色彩融合','Units','normalized','OuterPosition',[0 0 1 1]);
imshow(panorama_uint8); 
title(sprintf('%d张图像拼接结果 - 改进色彩融合 (%d x %d)', n, width, height));

% 显示权重分布
figure('Name','融合权重分布');
imagesc(weight_accum); colorbar;
title('融合权重分布（更平滑）');

if save_result
    imwrite(panorama_uint8, output_file);
    fprintf('结果已保存到: %s\n', output_file);
end

fprintf('图像拼接完成！\n');
fprintf('统计信息:\n');
fprintf('- 总图像数: %d\n', n);
fprintf('- 失败匹配: %d\n', failed_matches);
fprintf('- 全景图尺寸: %d x %d\n', width, height);
fprintf('- 色彩一致性强度: %.1f\n', color_consistency_strength);

