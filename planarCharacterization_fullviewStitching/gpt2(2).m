clc; clear; close all;
%% ====== 参数 ======
folder = "D:\Image together\序列拼接\10.29\6";   % 子图文件夹
save_result = false;                  % 是否保存拼接结果
output_file = "stitched_result.jpg"; % 输出文件名
maxPoints = 2000;                    % 最大特征点数量（平衡速度与精度）
ratioThreshold = 0.75;               % Lowe比率测试阈值（过滤误匹配）
Nreset = 5;                          % 重定位间隔（减少累积误差）
minContrast = 0.03;                  % FAST特征检测对比度阈值
tau = 0.05;                          % 方向一致性阈值（Frobenius范数），根据数据调节

% 支持的图像格式（覆盖主流类型）
image_extensions = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tif', '*.tiff'};

%% ====== 第一步：读取并排序图像（防空检查） ======
imgs_info = [];
for ext = image_extensions
    file_list = dir(fullfile(folder, ext{1}));
    if ~isempty(file_list)
        imgs_info = [imgs_info; file_list]; % 合并不同格式文件
    end
end

if ~isempty(imgs_info)
    imgs_info = imgs_info(~[imgs_info.isdir]);
end

if isempty(imgs_info)
    error('错误：在文件夹 "%s" 中未找到任何图像文件！\n支持格式：%s', ...
          folder, strjoin(cellfun(@(x) x(2:end), image_extensions, 'UniformOutput', false), ', '));
end

% 按文件名中的数字排序
names = {imgs_info.name};
nums = zeros(size(names));
for i = 1:numel(names)
    num_matches = regexp(names{i}, '\d+', 'match');
    if ~isempty(num_matches)
        nums(i) = str2double(num_matches{end});
    else
        nums(i) = i;
    end
end
[~, sort_idx] = sort(nums);
imgs_info = imgs_info(sort_idx);
img_count = numel(imgs_info);
fprintf('成功读取 %d 张图像\n', img_count);

%% ====== 第二步：初始化基准图像（第一张为全局基准） ======
try
    base_img = imread(fullfile(folder, imgs_info(1).name));
catch ME
    error('错误：无法读取基准图像 "%s"！原因：%s', ...
          fullfile(folder, imgs_info(1).name), ME.message);
end

base_img_double = im2double(base_img);
base_gray = rgb2gray(base_img_double);

% 初始化变换矩阵数组（第一张为单位矩阵，无变换）
tforms = repmat(projective2d(eye(3)), 1, img_count);
img_sizes = zeros(img_count, 2);
img_sizes(1, :) = size(base_gray);

%% ====== 第三步：逐张计算图像变换（方向一致性 + 反馈修正） ======
for img_idx = 2:img_count
    current_img_path = fullfile(folder, imgs_info(img_idx).name);
    try
        current_img = imread(current_img_path);
    catch ME
        warning('警告：无法读取图像 "%s"，将跳过该图像！原因：%s', ...
                current_img_path, ME.message);
        tforms(img_idx) = tforms(img_idx - 1);
        continue;
    end
    current_img_double = im2double(current_img);
    current_gray = rgb2gray(current_img_double);
    img_sizes(img_idx, :) = size(current_gray);

    %% 2. FAST 检测
    base_fast_points = detectFASTFeatures(base_gray, 'MinContrast', minContrast);
    current_fast_points = detectFASTFeatures(current_gray, 'MinContrast', minContrast);

    if base_fast_points.Count > maxPoints
        base_fast_points = selectStrongest(base_fast_points, maxPoints);
    end
    if current_fast_points.Count > maxPoints
        current_fast_points = selectStrongest(current_fast_points, maxPoints);
    end

    base_fast_coords = base_fast_points.Location;
    current_fast_coords = current_fast_points.Location;

    %% 3. SIFT 描述符
    try
        [base_sift_desc, base_sift_points] = extractFeatures(base_gray, base_fast_coords, 'Method', 'SIFT');
    catch
        % 兼容：当 extractFeatures 要求 points 对象时
        [base_sift_desc, base_sift_points] = extractFeatures(base_gray, base_fast_points, 'Method', 'SIFT');
    end
    try
        [current_sift_desc, current_sift_points] = extractFeatures(current_gray, current_fast_coords, 'Method', 'SIFT');
    catch
        [current_sift_desc, current_sift_points] = extractFeatures(current_gray, current_fast_points, 'Method', 'SIFT');
    end

    % 处理对象/矩阵类型
    if isobject(base_sift_desc)
        base_sift_data = base_sift_desc.Features;
    else
        base_sift_data = base_sift_desc;
    end
    if isobject(current_sift_desc)
        current_sift_data = current_sift_desc.Features;
    else
        current_sift_data = current_sift_desc;
    end

    % 通过坐标匹配获取有效索引（SIFTPoints可能是子集）
    [~, base_valid_idx] = ismembertol(base_sift_points.Location, base_fast_coords, 1e-6, 'ByRows', true);
    [~, current_valid_idx] = ismembertol(current_sift_points.Location, current_fast_coords, 1e-6, 'ByRows', true);
    base_valid_coords = base_fast_coords(base_valid_idx, :);
    current_valid_coords = current_fast_coords(current_valid_idx, :);

    if isempty(base_sift_data) || isempty(current_sift_data)
        warning('警告：图像 "%s" SIFT描述符为空，沿用前一张变换', imgs_info(img_idx).name);
        tforms(img_idx) = tforms(img_idx - 1);
        base_gray = current_gray;
        continue;
    end

    % 确保描述符维度一致（必要时截断）
    if size(base_sift_data, 2) ~= size(current_sift_data, 2)
        min_dim = min(size(base_sift_data, 2), size(current_sift_data, 2));
        base_sift_data = base_sift_data(:, 1:min_dim);
        current_sift_data = current_sift_data(:, 1:min_dim);
        if min_dim == 0
            warning('警告：图像 "%s" 特征描述符维度无效，沿用前一张变换', imgs_info(img_idx).name);
            tforms(img_idx) = tforms(img_idx - 1);
            base_gray = current_gray;
            continue;
        end
    end

    %% 4. 特征匹配（手动 Lowe 比率测试）
    base_sift_single = single(base_sift_data')';    % 每行一个特征向量
    current_sift_single = single(current_sift_data')';

    es = ExhaustiveSearcher(base_sift_single);
    [nearest_idx, distances] = knnsearch(es, current_sift_single, 'K', 2);

    valid_matches = distances(:,1) < ratioThreshold * distances(:,2);
    index_pairs = [nearest_idx(valid_matches,1), find(valid_matches)]; % [base_idx, current_idx]

    if size(index_pairs,1) < 6
        warning('警告：图像 "%s" 有效匹配点仅 %d 个（需≥6），沿用前一张变换', ...
                imgs_info(img_idx).name, size(index_pairs,1));
        tforms(img_idx) = tforms(img_idx - 1);
        base_gray = current_gray;
        continue;
    end

    matched_base_coords = base_valid_coords(index_pairs(:,1), :);
    matched_current_coords = current_valid_coords(index_pairs(:,2), :);

    %% 5. 双向变换估计（正向 + 反向）
    % 正向估计（current -> base）
    try
        [T_fw, inlier_fw_mask] = estimateGeometricTransform2D(...
            matched_current_coords, matched_base_coords, 'similarity', ...
            'Confidence', 99.9, 'MaxNumTrials', 2000);
    catch
        try
            [T_fw, inlier_fw_mask] = estimateGeometricTransform2D(...
                matched_current_coords, matched_base_coords, 'translation', ...
                'Confidence', 99.9, 'MaxNumTrials', 2000);
        catch
            warning('警告：正向变换估计失败，沿用前一张变换');
            tforms(img_idx) = tforms(img_idx - 1);
            base_gray = current_gray;
            continue;
        end
    end

    % 反向估计（base -> current）
    try
        [T_bw, inlier_bw_mask] = estimateGeometricTransform2D(...
            matched_base_coords, matched_current_coords, 'similarity', ...
            'Confidence', 99.9, 'MaxNumTrials', 2000);
    catch
        try
            [T_bw, inlier_bw_mask] = estimateGeometricTransform2D(...
                matched_base_coords, matched_current_coords, 'translation', ...
                'Confidence', 99.9, 'MaxNumTrials', 2000);
        catch
            % 如果反向估计失败，仅以正向为准
            geo_tform = T_fw;
            inlier_fw_mask = logical(inlier_fw_mask);
            fprintf('图像 %d：反向估计失败，使用正向变换（内点 %d）\n', img_idx, sum(inlier_fw_mask));
            % 累积并继续
            if rcond(geo_tform.T) < 1e-12
                warning('警告：图像 "%s" 正向变换矩阵病态，沿用前一张变换', imgs_info(img_idx).name);
                tforms(img_idx) = tforms(img_idx - 1);
            else
                tforms(img_idx).T = geo_tform.T * tforms(img_idx - 1).T;
            end
            base_gray = current_gray;
            continue;
        end
    end

    inlier_fw_mask = logical(inlier_fw_mask);
    inlier_bw_mask = logical(inlier_bw_mask);

    % 检查内点数量
    inlier_count_fw = sum(inlier_fw_mask);
    inlier_count_bw = sum(inlier_bw_mask);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    if inlier_count_fw < 6 && inlier_count_bw < 6
        warning('警告：图像 "%s" 两方向内点均不足（%d / %d），沿用前一张变换', ...
                imgs_info(img_idx).name, inlier_count_fw, inlier_count_bw);
        tforms(img_idx) = tforms(img_idx - 1);
        base_gray = current_gray;
        continue;
    end

    % 方向一致性判定（比较矩阵差异）
    try
        diff_T = norm(inv(T_bw.T) - T_fw.T, 'fro');
    catch
        diff_T = inf;
    end

    if diff_T < tau
        geo_tform = T_fw;
        fprintf('图像 %d 方向一致（差异 %.4f），内点 (fw/bw) = (%d/%d)\n', img_idx, diff_T, inlier_count_fw, inlier_count_bw);
    else
        % 反馈修正：将两方向的内点联合起来，基于联合内点重新估计变换
        union_inliers = (inlier_fw_mask | inlier_bw_mask);
        union_idx = find(union_inliers);
        if numel(union_idx) >= 6
            union_current = matched_current_coords(union_idx, :);
            union_base = matched_base_coords(union_idx, :);
            try
                [refined_tform, refined_mask] = estimateGeometricTransform2D(...
                    union_current, union_base, 'similarity', 'Confidence', 99.9, 'MaxNumTrials', 3000);
                % 检查 ref 的内点数与条件数
                if sum(refined_mask) >= 6 && rcond(refined_tform.T) > 1e-12
                    geo_tform = refined_tform;
                    fprintf('图像 %d 反馈重估成功（联合内点 %d），原差异 %.4f\n', img_idx, sum(refined_mask), diff_T);
                else
                    % 反馈失败，fallback 到比较好的方向（选择内点更多的一侧）
                    if inlier_count_fw >= inlier_count_bw
                        geo_tform = T_fw;
                        fprintf('图像 %d 反馈重估失败，回退正向（%d 内点）\n', img_idx, inlier_count_fw);
                    else
                        geo_tform = invert(T_bw); % invert 返回 projective2d -> 调整为 current->base
                        fprintf('图像 %d 反馈重估失败，回退反向（%d 内点）\n', img_idx, inlier_count_bw);
                    end
                end
            catch
                % 重估异常，fallback
                if inlier_count_fw >= inlier_count_bw
                    geo_tform = T_fw;
                else
                    geo_tform = invert(T_bw);
                end
                warning('图像 %d 反馈重估异常，使用回退策略', img_idx);
            end
        else
            % 联合内点不足，回退到较好的方向
            if inlier_count_fw >= inlier_count_bw
                geo_tform = T_fw;
                fprintf('图像 %d 联合内点不足(%d)，回退正向\n', img_idx, numel(union_idx));
            else
                geo_tform = invert(T_bw);
                fprintf('图像 %d 联合内点不足(%d)，回退反向\n', img_idx, numel(union_idx));
            end
        end
    end

    %% 6. 重定位逻辑（与第一张基准图匹配，减少累积误差）
    if mod(img_idx, Nreset) == 0
        base_ref_gray = rgb2gray(base_img_double);
        base_ref_fast = detectFASTFeatures(base_ref_gray, 'MinContrast', minContrast);
        if base_ref_fast.Count > maxPoints
            base_ref_fast = selectStrongest(base_ref_fast, maxPoints);
        end
        base_ref_coords = base_ref_fast.Location;
        try
            [base_ref_sift, base_ref_sift_points] = extractFeatures(base_ref_gray, base_ref_coords, 'Method', 'SIFT');
        catch
            [base_ref_sift, base_ref_sift_points] = extractFeatures(base_ref_gray, base_ref_fast, 'Method', 'SIFT');
        end

        if isobject(base_ref_sift)
            base_ref_sift_data = base_ref_sift.Features;
        else
            base_ref_sift_data = base_ref_sift;
        end

        if isempty(base_ref_sift_data)
            warning('警告：基准图重定位时SIFT描述符为空，继续累积变换');
            tforms(img_idx).T = geo_tform.T * tforms(img_idx - 1).T;
            base_gray = current_gray;
            continue;
        end

        % 保证维度一致
        if size(base_ref_sift_data, 2) ~= size(current_sift_data, 2)
            min_dim = min(size(base_ref_sift_data,2), size(current_sift_data,2));
            base_ref_sift_data = base_ref_sift_data(:,1:min_dim);
            current_sift_data = current_sift_data(:,1:min_dim);
            if min_dim == 0
                warning('警告：重定位时特征描述符维度无效，继续累积变换');
                tforms(img_idx).T = geo_tform.T * tforms(img_idx - 1).T;
                base_gray = current_gray;
                continue;
            end
        end

        % 重定位匹配（手动比率测试）
        base_ref_sift_single = single(base_ref_sift_data')';
        es_ref = ExhaustiveSearcher(base_ref_sift_single);
        [ref_nearest_idx, ref_distances] = knnsearch(es_ref, current_sift_single, 'K', 2);
        ref_valid_matches = ref_distances(:,1) < ratioThreshold * ref_distances(:,2);
        ref_index_pairs = [ref_nearest_idx(ref_valid_matches,1), find(ref_valid_matches)];

        if size(ref_index_pairs,1) >= 6
            [~, base_ref_valid_idx] = ismembertol(base_ref_sift_points.Location, base_ref_coords, 1e-6, 'ByRows', true);
            base_ref_valid_coords = base_ref_coords(base_ref_valid_idx, :);
            ref_matched_base = base_ref_valid_coords(ref_index_pairs(:,1), :);
            ref_matched_current = current_valid_coords(ref_index_pairs(:,2), :);

            try
                [ref_tform, ref_inlier_mask] = estimateGeometricTransform2D(...
                    ref_matched_current, ref_matched_base, 'similarity', 'Confidence', 99.9, 'MaxNumTrials', 2000);
            catch
                try
                    [ref_tform, ref_inlier_mask] = estimateGeometricTransform2D(...
                        ref_matched_current, ref_matched_base, 'translation', 'Confidence', 99.9, 'MaxNumTrials', 2000);
                catch
                    ref_tform = [];
                end
            end

            if ~isempty(ref_tform) && sum(ref_inlier_mask) >= 6 && rcond(ref_tform.T) > 1e-12
                tforms(img_idx).T = ref_tform.T; % 用基准图变换重置
                fprintf('图像 %d 重定位成功（有效内点：%d）\n', img_idx, sum(ref_inlier_mask));
            else
                tforms(img_idx).T = geo_tform.T * tforms(img_idx - 1).T;
                warning('警告：图像 %d 重定位失败，继续累积变换', img_idx);
            end
        else
            tforms(img_idx).T = geo_tform.T * tforms(img_idx - 1).T;
            warning('警告：图像 %d 重定位匹配点不足，继续累积变换', img_idx);
        end
    else
        % 非重定位：累积变换（当前变换 × 历史总变换）
        if rcond(geo_tform.T) < 1e-12
            warning('警告：图像 "%s" 变换矩阵病态，沿用前一张变换', imgs_info(img_idx).name);
            tforms(img_idx) = tforms(img_idx - 1);
        else
            tforms(img_idx).T = geo_tform.T * tforms(img_idx - 1).T;
        end
    end

    % 更新基准图（下一轮用当前图作为基准）
    base_gray = current_gray;
end

%% ====== 第四步：计算全景图输出范围（确定画布大小） ======
x_ranges = zeros(img_count, 2);
y_ranges = zeros(img_count, 2);

for i = 1:img_count
    try
        [x_lim, y_lim] = outputLimits(tforms(i), [1, img_sizes(i, 2)], [1, img_sizes(i, 1)]);
        x_min_i = x_lim(1); x_max_i = x_lim(2);
        y_min_i = y_lim(1); y_max_i = y_lim(2);
    catch
        [x_min_i, x_max_i, y_min_i, y_max_i] = outputLimits(tforms(i), [1, img_sizes(i, 2)], [1, img_sizes(i, 1)]);
    end
    x_ranges(i, :) = [x_min_i, x_max_i];
    y_ranges(i, :) = [y_min_i, y_max_i];
end

global_x_min = min(x_ranges(:));
global_x_max = max(x_ranges(:));
global_y_min = min(y_ranges(:));
global_y_max = max(y_ranges(:));

panorama_width = round(global_x_max - global_x_min);
panorama_height = round(global_y_max - global_y_min);

if panorama_width <= 0 || panorama_height <= 0
    error('错误：计算的全景图尺寸无效（宽：%d，高：%d），请检查图像变换是否正常', ...
          panorama_width, panorama_height);
end

panorama_view = imref2d([panorama_height, panorama_width], [global_x_min, global_x_max], [global_y_min, global_y_max]);
fprintf('全景图尺寸：宽=%d像素，高=%d像素\n', panorama_width, panorama_height);

%% ====== 第五步：图像拼接与融合（亮度补偿 + 羽化融合） ======
pixel_accum = zeros([panorama_height, panorama_width, 3], 'double');
weight_accum = zeros([panorama_height, panorama_width], 'double');

for i = 1:img_count
    try
        current_img = im2double(imread(fullfile(folder, imgs_info(i).name)));
    catch ME
        warning('警告：拼接阶段无法读取图像 "%s"，将跳过！原因：%s', ...
                fullfile(folder, imgs_info(i).name), ME.message);
        continue;
    end

    warped_img = imwarp(current_img, tforms(i), 'OutputView', panorama_view);
    mask = imwarp(true(size(current_img,1), size(current_img,2)), tforms(i), 'OutputView', panorama_view);

    if all(isnan(warped_img(:)))
        continue;
    end
    warped_img(isnan(warped_img)) = 0;

    overlap_region = (weight_accum > 0) & mask;
    if any(overlap_region(:))
        pano_gray = rgb2gray(pixel_accum ./ max(weight_accum, eps));
        current_gray = rgb2gray(warped_img);
        mean_pano = mean(pano_gray(overlap_region));
        mean_current = mean(current_gray(overlap_region));
        if mean_current > 0
            gain = mean_pano / mean_current;
            warped_img = warped_img * gain;
        end
    end

    dist_map = bwdist(~mask);
    dist_map = dist_map / max(dist_map(:) + eps);
    dist_map_3d = repmat(dist_map, [1,1,3]);

    pixel_accum = pixel_accum + warped_img .* dist_map_3d;
    weight_accum = weight_accum + dist_map;
end

weight_accum(weight_accum == 0) = eps;
panorama = pixel_accum ./ repmat(weight_accum, [1,1,3]);
panorama_uint8 = im2uint8(mat2gray(panorama));

%% ====== 拼接完成后旋转展示 ======
% 假设拼接结果存放在 stitched_result 变量中
% 如果是文件，可用 imread('stitched_result.jpg') 代替

% if exist('panorama_uint8', 'var')
%     figure('Name', 'Rotated Results', 'NumberTitle', 'off');
%     num_steps = 360 / 40;  % 每次旋转40°，共9步
%     for i = 0:num_steps-1
%         angle = i * 40;
%         rotated_img = imrotate(panorama_uint8, angle, 'bicubic', 'crop');  % 以中心为轴旋转
%         subplot(3, 3, i+1);
%         imshow(rotated_img);
%         title([num2str(angle) '°']);
%     end
% else
%     warning('未检测到 stitched_result 变量，请确认拼接结果是否已生成。');
% end


%% ====== 第六步：显示与保存结果 ======
figure('Name', '全景拼接结果', 'Units', 'normalized', 'OuterPosition', [0 0 1 1]);
imshow(panorama_uint8);
title('最终全景拼接结果', 'FontSize', 14, 'FontWeight', 'bold');

if save_result
    [save_dir, ~, ~] = fileparts(output_file);
    if ~isempty(save_dir) && ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end
    imwrite(panorama_uint8, output_file);
    fprintf('拼接结果已保存至：%s\n', fullfile(pwd, output_file));
end

fprintf('图像拼接完成！\n');
