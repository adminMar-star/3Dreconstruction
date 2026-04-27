clc; clear; close all;

%% ===================== 参数 =====================
folder = "D:\Image together\序列拼接\10.29\4\2";
save_result = false;
output_file = "stitched_no_reloc_no_feedback.jpg";

maxPoints = 2000;
ratioThreshold = 0.75;
minContrast = 0.03;

image_extensions = {'*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff'};

%% ===================== 读取并排序图像 =====================
imgs_info = [];
for ext = image_extensions
    imgs_info = [imgs_info; dir(fullfile(folder, ext{1}))];
end
imgs_info = imgs_info(~[imgs_info.isdir]);
assert(~isempty(imgs_info), '未找到图像');

names = {imgs_info.name};
nums = zeros(numel(names),1);
for i = 1:numel(names)
    m = regexp(names{i}, '\d+', 'match');
    nums(i) = isempty(m)*i + ~isempty(m)*str2double(m{end});
end
[~,idx] = sort(nums);
imgs_info = imgs_info(idx);
img_count = numel(imgs_info);

%% ===================== 初始化 =====================
ref_img  = im2double(imread(fullfile(folder, imgs_info(1).name)));
ref_gray = rgb2gray(ref_img);

tforms = repmat(projective2d(eye(3)), 1, img_count);
img_sizes = zeros(img_count,2);
img_sizes(1,:) = size(ref_gray);

%% ===================== 逐帧估计（无重定位 + 无反馈） =====================
for img_idx = 2:img_count

    current = im2double(imread(fullfile(folder, imgs_info(img_idx).name)));
    current_gray = rgb2gray(current);
    img_sizes(img_idx,:) = size(current_gray);

    %% FAST
    p_ref = detectFASTFeatures(ref_gray,'MinContrast',minContrast);
    p_cur = detectFASTFeatures(current_gray,'MinContrast',minContrast);

    if p_ref.Count > maxPoints
        p_ref = selectStrongest(p_ref, maxPoints);
    end
    if p_cur.Count > maxPoints
        p_cur = selectStrongest(p_cur, maxPoints);
    end

    %% === SIFT 安全接口（numeric 坐标） ===
    loc1 = p_ref.Location;
    loc2 = p_cur.Location;

    if size(loc1,1) < 6 || size(loc2,1) < 6
        tforms(img_idx) = tforms(img_idx-1);
        continue;
    end

    [f1, v1] = extractFeatures(ref_gray,     loc1, 'Method','SIFT');
    [f2, v2] = extractFeatures(current_gray, loc2, 'Method','SIFT');

    if isempty(f1) || isempty(f2)
        tforms(img_idx) = tforms(img_idx-1);
        continue;
    end

    %% Lowe 匹配
    es = ExhaustiveSearcher(single(f1));
    [idx_nn, dist] = knnsearch(es, single(f2), 'K', 2);
    good = dist(:,1) < ratioThreshold * dist(:,2);

    if sum(good) < 6
        tforms(img_idx) = tforms(img_idx-1);
        continue;
    end

    mp_ref = v1(idx_nn(good,1)).Location;
    mp_cur = v2(find(good)).Location;

    %% === 单向 RANSAC（无反馈修正） ===
    try
        geo_tform = estimateGeometricTransform2D( ...
            mp_cur, mp_ref, 'similarity', ...
            'Confidence',99.9,'MaxNumTrials',2000);
    catch
        tforms(img_idx) = tforms(img_idx-1);
        continue;
    end

    %% === 纯累计（无重定位） ===
    T_new = geo_tform.T * tforms(img_idx-1).T;

    if rcond(T_new) < 1e-12
        tforms(img_idx) = tforms(img_idx-1);
    else
        tforms(img_idx).T = T_new;
    end
end

%% ===================== 画布 =====================
xlim = zeros(img_count,2);
ylim = zeros(img_count,2);
for i = 1:img_count
    [xlim(i,:), ylim(i,:)] = outputLimits( ...
        tforms(i), [1 img_sizes(i,2)], [1 img_sizes(i,1)]);
end

panoView = imref2d( ...
    round([max(ylim(:))-min(ylim(:)), max(xlim(:))-min(xlim(:))]), ...
    [min(xlim(:)) max(xlim(:))], ...
    [min(ylim(:)) max(ylim(:))]);

%% ===================== 融合 =====================
acc = zeros([panoView.ImageSize 3],'double');
w   = zeros(panoView.ImageSize,'double');

for i = 1:img_count
    I = im2double(imread(fullfile(folder, imgs_info(i).name)));
    W = imwarp(I, tforms(i), 'OutputView', panoView);
    M = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoView);

    d = bwdist(~M);
    d = d ./ max(d(:) + eps);

    acc = acc + W .* d;
    w   = w   + d;
end

panorama = acc ./ max(w,eps);
panorama_uint8 = im2uint8(mat2gray(panorama));

%% ===================== 显示 =====================
figure('Name','No Reloc + No Feedback','Units','normalized','OuterPosition',[0 0 1 1]);
imshow(panorama_uint8);
title('拼接结果（无重定位 + 无反馈修正）','FontSize',14,'FontWeight','bold');

if save_result
    imwrite(panorama_uint8, output_file);
end

fprintf('✔ Exp-3（无重定位 + 无反馈，SIFT 安全版）完成\n');
