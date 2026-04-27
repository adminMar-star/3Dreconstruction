% run_feature_comparison_stitching.m
clc; clear;

folder = "D:\Image together\序列拼接\10.29\8";
output_folder = "D:\Image together\结果图像";
save_result = true;

imgs = readImages(folder);

methods = {'KAZE', 'SIFT', 'SURF', 'ORB', 'BRISK', 'MSER'};
results = cell(length(methods), 1);
runtimes = zeros(length(methods), 1);  % 用于存储每个方法的运行时间

for k = 1:length(methods)
    method = methods{k};
    fprintf('==== 处理算法: %s ====\n', method);
    
    timer_method = tic;  % ⏱️ 开始计时

    % --- 特征提取 ---
    [features_all, keypoints_all] = extractFeatures_All(imgs, method);

    % --- 图像拼接 ---
    result = bfsStitching(imgs, features_all, keypoints_all, @multiBandBlendRGB_Enhanced);

    runtimes(k) = toc(timer_method);  % ⏱️ 记录运行时间
    
    % --- 结果显示与保存 ---
     figure; imshow(result); title(['拼接结果 - ', method]);
    % if save_result
    %     imwrite(result, fullfile(output_folder, ['result_', lower(method), '.jpg']));
    % end
    
    % --- 质量指标 ---
    niqe_val = niqe(result);
    piqe_val = piqe(result);
    bris_val = brisque(result);
    
    fprintf('%s 拼接图像质量指标: NIQE=%.3f, PIQE=%.3f, BRISQUE=%.3f\n', ...
        method, niqe_val, piqe_val, bris_val);
    fprintf('%s 总运行时间: %.2f 秒\n\n', method, runtimes(k));

    % 保存结果
    results{k} = struct(...
        'method', method, ...
        'image', result, ...
        'niqe', niqe_val, ...
        'piqe', piqe_val, ...
        'brisque', bris_val, ...
        'runtime', runtimes(k));
end

%% ----------- 依赖函数 ------------

function imgs = readImages(folder)
    files = dir(fullfile(folder, '*.*g'));
    names = sort_nat({files.name});
    imgs = cellfun(@(n) im2double(imread(fullfile(folder, n))), names, 'UniformOutput', false);
end

function [features_all, keypoints_all] = extractFeatures_All(imgs, method)
    N = numel(imgs);
    features_all = cell(N,1);
    keypoints_all = cell(N,1);
    for i = 1:N
        gray = im2gray(imgs{i});
        switch upper(method)
            case 'KAZE'
                pts = detectKAZEFeatures(gray, 'Threshold', 0.001);
                [feats, pts] = extractFeatures(gray, pts);
            case 'SIFT'
                pts = detectSIFTFeatures(gray, 'ContrastThreshold', 0.01);
                [feats, pts] = extractFeatures(gray, pts);
            case 'SURF'
                pts = detectSURFFeatures(gray, 'MetricThreshold', 500);
                [feats, pts] = extractFeatures(gray, pts);
            case 'ORB'
                pts = detectORBFeatures(gray);
                [feats, pts] = extractFeatures(gray, pts);
            case 'BRISK'
                pts = detectBRISKFeatures(gray);
                [feats, pts] = extractFeatures(gray, pts);
            case 'MSER'
                pts = detectMSERFeatures(gray);
                [feats, pts] = extractFeatures(gray, pts);
            otherwise
                error('Unsupported feature extraction method: %s', method);
        end
        
        if isa(feats, 'binaryFeatures')
            feats = double(feats.Features);
        else
            feats = double(feats);
        end
        [pts_filtered, feats_filtered] = spatialNMS(pts, feats, 10);
        keypoints_all{i} = pts_filtered;
        features_all{i} = feats_filtered;
        fprintf('图像 %d - %s 特征点数: %d\n', i, method, pts_filtered.Count);
    end
end

function result = bfsStitching(imgs, features_all, keypoints_all, fusion_fn)
num_imgs = numel(imgs);
    canvas_size = round([sum(cellfun(@(x) size(x,1), imgs)), sum(cellfun(@(x) size(x,2), imgs))]*1.5);
    adj_matrix = zeros(num_imgs);
    
    % 创建匹配点可视化文件夹
    match_vis_folder = 'D:\Image together\匹配点可视化';
    if ~exist(match_vis_folder, 'dir')
        mkdir(match_vis_folder);
    end
    
    % 构建邻接图并可视化匹配点
    for i = 1:num_imgs
        for j = i+1:num_imgs
            indexPairs = matchFeatures(features_all{i}, features_all{j}, 'Unique', true, 'MaxRatio', 0.7);
            if size(indexPairs,1) >= 4
                adj_matrix(i,j) = size(indexPairs,1);
                adj_matrix(j,i) = size(indexPairs,1);
                
                % 可视化匹配点
                visualizeMatches(imgs{i}, imgs{j}, ...
                                keypoints_all{i}, keypoints_all{j}, ...
                                indexPairs, ...
                                fullfile(match_vis_folder, sprintf('match_%d_%d.jpg', i, j)));
            end
        end
    end

    canvas = zeros(canvas_size(1), canvas_size(2), 3);
    mask = zeros(canvas_size(1), canvas_size(2));
    positions = NaN(num_imgs, 2);
    used = false(num_imgs, 1);
    queue = [];

    root = ceil(num_imgs / 2);
    [h0, w0, ~] = size(imgs{root});
    pos = [canvas_size(2)/2 - w0/2, canvas_size(1)/2 - h0/2];
    positions(root,:) = pos; used(root) = true; queue(end+1) = root;
    canvas(round(pos(2))+(1:h0), round(pos(1))+(1:w0), :) = imgs{root};
    mask(round(pos(2))+(1:h0), round(pos(1))+(1:w0)) = 1;
    
    while ~isempty(queue)
        current = queue(1); queue(1) = [];
        for neighbor = find(adj_matrix(current,:) > 0)
            if used(neighbor), continue; end
            indexPairs = matchFeatures(features_all{current}, features_all{neighbor}, 'Unique', true, 'MaxRatio', 0.5);
            if size(indexPairs,1) < 4, continue; end

            matched1 = keypoints_all{current}(indexPairs(:,1));
            matched2 = keypoints_all{neighbor}(indexPairs(:,2));

            try
                [tform, inlierIdx] = estimateGeometricTransform2D(matched2, matched1, 'similarity');
            catch
                continue;
            end
            
            fprintf('匹配点数: %d, 内点数: %d\n', size(indexPairs,1), sum(inlierIdx));

            offset = transformPointsForward(tform, [0 0]);
            pos1 = positions(current,:);
            pos2 = pos1 + offset;
            [h2, w2, ~] = size(imgs{neighbor});
            x2 = round(pos2(1)); y2 = round(pos2(2));

            if y2 <= 0 || x2 <= 0 || y2+h2-1 > canvas_size(1) || x2+w2-1 > canvas_size(2)
                continue;
            end

            region = canvas(y2:y2+h2-1, x2:x2+w2-1, :);
            maskA = mask(y2:y2+h2-1, x2:x2+w2-1);
            maskB = ones(h2, w2);
            blended = fusion_fn(region, maskA, imgs{neighbor}, maskB, 5);

            canvas(y2:y2+h2-1, x2:x2+w2-1, :) = blended;
            mask(y2:y2+h2-1, x2:x2+w2-1) = 1;
            positions(neighbor,:) = [x2, y2];
            used(neighbor) = true; queue(end+1) = neighbor;
        end
    end

    [row, col] = find(mask > 0);
    if ~isempty(row)
        result = canvas(min(row):max(row), min(col):max(col), :);
    else
        result = canvas;
    end
end

function blended = multiBandBlendRGB_Enhanced(imgA, maskA, imgB, maskB, levels)
    blended = zeros(size(imgA));
    for ch = 1:3
        blended(:,:,ch) = multiBandBlend_Enhanced(imgA(:,:,ch), maskA, imgB(:,:,ch), maskB, levels);
    end
end

function blended = multiBandBlend_Enhanced(imgA, maskA, imgB, maskB, levels)
    if nargin < 5, levels = 5; end
    imgA = im2double(imgA); imgB = im2double(imgB);
    maskA = im2double(maskA); maskB = im2double(maskB);

    gaussA = generateWeight(size(maskA), 0.6);
    gaussB = generateWeight(size(maskB), 0.6);
    distA = dist2border(maskA); distB = dist2border(maskB);
    distA = distA / (max(distA(:)) + eps);
    distB = distB / (max(distB(:)) + eps);
    maskA_weight = maskA .* gaussA .* distA;
    maskB_weight = maskB .* gaussB .* distB;

    weight = maskA_weight ./ (maskA_weight + maskB_weight + eps);

    LA = buildLaplacianPyramid(imgA, levels);
    LB = buildLaplacianPyramid(imgB, levels);
    GW = buildGaussianPyramid(weight, levels);
    
    blendedL = blendLaplacianPyramids(LA, LB, GW);
    blended = reconstructFromLaplacianPyramid(blendedL);
    blended = min(max(blended, 0), 1);
end

function dist_map = dist2border(mask)
    border = (mask == 0);
    border(1,:) = 1; border(end,:) = 1;
    border(:,1) = 1; border(:,end) = 1;
    dist_map = bwdist(border, 'chessboard');
    dist_map = double(dist_map);
end

function w = generateWeight(sz, sigma)
    if nargin < 2, sigma = 0.5; end
    [h, w_] = deal(sz(1), sz(2));
    [X, Y] = meshgrid(linspace(-1,1,w_), linspace(-1,1,h));
    dist = sqrt(X.^2 + Y.^2);
    w = exp(-dist.^2 / (2*sigma^2));
end

function G = buildGaussianPyramid(I, levels)
    G = cell(levels, 1);
    G{1} = I;
    for i = 2:levels
        G{i} = impyramid(G{i-1}, 'reduce');
    end
end

function L = buildLaplacianPyramid(I, levels)
    G = buildGaussianPyramid(I, levels);
    L = cell(levels,1);
    for i = 1:levels-1
        expanded = imresize(impyramid(G{i+1}, 'expand'), size(G{i}));
        L{i} = G{i} - expanded;
    end
    L{levels} = G{levels};
end

function blendedL = blendLaplacianPyramids(LA, LB, GW)
    levels = length(LA);
    blendedL = cell(levels, 1);
    for i = 1:levels
        blendedL{i} = LA{i} .* GW{i} + LB{i} .* (1 - GW{i});
    end
end

function Irec = reconstructFromLaplacianPyramid(L)
    levels = length(L);
    Irec = L{levels};
    for i = levels-1:-1:1
        Irec = imresize(impyramid(Irec, 'expand'), size(L{i}));
        Irec = Irec + L{i};
    end
end

function [filtered_pts, filtered_feats] = spatialNMS(pts, feats, radius)
    locs = round(pts.Location);
    selected = false(pts.Count, 1);
    used = false(size(locs,1),1);
    for i = 1:size(locs,1)
        if used(i), continue; end
        dists = sqrt(sum((locs - locs(i,:)).^2,2));
        near = dists < radius;
        near_idx = find(near);
        [~, best] = max(pts.Metric(near_idx));
        selected(near_idx(best)) = true;
        used(near_idx) = true;
    end
    filtered_pts = pts(selected);
    filtered_feats = feats(selected,:);
end

function [cs,index] = sort_nat(c)
    [~,index] = sort(lower(regexprep(c,'\\d+','${num2str(str2double($0),''%010d'')}')));
    cs = c(index);
end
function visualizeMatches(img1, img2, pts1, pts2, matches, savepath)
    % 转换为RGB图像（如果是灰度图）
    if size(img1,3) == 1
        img1 = repmat(img1, [1 1 3]);
    end
    if size(img2,3) == 1
        img2 = repmat(img2, [1 1 3]);
    end
    
    % 创建拼接后的画布
    [h1, w1, ~] = size(img1);
    [h2, w2, ~] = size(img2);
    canvas = zeros(max(h1,h2), w1+w2, 3);
    canvas(1:h1, 1:w1, :) = img1;
    canvas(1:h2, w1+1:w1+w2, :) = img2;
    
    figure('Visible', 'off'); % 不显示图形窗口
    imshow(canvas); hold on;
    
    % 绘制关键点
    plot(pts1.selectStrongest(50), 'showOrientation', true);
    plot(pts2.selectStrongest(50).Location(:,1)+w1, pts2.selectStrongest(50).Location(:,2), 'g+');
    
    % 绘制匹配线
    matchedPts1 = pts1(matches(:,1)).Location;
    matchedPts2 = pts2(matches(:,2)).Location;
    
    for k = 1:size(matches,1)
        line([matchedPts1(k,1), matchedPts2(k,1)+w1], ...
             [matchedPts1(k,2), matchedPts2(k,2)], ...
             'Color', 'y', 'LineWidth', 1.5);
    end
    
    title(sprintf('匹配点数量: %d', size(matches,1)));
    hold off;
    
    % 保存图像
    frame = getframe(gca);
    imwrite(frame.cdata, savepath);
    close(gcf);
end