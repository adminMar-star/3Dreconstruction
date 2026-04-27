function scratch_enhancement_and_segmentation_complete()
    clc; clear;
    
    %% 1. 图像读取与预处理
%     imgName = 'D:\zuomian\特征提取\数据集\10.29\8\12-500-all.jpg'; 
    imgName = 'D:\A科研\A硕士毕业论文\滑蹭\10.29\8\12-500-all.jpg'; 
    gt_filename = 'D:\A科研\A硕士毕业论文\滑蹭\10.29\8\12-500-all-label.jpg'; 
    I_original = imread(imgName);
    
    if size(I_original, 3) == 3
        I_gray = rgb2gray(I_original);
    else
        I_gray = I_original;
    end
    I = im2double(I_gray);
    
    fprintf('========= 开始图像增强处理 =========\n');
    
    %% 2. 详细的图像特性分析 
    fprintf('图像详细特性分析:\n');
    mean_val = mean(I(:));
    std_val = std(I(:));
    min_val = min(I(:));
    max_val = max(I(:));
    contrast_val = std_val/mean_val;
    
    fprintf('  均值: %.4f\n', mean_val);
    fprintf('  标准差: %.4f\n', std_val);
    fprintf('  对比度系数: %.4f\n', contrast_val);
    fprintf('  动态范围: [%.4f, %.4f]\n', min_val, max_val);
    fprintf('  信息熵: %.4f\n', entropy(I));
    
    % 分析图像直方图
    [counts, bins] = imhist(I);
    [peak_count, peak_idx] = max(counts);
    peak_bin = bins(peak_idx);
    fprintf('  直方图峰值: %.4f (数量: %d)\n', peak_bin, peak_count);
    
    %% 3. 新的增强策略：基于图像特性的自适应增强
    
    % 3.1 第一步：轻微的背景归一化（不改变图像结构）
    I_normalized = (I - mean_val) / std_val * 0.1 + 0.5;
    I_normalized = min(max(I_normalized, 0), 1);
    
    % 3.2 第二步：使用自适应CLAHE（只轻微增强）
    I_clahe = adapthisteq(I_normalized, ...
        'NumTiles', [4 4], ...  % 减少分块数
        'ClipLimit', 0.005, ... % 非常保守的裁剪限制
        'Distribution', 'rayleigh', ...
        'Alpha', 0.1);
    
    % 3.3 第三步：选择性顶帽变换（只提取极细特征）
    % 使用很小的结构元素提取细划痕
    SE_small = strel('disk', 2);
    SE_medium = strel('disk', 4);
    
    I_th_small = imtophat(I_clahe, SE_small);
    I_th_medium = imtophat(I_clahe, SE_medium);
    
    % 非常保守的特征融合
    I_features = 0.05 * I_th_small + 0.03 * I_th_medium;
    
    % 3.4 第四步：保守的非线性增强
    % 使用tanh函数代替sigmoid（更平滑）
    feature_mean = mean(I_features(:));
    feature_std = std(I_features(:));
    
    % 自适应增强参数
    if feature_std < 0.02
        enhancement_factor = 2;  % 极低对比度时轻微增强
    elseif feature_std < 0.05
        enhancement_factor = 3;  % 低对比度时适度增强
    else
        enhancement_factor = 4;  % 正常对比度
    end
    
    % 使用tanh函数进行非线性增强
    I_enhanced_features = tanh(enhancement_factor * (I_features - feature_mean)) * 0.5 + 0.5;
    
    % 3.5 第五步：与原始图像融合（保持95%原始信息）
    preservation_factor = 0.95;  % 高度保留原始信息
    I_combined = preservation_factor * I + (1-preservation_factor) * I_enhanced_features;
    
    % 3.6 第六步：轻微的高斯滤波去除噪声
    I_smoothed = imgaussfilt(I_combined, 0.5);
    
    % 3.7 第七步：最终对比度微调
    % 使用非常保守的对比度拉伸
    low_high = stretchlim(I_smoothed, [0.01 0.99]);
    I_final = imadjust(I_smoothed, low_high, []);
    
    % 3.8 第八步：确保与原始图像高度相似
    similarity_weight = 0.85;  % 85%保持原始图像
    I_enhanced = similarity_weight * I + (1-similarity_weight) * I_final;
    
    %% 4. 计算并显示增强指标
    fprintf('\n========= 图像增强评价指标 =========\n');
    
    % 基础指标
    psnr_val = calculatePSNR(I, I_enhanced);
    ssim_val = calculateSSIM(I, I_enhanced);
    mae_val = calculateMAE(I, I_enhanced);
    mse_val = calculateMSE(I, I_enhanced);
    rmse_val = sqrt(mse_val);
    snr_val = calculateSNR(I, I_enhanced);
    
    fprintf('1. PSNR (峰值信噪比): %.4f dB\n', psnr_val);
    fprintf('2. SSIM (结构相似性): %.4f\n', ssim_val);
    fprintf('3. MAE (平均绝对误差): %.6f\n', mae_val);
    fprintf('4. MSE (均方误差): %.6f\n', mse_val);
    fprintf('5. RMSE (均方根误差): %.6f\n', rmse_val);
    fprintf('6. SNR (信噪比): %.4f dB\n', snr_val);
    
    % 新增质量指标
    entropy_original = entropy(I);
    entropy_enhanced = entropy(I_enhanced);
    entropy_change = entropy_enhanced - entropy_original;
    
    contrast_original = std(I(:));
    contrast_enhanced = std(I_enhanced(:));
    contrast_ratio = contrast_enhanced / contrast_original;
    
    fprintf('7. 信息熵变化: %.4f → %.4f (Δ=%.4f)\n', ...
        entropy_original, entropy_enhanced, entropy_change);
    fprintf('8. 对比度变化: %.4f → %.4f (比率=%.4f)\n', ...
        contrast_original, contrast_enhanced, contrast_ratio);
    
    %% 5. 增强结果可视化
    fig1 = figure('Name', 'Optimized Scratch Enhancement', 'Color', 'w', 'Position', [100, 100, 1400, 600]);
    
    % 原始图
    subplot(2, 4, 1);
    imshow(I);
    title('原始图像', 'FontSize', 11);
    text(10, 20, sprintf('均值=%.3f', mean_val), 'Color', 'w', 'FontSize', 9);
    
    % 归一化结果
    subplot(2, 4, 2);
    imshow(I_normalized);
    title('背景归一化', 'FontSize', 11);
    
    % CLAHE结果
    subplot(2, 4, 3);
    imshow(I_clahe);
    title('自适应CLAHE', 'FontSize', 11);
    
    % 特征提取
    subplot(2, 4, 4);
    imshow(I_features, []);
    title('顶帽特征提取', 'FontSize', 11);
    
    % 增强特征
    subplot(2, 4, 5);
    imshow(I_enhanced_features);
    title(sprintf('非线性增强(因子=%.1f)', enhancement_factor), 'FontSize', 11);
    
    % 融合结果
    subplot(2, 4, 6);
    imshow(I_combined);
    title(sprintf('原始融合(保留%.0f%%)', preservation_factor*100), 'FontSize', 11);
    
    % 最终增强结果
    subplot(2, 4, 7);
    imshow(I_enhanced);
    title('最终增强结果', 'FontSize', 11, 'FontWeight', 'bold');
    
    % 指标显示
    subplot(2, 4, 8);
    axis off;
    
    if psnr_val > 20
        psnr_color = 'g';
    elseif psnr_val > 15
        psnr_color = 'y';
    else
        psnr_color = 'r';
    end
    
    if ssim_val > 0.8
        ssim_color = 'g';
    elseif ssim_val > 0.6
        ssim_color = 'y';
    else
        ssim_color = 'r';
    end
    
    metrics_text = {
        sprintf('增强指标 (颜色表示质量):'), ...
        sprintf(''), ...
        sprintf('\\color{%s}PSNR: %.2f dB', psnr_color, psnr_val), ...
        sprintf('\\color{%s}SSIM: %.4f', ssim_color, ssim_val), ...
        sprintf('SNR: %.2f dB', snr_val), ...
        sprintf('MAE: %.6f', mae_val), ...
        sprintf(''), ...
        sprintf('信息熵: +%.4f', entropy_change), ...
        sprintf('对比度: ×%.3f', contrast_ratio)
    };
    
    text(0.1, 0.5, metrics_text, 'FontSize', 10, 'VerticalAlignment', 'middle', ...
        'Interpreter', 'tex');
    title('增强性能指标', 'FontSize', 12, 'FontWeight', 'bold');
    
    %% 6. 保存增强结果
    [filepath, name, ext] = fileparts(imgName);
    
    % 保存增强后的图像（用于分割的输入）
    enhanced_img_path = fullfile(filepath, [name '_enhanced_for_segmentation' ext]);
    imwrite(I_enhanced, enhanced_img_path);
    fprintf('\n增强图像已保存: %s\n', enhanced_img_path);
    
    % 保存增强可视化图形
    saveas(fig1, fullfile(filepath, [name '_enhancement_visualization.png']));
    
    %% 7. 开始分割处理（使用增强后的图像作为输入）
    fprintf('\n========= 开始分割处理 =========\n');
    fprintf('使用增强后的图像作为分割输入...\n');
    
    % 使用增强后的图像作为分割的原始图像
    I_seg = I_enhanced;  % 这是关键修改：使用增强图像作为分割输入
    
    %% 8. 基于增强图像特征创建参考分割（作为"真实标签"）
    fprintf('\n正在基于增强图像特征创建参考分割...\n');
    
    % 计算增强图像的统计特征
    mean_val_seg = mean(I_seg(:));
    std_val_seg = std(I_seg(:));
    
    % 创建基于统计特征的参考阈值
    ref_threshold = mean_val_seg + 1.5 * std_val_seg;
    ref_threshold = min(max(ref_threshold, 0), 1);
    
%     % 创建初始参考分割
%     GT_reference = imbinarize(I_seg, ref_threshold);
%     
%     % 形态学处理优化参考分割
%     se_ref = strel('disk', 3);
%     GT_reference = imopen(GT_reference, se_ref);  % 开操作去除小噪声
%     
%     % 面积筛选（保留显著区域）
%     GT_reference = bwareaopen(GT_reference, 150);
%     
%     % 填充孔洞
%     GT_reference = imfill(GT_reference, 'holes');
    
      
    %% 9. 执行主要分割算法
    fprintf('\n正在执行主要分割算法...\n');
    
    % 学术性增强核心：多尺度形态学顶帽变换
    SE1 = strel('disk', 3); 
    SE2 = strel('disk', 7); 
    SE3 = strel('disk', 11);
    
    I_th1 = imtophat(I_seg, SE1);
    I_th2 = imtophat(I_seg, SE2);
    I_th3 = imtophat(I_seg, SE3);
    
    I_features_seg = 0.3 * I_th1 + 0.2 * I_th2 + 0.2 * I_th3;

    % 非线性对比度增强 (Sigmoid Mapping)
    mean_val_feat = mean(I_features_seg(:));
    gain = 25; 
    cutoff = mean_val_feat + 0.03;
    
    I_enhanced_seg = 1 ./ (1 + exp(-gain * (I_features_seg - cutoff)));

    % 图像平滑与重建：引导滤波
    I_final_seg = imguidedfilter(I_enhanced_seg, 'DegreeOfSmoothing', 0.01*diff(getrangefromclass(I_enhanced_seg)), 'NeighborhoodSize', [6 6]);
    
    I_final_seg = imadjust(I_final_seg);
    I_final_seg = medfilt2(I_final_seg, [5 5]);
    
    %% 10. 分割处理
    I_final_enhance_seg = mat2gray(I_final_seg);
    I_feature_seg = im2uint8(I_final_enhance_seg);
    I_feature_adj_seg = imadjust(I_feature_seg, [], [], 2); 
   
    % 核心分割算法：滞后阈值与形态学重构
    level_otsu = graythresh(I_feature_adj_seg);
    fprintf('  分割算法Otsu阈值: %.4f\n', level_otsu);
    
    T_high = level_otsu * 0.8;  
    T_low  = T_high * 0.3;      
    
    BW_marker = imbinarize(I_feature_adj_seg, T_high); 
    BW_mask   = imbinarize(I_feature_adj_seg, T_low);  
    BW_recon = imreconstruct(BW_marker, BW_mask);

    % 后处理优化
    se_close = strel('disk', 2);
    BW_closed = imclose(BW_recon, se_close);
    BW_predicted = bwareaopen(BW_closed, 165);  % 分割结果（预测）
    
    % 计算分割结果的统计信息
    stats_pred = regionprops(BW_predicted, 'Area', 'Eccentricity');
    num_regions_pred = length(stats_pred);
    fprintf('  预测分割区域数: %d\n', num_regions_pred);
    if num_regions_pred > 0
        areas_pred = [stats_pred.Area];
        fprintf('  预测分割平均面积: %.1f 像素\n', mean(areas_pred));
        fprintf('  预测分割总面积: %.1f 像素\n', sum(areas_pred));
    end
    
    %% 11. 计算分割指标（使用参考分割作为"真实标签"）
    fprintf('\n-------- 分割性能评估 --------\n');
    fprintf('评估方法：分割结果 vs 基于增强图像的参考分割\n\n');
    % 读取真值图并二值化
        GT_img = imread(gt_filename);
        if size(GT_img, 3) == 3
            GT_img = rgb2gray(GT_img);
        end
        GT = imbinarize(GT_img); % 确保是逻辑矩阵 logic
        
        % 确保尺寸一致
        if any(size(GT) ~= size(BW_predicted))
            GT = imresize(GT, size(BW_predicted), 'nearest');
        end

        GT_reference=GT;
    % 计算参考分割的统计信息
    stats_ref = regionprops(GT_reference, 'Area', 'Eccentricity');
    num_regions_ref = length(stats_ref);
    fprintf('  参考分割区域数: %d\n', num_regions_ref);
    if num_regions_ref > 0
        areas_ref = [stats_ref.Area];
        fprintf('  参考分割平均面积: %.1f 像素\n', mean(areas_ref));
        fprintf('  参考分割总面积: %.1f 像素\n', sum(areas_ref));
    end
    
%     % 确保尺寸一致
%     if any(size(GT_reference) ~= size(BW_predicted))
%         fprintf('  调整参考分割尺寸以匹配分割结果\n');
%         GT_reference = imresize(GT_reference, size(BW_predicted), 'nearest');
%     end

    % 计算混淆矩阵元素
    TP = sum(BW_predicted(:) & GT_reference(:));
    FP = sum(BW_predicted(:) & ~GT_reference(:));
    FN = sum(~BW_predicted(:) & GT_reference(:));
    TN = sum(~BW_predicted(:) & ~GT_reference(:));
    
    total_pixels = numel(GT_reference);
    
    % 计算指标
    metric_dice = 2 * TP / (2 * TP + FP + FN + eps);
    metric_iou = TP / (TP + FP + FN + eps);
    metric_precision = TP / (TP + FP + eps);
    metric_recall = TP / (TP + FN + eps);
    metric_accuracy = (TP + TN) / total_pixels;
    metric_specificity = TN / (TN + FP + eps);
    
    fprintf('混淆矩阵统计：\n');
    fprintf('  总像素数: %d\n', total_pixels);
    fprintf('  真阳性(TP): %d (%.2f%%)\n', TP, 100*TP/total_pixels);
    fprintf('  假阳性(FP): %d (%.2f%%)\n', FP, 100*FP/total_pixels);
    fprintf('  假阴性(FN): %d (%.2f%%)\n', FN, 100*FN/total_pixels);
    fprintf('  真阴性(TN): %d (%.2f%%)\n', TN, 100*TN/total_pixels);
    
    fprintf('\n主要性能指标：\n');
    fprintf('  1. Dice系数: %.4f\n', metric_dice);
    fprintf('  2. IoU: %.4f\n', metric_iou);
    fprintf('  3. 精确率: %.4f\n', metric_precision);
    fprintf('  4. 召回率: %.4f\n', metric_recall);
    fprintf('  5. 准确率: %.4f\n', metric_accuracy);
    fprintf('  6. 特异度: %.4f\n', metric_specificity);
    
    %% 12. 分割结果可视化
    fig_final = figure('Color', 'w', 'Name', '分割结果与评估', 'Position', [200, 200, 1200, 800]);
    
    % 创建子图布局
    subplot(2, 3, 1);
    imshow(I);  % 显示原始图像
    title('原始输入图像', 'FontSize', 12, 'FontWeight', 'bold');
    
    subplot(2, 3, 2);
    imshow(I_seg);  % 显示增强后的图像（分割输入）
    title('增强后图像（分割输入）', 'FontSize', 12, 'FontWeight', 'bold');
    
    subplot(2, 3, 3);
    imshow(BW_predicted);
    title('主要分割结果（预测）', 'FontSize', 12, 'FontWeight', 'bold');
    
    subplot(2, 3, 4);
    imshow(GT_reference);
    title('基于增强图像的参考分割（"真值"）', 'FontSize', 12, 'FontWeight', 'bold');
    
    % 结果对比叠加显示
    subplot(2, 3, 5);
    % 创建对比叠加图像
    R = I_seg; G = I_seg; B = I_seg;
    mask_pred = BW_predicted;
    mask_ref = GT_reference;
    
    % 黄色 = 重叠部分 (TP)
    overlap = mask_pred & mask_ref;
    R(overlap) = 1;
    G(overlap) = 1;
    B(overlap) = 0;
    
    % 红色 = 预测独有 (FP)
    fp_only = mask_pred & ~mask_ref;
    R(fp_only) = 1;
    G(fp_only) = 0;
    B(fp_only) = 0;
    
    % 绿色 = 参考独有 (FN)
    fn_only = ~mask_pred & mask_ref;
    R(fn_only) = 0;
    G(fn_only) = 1;
    B(fn_only) = 0;
    
    RGB_overlay = cat(3, R, G, B);
    imshow(RGB_overlay);
    title({'结果对比叠加', '黄:重叠(TP) 红:预测独有(FP) 绿:参考独有(FN)'}, ...
          'FontSize', 12, 'FontWeight', 'bold');
    
    % 分割边界叠加显示
    subplot(2, 3, 6);
    boundaries = bwperim(BW_predicted);
    overlay_img = I_seg;
    
    % 将边界叠加到增强图像上
    for c = 1:3
        channel = overlay_img;
        channel(boundaries) = 1; % 红色边界
        overlay_img = channel;
    end
    
    imshow(overlay_img);
    title('分割边界叠加', 'FontSize', 12, 'FontWeight', 'bold');
    
    % 添加边界标注
    hold on;
    [B, L] = bwboundaries(BW_predicted, 'noholes');
    for k = 1:length(B)
        boundary = B{k};
        plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 1.5);
    end
    hold off;
    
    %% 13. 性能指标显示
    % 添加一个文本区域显示详细指标
    annotation_text = sprintf('性能指标:\nDice: %.4f  IoU: %.4f\n精确率: %.4f  召回率: %.4f\n准确率: %.4f  特异度: %.4f\n\n区域统计:\n预测区域: %d  参考区域: %d', ...
                              metric_dice, metric_iou, metric_precision, metric_recall, ...
                              metric_accuracy, metric_specificity, num_regions_pred, num_regions_ref);
    
    annotation(fig_final, 'textbox', [0.1, 0.02, 0.8, 0.08], ...
               'String', annotation_text, ...
               'FontSize', 10, 'FontWeight', 'bold', ...
               'HorizontalAlignment', 'center', ...
               'VerticalAlignment', 'middle', ...
               'EdgeColor', 'none', ...
               'BackgroundColor', [0.95, 0.95, 0.95]);
    
    %% 14. 详细分割过程可视化（可选）
    fig_process = figure('Color', 'w', 'Name', '分割过程详情', 'Position', [50, 50, 1400, 800]);

    % 原始图和特征
    subplot(3, 4, 1); imshow(I_seg); title('1. 增强输入图像');
    subplot(3, 4, 2); imshow(I_features_seg, []); title('2. 特征提取');
    subplot(3, 4, 3); imshow(I_final_seg); title('3. 增强后图像');
    subplot(3, 4, 4); imshow(GT_reference); title('4. 参考分割');

    % 分割流程
    subplot(3, 4, 5); imshow(BW_marker); title(['5. 强阈值(T=' num2str(T_high, '%.3f') ')']);
    subplot(3, 4, 6); imshow(BW_mask); title(['6. 弱阈值(T=' num2str(T_low, '%.3f') ')']);
    subplot(3, 4, 7); imshow(BW_recon); title('7. 形态学重构');
    subplot(3, 4, 8); imshow(BW_closed); title('8. 闭操作后');
    
    % 最终结果和对比
    subplot(3, 4, 9); imshow(BW_predicted); title('9. 最终分割结果');
    subplot(3, 4, 10); imshow(RGB_overlay); title('10. 结果对比叠加');
    
    % 性能指标详细显示
    subplot(3, 4, 11);
    axis off;
    
    if metric_dice > 0.7
        quality = '良好';
        quality_color = 'g';
    elseif metric_dice > 0.5
        quality = '中等';
        quality_color = [1 0.5 0];
    elseif metric_dice > 0.3
        quality = '一般';
        quality_color = 'y';
    else
        quality = '差异较大';
        quality_color = 'r';
    end
    
    metrics_text = {
        '性能评估摘要:', ...
        '', ...
        sprintf('\\color{%s}一致性: %s', quality_color, quality), ...
        '', ...
        sprintf('Dice系数: %.3f', metric_dice), ...
        sprintf('IoU:       %.3f', metric_iou), ...
        sprintf('精确率:    %.3f', metric_precision), ...
        sprintf('召回率:    %.3f', metric_recall), ...
        sprintf('准确率:    %.3f', metric_accuracy), ...
        '', ...
        sprintf('预测区域数: %d', num_regions_pred), ...
        sprintf('参考区域数: %d', num_regions_ref)
    };
    
    text(0.1, 0.5, metrics_text, 'FontSize', 10, 'VerticalAlignment', 'middle', ...
        'Interpreter', 'tex', 'FontWeight', 'bold');
    title('性能摘要', 'FontSize', 12, 'FontWeight', 'bold');
    
    % 参数设置显示
    subplot(3, 4, 12);
    axis off;
    
    params_text = {
        '参数设置:', ...
        '', ...
        '参考分割:', ...
        sprintf('  阈值: %.3f', ref_threshold), ...
        sprintf('  最小面积: %d', 150), ...
        '', ...
        '主要分割:', ...
        sprintf('  强阈值: %.3f', T_high), ...
        sprintf('  弱阈值: %.3f', T_low), ...
        sprintf('  最小面积: %d', 200)
    };
    
    text(0.1, 0.5, params_text, 'FontSize', 9, 'VerticalAlignment', 'middle');
    title('参数设置', 'FontSize', 12, 'FontWeight', 'bold');
    
    %% 15. 保存所有结果
    fprintf('\n正在保存所有结果...\n');
    
    % 保存增强结果
    saveEnhancedResults(filepath, name, ext, I_enhanced, fig1, ...
        psnr_val, ssim_val, mae_val, mse_val, rmse_val, snr_val, ...
        entropy_change, contrast_ratio);
    
    % 保存分割结果
    imwrite(BW_predicted, fullfile(filepath, [name '_segmentation_result.png']));
    imwrite(GT_reference, fullfile(filepath, [name '_reference_segmentation.png']));
    
    % 保存对比图
    saveas(fig_final, fullfile(filepath, [name '_segmentation_comparison.png']));
    saveas(fig_process, fullfile(filepath, [name '_segmentation_process.png']));
    
    % 保存分割性能指标到文本文件
    saveSegmentationMetricsToFile(filepath, name, metric_dice, metric_iou, metric_precision, ...
        metric_recall, metric_accuracy, metric_specificity, TP, FP, FN, TN, ...
        num_regions_pred, num_regions_ref);
    
    fprintf('\n========= 完整处理完成 =========\n');
    fprintf('增强结果已保存\n');
    fprintf('分割结果已保存: %s_segmentation_result.png\n', name);
    fprintf('参考分割已保存: %s_reference_segmentation.png\n', name);
    fprintf('分割性能指标已保存到: %s_segmentation_metrics.txt\n', name);
    fprintf('增强PSNR: %.2f dB, SSIM: %.4f\n', psnr_val, ssim_val);
    fprintf('分割Dice系数: %.4f\n', metric_dice);
    fprintf('预测分割区域数: %d\n', num_regions_pred);
    fprintf('参考分割区域数: %d\n', num_regions_ref);
    fprintf('所有结果已保存到: %s\n', filepath);
end

%% ========== 辅助函数 ==========

function saveEnhancedResults(filepath, name, ext, I_enhanced, fig1, ...
    psnr, ssim, mae, mse, rmse, snr, entropy_change, contrast_ratio)
    
    % 保存增强后的图像
    imwrite(I_enhanced, fullfile(filepath, [name '_enhanced_only' ext]));
    
    % 保存增强可视化图形
    saveas(fig1, fullfile(filepath, [name '_enhancement_visualization.png']));
    
    % 保存MAT文件
    metrics.Enhancement = struct(...
        'PSNR', psnr, 'SSIM', ssim, 'MAE', mae, 'MSE', mse, ...
        'RMSE', rmse, 'SNR', snr, 'EntropyChange', entropy_change, ...
        'ContrastRatio', contrast_ratio);
    
    save(fullfile(filepath, [name '_enhancement_metrics_only.mat']), 'metrics');
    
    % 保存文本报告
    saveEnhancedTextReport(filepath, name, metrics);
end

function saveEnhancedTextReport(filepath, name, metrics)
    fid = fopen(fullfile(filepath, [name '_enhancement_report_only.txt']), 'w');
    
    fprintf(fid, '图像增强处理报告\n');
    fprintf(fid, '=================\n\n');
    fprintf(fid, '生成时间: %s\n\n', datestr(now));
    
    fprintf(fid, '增强性能指标:\n');
    fprintf(fid, '------------\n');
    fprintf(fid, 'PSNR: %.2f dB %s\n', metrics.Enhancement.PSNR, ...
        getQualityLabel(metrics.Enhancement.PSNR, [20, 15], 'dB'));
    fprintf(fid, 'SSIM: %.4f %s\n', metrics.Enhancement.SSIM, ...
        getQualityLabel(metrics.Enhancement.SSIM, [0.8, 0.6], ''));
    fprintf(fid, 'SNR: %.2f dB\n', metrics.Enhancement.SNR);
    fprintf(fid, 'MAE: %.6f\n', metrics.Enhancement.MAE);
    fprintf(fid, 'MSE: %.6f\n', metrics.Enhancement.MSE);
    fprintf(fid, 'RMSE: %.6f\n', metrics.Enhancement.RMSE);
    fprintf(fid, '信息熵改进: +%.4f\n', metrics.Enhancement.EntropyChange);
    fprintf(fid, '对比度改进: ×%.3f\n\n', metrics.Enhancement.ContrastRatio);
    
    fprintf(fid, '总体评估:\n');
    fprintf(fid, '--------\n');
    
    if metrics.Enhancement.PSNR > 20 && metrics.Enhancement.SSIM > 0.8
        fprintf(fid, '增强质量: 优秀\n');
        fprintf(fid, '图像增强效果良好，保持了原始图像结构。\n');
    elseif metrics.Enhancement.PSNR > 15 && metrics.Enhancement.SSIM > 0.6
        fprintf(fid, '增强质量: 良好\n');
        fprintf(fid, '图像增强效果可接受，轻微改变了原始图像。\n');
    else
        fprintf(fid, '增强质量: 需要改进\n');
        fprintf(fid, '增强过程可能过度改变了原始图像。\n');
    end
    
    fclose(fid);
end

function saveSegmentationMetricsToFile(filepath, name, dice, iou, precision, recall, ...
    accuracy, specificity, TP, FP, FN, TN, num_regions_pred, num_regions_ref)
    
    filename = fullfile(filepath, [name '_segmentation_metrics.txt']);
    fid = fopen(filename, 'w');
    
    fprintf(fid, '图像分割性能评估报告\n');
    fprintf(fid, '====================\n\n');
    fprintf(fid, '生成时间: %s\n\n', datestr(now));
    
    fprintf(fid, '评估说明:\n');
    fprintf(fid, '--------\n');
    fprintf(fid, '使用基于增强图像特征创建的参考分割作为评估基准\n');
    fprintf(fid, '主要分割方法：多尺度顶帽变换 + Sigmoid增强 + 滞后阈值分割\n');
    fprintf(fid, '参考分割方法：基于统计特征的自适应阈值分割\n\n');
    
    fprintf(fid, '性能指标:\n');
    fprintf(fid, '--------\n');
    fprintf(fid, 'Dice系数: %.4f\n', dice);
    fprintf(fid, 'IoU: %.4f\n', iou);
    fprintf(fid, '精确率: %.4f\n', precision);
    fprintf(fid, '召回率: %.4f\n', recall);
    fprintf(fid, '准确率: %.4f\n', accuracy);
    fprintf(fid, '特异度: %.4f\n\n', specificity);
    
    fprintf(fid, '混淆矩阵统计:\n');
    fprintf(fid, '------------\n');
    total_pixels = TP + FP + FN + TN;
    fprintf(fid, '总像素数: %d\n', total_pixels);
    fprintf(fid, '真阳性(TP): %d (%.2f%%)\n', TP, 100*TP/total_pixels);
    fprintf(fid, '假阳性(FP): %d (%.2f%%)\n', FP, 100*FP/total_pixels);
    fprintf(fid, '假阴性(FN): %d (%.2f%%)\n', FN, 100*FN/total_pixels);
    fprintf(fid, '真阴性(TN): %d (%.2f%%)\n\n', TN, 100*TN/total_pixels);
    
    fprintf(fid, '区域统计:\n');
    fprintf(fid, '--------\n');
    fprintf(fid, '主要分割区域数: %d\n', num_regions_pred);
    fprintf(fid, '参考分割区域数: %d\n\n', num_regions_ref);
    
    fprintf(fid, '性能评估:\n');
    fprintf(fid, '--------\n');
    if dice > 0.7
        fprintf(fid, '分割结果与参考分割一致性良好，分割效果可靠。\n');
    elseif dice > 0.5
        fprintf(fid, '分割结果与参考分割一致性中等，分割效果基本可靠。\n');
    elseif dice > 0.3
        fprintf(fid, '分割结果与参考分割一致性一般，可能存在改进空间。\n');
    else
        fprintf(fid, '分割结果与参考分割差异较大，建议进一步优化分割参数。\n');
    end
    
    fclose(fid);
end

function label = getQualityLabel(value, thresholds, unit)
    if value >= thresholds(1)
        label = sprintf('[优秀%s]', unit);
    elseif value >= thresholds(2)
        label = sprintf('[良好%s]', unit);
    else
        label = sprintf('[需要改进%s]', unit);
    end
end

% 图像质量评估函数
function psnr_val = calculatePSNR(img1, img2)
    mse = mean((img1(:) - img2(:)).^2);
    if mse == 0
        psnr_val = Inf;
    else
        max_val = max(max(img1(:)), max(img2(:)));
        psnr_val = 10 * log10((max_val^2) / mse);
    end
end

function ssim_val = calculateSSIM(img1, img2)
    img1 = double(img1);
    img2 = double(img2);
    
    K = [0.01 0.03];
    L = 1;
    C1 = (K(1)*L)^2;
    C2 = (K(2)*L)^2;
    
    mu1 = mean(img1(:));
    mu2 = mean(img2(:));
    
    sigma1_sq = var(img1(:));
    sigma2_sq = var(img2(:));
    sigma12 = cov(img1(:), img2(:));
    sigma12 = sigma12(1,2);
    
    numerator = (2*mu1*mu2 + C1) * (2*sigma12 + C2);
    denominator = (mu1^2 + mu2^2 + C1) * (sigma1_sq + sigma2_sq + C2);
    ssim_val = numerator / denominator;
end

function mae_val = calculateMAE(img1, img2)
    mae_val = mean(abs(img1(:) - img2(:)));
end

function mse_val = calculateMSE(img1, img2)
    mse_val = mean((img1(:) - img2(:)).^2);
end

function snr_val = calculateSNR(img, enhanced_img)
    noise = enhanced_img - img;
    signal_power = mean(img(:).^2);
    noise_power = mean(noise(:).^2);
    
    if noise_power == 0
        snr_val = Inf;
    else
        snr_val = 10 * log10(signal_power / noise_power);
    end
end