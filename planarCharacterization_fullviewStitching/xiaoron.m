ref = imread("D:\Image together\序列拼接\10.29\消融指标\8-all.jpg");
result = imread("D:\Image together\序列拼接\10.29\分割指标\SURF.png");

metrics = evaluate_stitched_quality(ref, result);
function metrics = evaluate_stitched_quality(ref_img, stitched_img)
% 评估拼接图像的质量（有参考图像）
% 包含 PSNR, SSIM, MSE, RMSE, UQI, FSIM, Edge SSIM, 可选 GMSD

    % --- 转为灰度 & double ---
    if size(ref_img,3) == 3
        ref_img = rgb2gray(ref_img);
    end
    if size(stitched_img,3) == 3
        stitched_img = rgb2gray(stitched_img);
    end
    ref_img = im2double(ref_img);
    stitched_img = im2double(stitched_img);

    % --- 尺寸统一 ---
    if ~isequal(size(ref_img), size(stitched_img))
        target_size = min(size(ref_img), size(stitched_img));
        ref_img = imresize(ref_img, target_size);
        stitched_img = imresize(stitched_img, target_size);
    end

    % === 指标计算 ===
    metrics = struct();

    % PSNR
    metrics.psnr = psnr(stitched_img, ref_img);

    % SSIM
    [metrics.ssim, metrics.ssim_map] = ssim(stitched_img, ref_img);

    % MSE / RMSE
    metrics.mse = immse(stitched_img, ref_img);
    metrics.rmse = sqrt(metrics.mse);

    % UQI
    metrics.uqi = uqi_index(ref_img, stitched_img);

    % FSIM (需 FeatureSIM.m)
    if exist('FeatureSIM', 'file') == 2
        try
            metrics.fsim = FeatureSIM(ref_img, stitched_img);
        catch
            warning('FSIM 计算失败，请检查 FeatureSIM 函数。');
            metrics.fsim = NaN;
        end
    else
        metrics.fsim = NaN;
    end

    % Edge SSIM (Sobel 边缘图结构相似性)
    % Edge SSIM (Sobel 边缘图结构相似性，需转为 double 类型)
sobel_ref = im2double(edge(ref_img, 'sobel'));
sobel_test = im2double(edge(stitched_img, 'sobel'));
metrics.edge_ssim = ssim(sobel_test, sobel_ref);


    % GMSD (需 GMSD.m)
    if exist('GMSD', 'file') == 2
        try
            [metrics.gmsd, ~] = GMSD(ref_img, stitched_img);
        catch
            warning('GMSD 计算失败，请检查 GMSD 函数。');
            metrics.gmsd = NaN;
        end
    else
        metrics.gmsd = NaN;
    end

    % === 输出结果 ===
    fprintf('【拼接质量评估指标】\n');
    fprintf('  PSNR       : %.2f dB\n', metrics.psnr);
    fprintf('  SSIM       : %.4f\n', metrics.ssim);
    fprintf('  MSE        : %.6f\n', metrics.mse);
    fprintf('  RMSE       : %.6f\n', metrics.rmse);
    fprintf('  UQI        : %.4f\n', metrics.uqi);
    if ~isnan(metrics.fsim), fprintf('  FSIM       : %.4f\n', metrics.fsim); end
    fprintf('  Edge SSIM  : %.4f\n', metrics.edge_ssim);
    if ~isnan(metrics.gmsd), fprintf('  GMSD       : %.4f\n', metrics.gmsd); end
end

%% ===== 子函数：UQI 实现 =====
function val = uqi_index(x, y)
    x = double(x); y = double(y);
    N = numel(x);
    mx = mean(x(:));
    my = mean(y(:));
    sigx2 = var(x(:));
    sigy2 = var(y(:));
    sigxy = sum((x(:)-mx).*(y(:)-my)) / (N - 1);
    val = (4 * mx * my * sigxy) / ((mx^2 + my^2)*(sigx2 + sigy2) + eps);
end
