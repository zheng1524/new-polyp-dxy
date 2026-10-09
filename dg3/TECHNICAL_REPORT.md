# Quality-aware Sharp5 technical report

## Blind protocol

候选池严格为既有 raw Sharp5 每个冻结时间 bin 的现成候选；按既有 `sharpness` 先取 3，而不是重新抽帧。ROI Tenengrad 是 Sobel 梯度能量均值，Laplacian 是 ROI 方差代理；三者均只在 polyp interior、polyp boundary neighbourhood 或 ring boundary neighbourhood计算，故背景纹理与 ROI 面积不会主导。每 bin 保留 ROI Tenengrad ≥ 75% 最大值且 Laplacian ≥ 50% 最大值者；再要求 frozen V2 valid、ring最大连通成分 ≥80%、frontality≥0.35、ring-adjacency可信原始边缘≥35%且至少一弧。通过者按可信比例、可信弧长、ROI Tenengrad、原 sharpness 作字典序排序。

所有阶段 GT 隔离；SHA 在 `inputs/blind_sha256.txt`。G0/G3 fit、scale、`Dxy_i`、>=4 帧 capture median、formal fusion 和 Ip8 fallback 均复用且未修改。**本轮是 selector-alone ablation：没有再次运行 S6 的 post-capture CPAG retry**，因此 B/D 不是可直接晋升的 S6 替代物；尤其 B 的尾部用以验证质量 gate 不可取代既有安全层。

## Results

method,coverage,median_abs_error_pct,MAE_mm,mean_abs_error_pct,within5_pct,within10_pct,p90_pct,p95_pct,max_pct,mean_signed_error_pct,mean_error_mm
A_original_sharp5_G0,39,4.995421085025246,0.6110956088909288,6.419134827553017,51.28205128205128,79.48717948717949,13.44186680314535,15.205804797391824,23.685498698929642,-0.0008179620259960953,-0.13750153283600877
B_quality_aware_G0,39,4.980738446606654,0.8200797945100874,7.813814236591964,51.28205128205128,79.48717948717949,14.976901582945223,21.589319786867147,69.90151830178627,-2.383667811980607,-0.40718349282286587
C_original_sharp5_G3,39,5.003613248666661,0.6224640390005161,6.913802348365914,48.717948717948715,76.92307692307693,13.447350158929819,18.954768494847585,29.822176018674373,0.44999317484415446,-0.17497640882873378
D_quality_aware_G3,39,4.980738446606654,0.6310721590553092,6.848065825239748,51.28205128205128,74.35897435897436,14.829865553922517,20.219854095804195,23.685498698929642,0.6222096746190007,-0.14376615531526868


选择变化 291/395，missing 117；quality capture eligible：{'B_quality_aware_G0': 54, 'D_quality_aware_G3': 40}。详表：`tables/quality_candidate_features_blind.csv`、`quality_selection_audit_blind.csv`、frame/capture/group CSV。
