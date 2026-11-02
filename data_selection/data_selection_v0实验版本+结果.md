# data_selection v0 实验版本+结果

只记录已经发生的事。论文格子保持 `NOT_RUN`，不适用的遗忘列写 `N/A`。不把 `NOT_RUN` 或 `N/A` 写成 0。没有 ours / v1。

合成接口烟测单独放在第 11 节，版本名带 `SYNTHETIC`。它的成功率来自合成随机过程，不是 RoboMimic 或 LIBERO。


## 1. RoboMimic Lift MH

图像 Diffusion Policy 未训练。下面每一格都是未跑。

<!-- TABLE_ID: robomimic_lift_mh -->

|版本|Method|Benchmark|Task|Protocol|Model|Data Budget|Final Success|Success AUC|Forgetting|Total Compute|状态|结果解释|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|v0_not_run|all_data|robomimic_lift_mh|Lift MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|random|robomimic_lift_mh|Lift MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|fifo|robomimic_lift_mh|Lift MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|reservoir|robomimic_lift_mh|Lift MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|fixed_score|robomimic_lift_mh|Lift MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。没有外部分数时返回 BLOCKED。图像训练未跑。|
|v0_not_run|cupid|robomimic_lift_mh|Lift MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|官方入口已定位但未执行，不退回 Random。|
|v0_not_run|deminf|robomimic_lift_mh|Lift MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|JAX/OpenX 栈，不是图像 Diffusion Policy。未执行。|
|v0_not_run|datamil|robomimic_lift_mh|Lift MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|原文 LIBERO 用 Octo。未执行，也不替换骨干。|
|v0_not_run|demo_score|robomimic_lift_mh|Lift MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|需要成功/失败 rollout 分类器。未训练。|
|v0_not_run|strap|robomimic_lift_mh|Lift MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|选择子轨迹。episode 预算不能代替。未执行。|


## 2. RoboMimic Can MH

图像 Diffusion Policy 未训练。下面每一格都是未跑。

<!-- TABLE_ID: robomimic_can_mh -->

|版本|Method|Benchmark|Task|Protocol|Model|Data Budget|Final Success|Success AUC|Forgetting|Total Compute|状态|结果解释|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|v0_not_run|all_data|robomimic_can_mh|Can MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|random|robomimic_can_mh|Can MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|fifo|robomimic_can_mh|Can MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|reservoir|robomimic_can_mh|Can MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|fixed_score|robomimic_can_mh|Can MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。没有外部分数时返回 BLOCKED。图像训练未跑。|
|v0_not_run|cupid|robomimic_can_mh|Can MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|官方入口已定位但未执行，不退回 Random。|
|v0_not_run|deminf|robomimic_can_mh|Can MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|JAX/OpenX 栈，不是图像 Diffusion Policy。未执行。|
|v0_not_run|datamil|robomimic_can_mh|Can MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|原文 LIBERO 用 Octo。未执行，也不替换骨干。|
|v0_not_run|demo_score|robomimic_can_mh|Can MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|需要成功/失败 rollout 分类器。未训练。|
|v0_not_run|strap|robomimic_can_mh|Can MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|选择子轨迹。episode 预算不能代替。未执行。|


## 3. RoboMimic Square MH

图像 Diffusion Policy 未训练。下面每一格都是未跑。

<!-- TABLE_ID: robomimic_square_mh -->

|版本|Method|Benchmark|Task|Protocol|Model|Data Budget|Final Success|Success AUC|Forgetting|Total Compute|状态|结果解释|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|v0_not_run|all_data|robomimic_square_mh|Square MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|random|robomimic_square_mh|Square MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|fifo|robomimic_square_mh|Square MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|reservoir|robomimic_square_mh|Square MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|fixed_score|robomimic_square_mh|Square MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。没有外部分数时返回 BLOCKED。图像训练未跑。|
|v0_not_run|cupid|robomimic_square_mh|Square MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|官方入口已定位但未执行，不退回 Random。|
|v0_not_run|deminf|robomimic_square_mh|Square MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|JAX/OpenX 栈，不是图像 Diffusion Policy。未执行。|
|v0_not_run|datamil|robomimic_square_mh|Square MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|原文 LIBERO 用 Octo。未执行，也不替换骨干。|
|v0_not_run|demo_score|robomimic_square_mh|Square MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|需要成功/失败 rollout 分类器。未训练。|
|v0_not_run|strap|robomimic_square_mh|Square MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|选择子轨迹。episode 预算不能代替。未执行。|


## 4. RoboMimic Transport MH

图像 Diffusion Policy 未训练。下面每一格都是未跑。

<!-- TABLE_ID: robomimic_transport_mh -->

|版本|Method|Benchmark|Task|Protocol|Model|Data Budget|Final Success|Success AUC|Forgetting|Total Compute|状态|结果解释|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|v0_not_run|all_data|robomimic_transport_mh|Transport MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|random|robomimic_transport_mh|Transport MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|fifo|robomimic_transport_mh|Transport MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|reservoir|robomimic_transport_mh|Transport MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|fixed_score|robomimic_transport_mh|Transport MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。没有外部分数时返回 BLOCKED。图像训练未跑。|
|v0_not_run|cupid|robomimic_transport_mh|Transport MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|官方入口已定位但未执行，不退回 Random。|
|v0_not_run|deminf|robomimic_transport_mh|Transport MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|JAX/OpenX 栈，不是图像 Diffusion Policy。未执行。|
|v0_not_run|datamil|robomimic_transport_mh|Transport MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|原文 LIBERO 用 Octo。未执行，也不替换骨干。|
|v0_not_run|demo_score|robomimic_transport_mh|Transport MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|需要成功/失败 rollout 分类器。未训练。|
|v0_not_run|strap|robomimic_transport_mh|Transport MH|robomimic_image_mh|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|选择子轨迹。episode 预算不能代替。未执行。|


## 5. LIBERO-10 Data Selection

图像 Diffusion Policy 未训练。下面每一格都是未跑。

<!-- TABLE_ID: libero_10 -->

|版本|Method|Benchmark|Task|Protocol|Model|Data Budget|Final Success|Success AUC|Forgetting|Total Compute|状态|结果解释|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|v0_not_run|all_data|libero_10|LIBERO-10|libero_10_published|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|random|libero_10|LIBERO-10|libero_10_published|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|fifo|libero_10|LIBERO-10|libero_10_published|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|reservoir|libero_10|LIBERO-10|libero_10_published|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|fixed_score|libero_10|LIBERO-10|libero_10_published|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。没有外部分数时返回 BLOCKED。图像训练未跑。|
|v0_not_run|cupid|libero_10|LIBERO-10|libero_10_published|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|官方入口已定位但未执行，不退回 Random。|
|v0_not_run|deminf|libero_10|LIBERO-10|libero_10_published|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|JAX/OpenX 栈，不是图像 Diffusion Policy。未执行。|
|v0_not_run|datamil|libero_10|LIBERO-10|libero_10_published|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|原文 LIBERO 用 Octo。未执行，也不替换骨干。|
|v0_not_run|demo_score|libero_10|LIBERO-10|libero_10_published|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|需要成功/失败 rollout 分类器。未训练。|
|v0_not_run|strap|libero_10|LIBERO-10|libero_10_published|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|选择子轨迹。episode 预算不能代替。未执行。|


## 6. LIBERO Closed-loop Online Learning

图像 Diffusion Policy 未训练。下面每一格都是未跑。

<!-- TABLE_ID: libero_online -->

|版本|Method|Benchmark|Task|Protocol|Model|Data Budget|Final Success|Success AUC|Forgetting|Total Compute|状态|结果解释|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|v0_not_run|all_data|libero_10|LIBERO-10 online|online_closed_loop|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|random|libero_10|LIBERO-10 online|online_closed_loop|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|fifo|libero_10|LIBERO-10 online|online_closed_loop|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|reservoir|libero_10|LIBERO-10 online|online_closed_loop|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|fixed_score|libero_10|LIBERO-10 online|online_closed_loop|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|控制组。没有外部分数时返回 BLOCKED。图像训练未跑。|
|v0_not_run|cupid|libero_10|LIBERO-10 online|online_closed_loop|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|官方入口已定位但未执行，不退回 Random。|
|v0_not_run|deminf|libero_10|LIBERO-10 online|online_closed_loop|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|JAX/OpenX 栈，不是图像 Diffusion Policy。未执行。|
|v0_not_run|datamil|libero_10|LIBERO-10 online|online_closed_loop|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|原文 LIBERO 用 Octo。未执行，也不替换骨干。|
|v0_not_run|demo_score|libero_10|LIBERO-10 online|online_closed_loop|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|需要成功/失败 rollout 分类器。未训练。|
|v0_not_run|strap|libero_10|LIBERO-10 online|online_closed_loop|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|选择子轨迹。episode 预算不能代替。未执行。|


## 7. LIBERO Continual / Forgetting

图像 Diffusion Policy 未训练。下面每一格都是未跑。

<!-- TABLE_ID: libero_continual -->

|版本|Method|Benchmark|Task|Protocol|Model|Data Budget|Final Success|Success AUC|Forgetting|Total Compute|状态|结果解释|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|v0_not_run|all_data|libero_continual|task incremental|continual_learning|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|random|libero_continual|task incremental|continual_learning|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|fifo|libero_continual|task incremental|continual_learning|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|reservoir|libero_continual|task incremental|continual_learning|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|控制组。选择器已在 CPU 测试，图像训练未跑。|
|v0_not_run|fixed_score|libero_continual|task incremental|continual_learning|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|控制组。没有外部分数时返回 BLOCKED。图像训练未跑。|
|v0_not_run|cupid|libero_continual|task incremental|continual_learning|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|官方入口已定位但未执行，不退回 Random。|
|v0_not_run|deminf|libero_continual|task incremental|continual_learning|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|JAX/OpenX 栈，不是图像 Diffusion Policy。未执行。|
|v0_not_run|datamil|libero_continual|task incremental|continual_learning|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|原文 LIBERO 用 Octo。未执行，也不替换骨干。|
|v0_not_run|demo_score|libero_continual|task incremental|continual_learning|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|需要成功/失败 rollout 分类器。未训练。|
|v0_not_run|strap|libero_continual|task incremental|continual_learning|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|选择子轨迹。episode 预算不能代替。未执行。|


## 8. HinFlow Original Reference

HinFlow 不是数据选择 baseline，也不进入 Image-DP 排行榜。hindsight 不进入其他方法的信息预算。官方训练未跑。

<!-- TABLE_ID: hinflow_reference -->

|版本|Method|Benchmark|Task|Protocol|Model|Data Budget|Final Success|Success AUC|Forgetting|Total Compute|状态|结果解释|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|v0_not_run|hinflow|libero_hinflow|libero_butter / libero_book / libero_chocolate / libero_microwave|hinflow_original|author_original_not_image_dp|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|任务名来自 HinFlow README。训练未跑。ManiSkill 三个任务不在本阶段。|

## 9. Published Paper vs Reproduction

论文表里的数字没有抽取。本地也没有对应运行。

<!-- TABLE_ID: paper_vs_reproduction -->

|版本|Method|Benchmark|Task|Protocol|Model|Data Budget|Final Success|Success AUC|Forgetting|Total Compute|状态|结果解释|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|v0_not_run|cupid|paper_table|NOT_EXTRACTED|original|author_original|NOT_RUN|NOT_EXTRACTED|NOT_EXTRACTED|N/A|NOT_RUN|NOT_RUN|本地 NOT_RUN。未从 PDF 抄数字。|
|v0_not_run|deminf|paper_table|NOT_EXTRACTED|original|author_original|NOT_RUN|NOT_EXTRACTED|NOT_EXTRACTED|N/A|NOT_RUN|NOT_RUN|本地 NOT_RUN。未从 PDF 抄数字。|
|v0_not_run|datamil|paper_table|NOT_EXTRACTED|original|octo_not_image_dp|NOT_RUN|NOT_EXTRACTED|NOT_EXTRACTED|N/A|NOT_RUN|NOT_RUN|本地 NOT_RUN。原文骨干不是本表的 Diffusion Policy。|
|v0_not_run|demo_score|paper_table|NOT_EXTRACTED|original|author_original|NOT_RUN|NOT_EXTRACTED|NOT_EXTRACTED|N/A|NOT_RUN|NOT_RUN|本地 NOT_RUN。README 里的相对提升不是本仓库测到的数。|
|v0_not_run|strap|paper_table|NOT_EXTRACTED|original|robomimic_strap|NOT_RUN|NOT_EXTRACTED|NOT_EXTRACTED|N/A|NOT_RUN|NOT_RUN|本地 NOT_RUN。|
|v0_not_run|hinflow|paper_table|NOT_EXTRACTED|original|author_original|NOT_RUN|NOT_EXTRACTED|NOT_EXTRACTED|N/A|NOT_RUN|NOT_RUN|本地 NOT_RUN。与 Image-DP 排行榜分开。|

## 10. Cross-method Findings

没有可比较的已跑论文格子，因此没有排序，也没有胜负结论。控制组 CPU 测试只说明预算和接口，不说明策略成功率。

<!-- TABLE_ID: cross_method -->

|版本|Method|Benchmark|Task|Protocol|Model|Data Budget|Final Success|Success AUC|Forgetting|Total Compute|状态|结果解释|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|v0_not_run|all_methods|all_benchmarks|all_tasks|all_protocols|image_cnn_diffusion_policy|NOT_RUN|NOT_RUN|NOT_RUN|N/A|NOT_RUN|NOT_RUN|没有跨方法结论。|

## 11. 合成接口烟测（不是论文数字）

命令：`python scripts/run_online_baselines.py --method random --benchmark robomimic_lift_mh --seed 0 --synthetic --output-dir results/synthetic/interface_smoke`

`requested_benchmark` 只是 CLI 参数。结果状态是 `SYNTHETIC_INTERFACE_SMOKE`，策略模型没有更新，`training_optimizer_steps` 为 0。seed 0 的两轮合成评估曲线是 `(8, 0.0)` 和 `(16, 0.0)`，所以成功率和 AUC 都算成 0.0。这是该合成过程的结果，不能抄进第 1–10 节。

原始文件：`results/synthetic/interface_smoke/result.json`。

<!-- TABLE_ID: synthetic_interface -->

|版本|Method|Benchmark|Task|Protocol|Model|Data Budget|Final Success|Success AUC|Forgetting|Total Compute|状态|结果解释|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|SYNTHETIC_INTERFACE_SMOKE|random|synthetic_vector|synthetic_task|online_closed_loop_synthetic|NONE_SYNTHETIC_NO_DP_UPDATE|selected_episodes=1; selected_transitions=4|0.0|0.0|N/A|cpu_hours=4.3414377917846045e-08; gpu_hours=N/A|SYNTHETIC_INTERFACE_SMOKE|seed 0。环境步 16。专家查询 1。失败 rollout 没有进入模仿池。hindsight 没有注入。不是论文数字。|
