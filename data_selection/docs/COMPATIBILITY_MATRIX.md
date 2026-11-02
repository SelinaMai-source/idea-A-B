# 兼容矩阵

实现状态来自适配器和 `manifests/method_facts.yaml`。仓库 URL 本身不会把格子标成 READY。没有 Image-DP 训练任务被标成可跑。

| 方法 | 类型 | RoboMimic | LIBERO | Image-DP | 原文复现 | 统一适配 | 在线 | 实现状态 | 排行榜 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_data | control | SELECTOR_INTERFACE_ONLY | SELECTOR_INTERFACE_ONLY | BLOCKED_NO_CUDA | NOT_APPLICABLE | INTERFACE_ONLY_NO_IMAGE_DP_TRAINING | SYNTHETIC_INTERFACE_ONLY | IMPLEMENTED_CPU_SELECTOR | False |
| cupid | published_method | SOURCE_PRESENT_RUNTIME_BLOCKED | NOT_APPLICABLE | BLOCKED_NO_CUDA | BLOCKED_NO_CUDA | NOT_IMPLEMENTED | NOT_IMPLEMENTED | BLOCKED | False |
| datamil | published_method | NOT_APPLICABLE | SOURCE_PRESENT_OCTO_NOT_IMAGE_DP | BLOCKED_NO_CUDA | BLOCKED | NOT_IMPLEMENTED | NOT_IMPLEMENTED | BLOCKED | False |
| deminf | published_method | SOURCE_PRESENT_DIFFERENT_BACKBONE | NOT_APPLICABLE | BLOCKED_NO_CUDA | BLOCKED | NOT_IMPLEMENTED | NOT_IMPLEMENTED | BLOCKED | False |
| demo_score | published_method | SOURCE_PRESENT_RUNTIME_BLOCKED | NOT_APPLICABLE | BLOCKED_NO_CUDA | BLOCKED_NO_CUDA | NOT_IMPLEMENTED | NOT_IMPLEMENTED | BLOCKED | False |
| fifo | control | SELECTOR_INTERFACE_ONLY | SELECTOR_INTERFACE_ONLY | BLOCKED_NO_CUDA | NOT_APPLICABLE | INTERFACE_ONLY_NO_IMAGE_DP_TRAINING | SYNTHETIC_INTERFACE_ONLY | IMPLEMENTED_CPU_SELECTOR | False |
| fixed_score | control | SELECTOR_INTERFACE_ONLY | SELECTOR_INTERFACE_ONLY | BLOCKED_NO_CUDA | NOT_APPLICABLE | INTERFACE_ONLY_NO_IMAGE_DP_TRAINING | SYNTHETIC_INTERFACE_ONLY | IMPLEMENTED_CPU_SELECTOR | False |
| random | control | SELECTOR_INTERFACE_ONLY | SELECTOR_INTERFACE_ONLY | BLOCKED_NO_CUDA | NOT_APPLICABLE | INTERFACE_ONLY_NO_IMAGE_DP_TRAINING | SYNTHETIC_INTERFACE_ONLY | IMPLEMENTED_CPU_SELECTOR | False |
| reservoir | control | SELECTOR_INTERFACE_ONLY | SELECTOR_INTERFACE_ONLY | BLOCKED_NO_CUDA | NOT_APPLICABLE | INTERFACE_ONLY_NO_IMAGE_DP_TRAINING | SYNTHETIC_INTERFACE_ONLY | IMPLEMENTED_CPU_SELECTOR | False |
| strap | published_method | NOT_THE_PUBLISHED_LIBERO_PROTOCOL | SOURCE_PRESENT_RUNTIME_BLOCKED | BLOCKED_NO_CUDA | BLOCKED_NO_CUDA | NOT_IMPLEMENTED | NOT_IMPLEMENTED | BLOCKED | False |
| hinflow | online_learning_reference | NOT_APPLICABLE | SEPARATE_HINFLOW_TASKS_NOT_LIBERO10 | NOT_APPLICABLE | BLOCKED_NO_CUDA | NOT_APPLICABLE | NOT_APPLICABLE | BLOCKED | False |

## 不支持或未验证的组合

- all_data / libero_hinflow / ：HinFlow's four-task protocol is not a cell of this selector's Image-DP comparison.
- cupid / libero_hinflow / ：HinFlow's four-task protocol is not a cell of this selector's Image-DP comparison.
- datamil / libero_hinflow / ：HinFlow's four-task protocol is not a cell of this selector's Image-DP comparison.
- deminf / libero_hinflow / ：HinFlow's four-task protocol is not a cell of this selector's Image-DP comparison.
- demo_score / libero_hinflow / ：HinFlow's four-task protocol is not a cell of this selector's Image-DP comparison.
- fifo / libero_hinflow / ：HinFlow's four-task protocol is not a cell of this selector's Image-DP comparison.
- fixed_score / libero_hinflow / ：HinFlow's four-task protocol is not a cell of this selector's Image-DP comparison.
- random / libero_hinflow / ：HinFlow's four-task protocol is not a cell of this selector's Image-DP comparison.
- reservoir / libero_hinflow / ：HinFlow's four-task protocol is not a cell of this selector's Image-DP comparison.
- strap / libero_hinflow / ：HinFlow's four-task protocol is not a cell of this selector's Image-DP comparison.
- hinflow / unified_image_dp_leaderboard / ：HinFlow's backbone is not the image CNN Diffusion Policy, so it is excluded from that leaderboard.
- * / libero_10 / forgetting：Forgetting and BWT apply only to the separate task-incremental LIBERO protocol.
- * / libero_90 / forgetting：Forgetting and BWT apply only to the separate task-incremental LIBERO protocol.
- * / libero_hinflow / forgetting：Forgetting and BWT apply only to the separate task-incremental LIBERO protocol.
- * / robomimic_can_mh / forgetting：Forgetting and BWT apply only to the separate task-incremental LIBERO protocol.
- * / robomimic_lift_mh / forgetting：Forgetting and BWT apply only to the separate task-incremental LIBERO protocol.
- * / robomimic_square_mh / forgetting：Forgetting and BWT apply only to the separate task-incremental LIBERO protocol.
- * / robomimic_transport_mh / forgetting：Forgetting and BWT apply only to the separate task-incremental LIBERO protocol.

## 当前可跑

- all_data on synthetic_vector (cpu_selector_or_online_interface, cpu)
- fifo on synthetic_vector (cpu_selector_or_online_interface, cpu)
- fixed_score on synthetic_vector (cpu_selector_or_online_interface, cpu)
- random on synthetic_vector (cpu_selector_or_online_interface, cpu)
- reservoir on synthetic_vector (cpu_selector_or_online_interface, cpu)
