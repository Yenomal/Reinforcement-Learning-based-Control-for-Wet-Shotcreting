## 如何使用

- 安装：`python -m pip install -e .`

- 训练：`python scripts/train.py`

- 评估：`python scripts/eval.py eval.checkpoint=outputs/runs/<run>/final.pt`

- 生成可达区域缓存：`python scripts/build_reachability_map.py --force --device cuda`

- 岩壁环境生成与可视化：`python scripts/visualize_rock_env.py`

- 所有脚本都支持 Hydra overrides，例如：`python scripts/train.py algorithm=sac train.device=cpu`
