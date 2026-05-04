## 如何使用

- 岩壁环境生成与可视化：`uv run python scripts/visualize_rock_env.py`

- 训练：`uv run python scripts/train.py`

- 评估：`uv run python scripts/eval.py eval.checkpoint=outputs/runs/<run>/final.pt`

- 生成可达区域缓存：`uv run python scripts/build_reachability_map.py --force --device cuda`

- 所有脚本都支持 Hydra overrides，例如：`uv run python scripts/train.py algorithm=sac train.device=cpu`
