## 如何使用

- rock_env生成+可视化：uv run python -m src.rock_env.env

- planner生成起点+终点：uv run python -m src.component.planner

- train：uv run python -m src.train --config path/to/config.yaml（也可以不提供config直接修改./src/config.yaml），训练后看轨迹图xdg-open outputs/runs/xxx/training_curves.html

- eval：uv run python -m src.eval

- 生成可达区域缓存：uv run python -m src.component.reachability_map --force --device cuda

- 生成3D岩壁：uv run python -m src.rock_3D.tools.build_tunnel_environment.py

- 切换新的岩壁环境：生成新的html，pybullet同步、可达图同步