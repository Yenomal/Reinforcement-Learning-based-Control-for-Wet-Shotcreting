# shipen 4DoF 可视化

这个目录放的是当前这版 `4DoF` PyBullet 可视化资产。

## 当前约定

- 底盘固定，不做动力学。
- 只保留 `4` 个关节：
  - `turret_yaw`
  - `shoulder_pitch`
  - `elbow_pitch`
  - `wrist_pitch`
- `0` 姿态采用当前建模姿态。
- `pitch` 正方向按右手定则设置为侧视图里“往上抬”。
- `yaw` 正方向按右手定则设置为俯视图里逆时针。

## 生成资产

在当前工作目录执行：

```bash
blender -b shipen.blend --python tools/build_visualization_assets.py
```

## 启动 PyBullet

先确保环境里有 `pybullet`：

```bash
python -m pip install pybullet
```

然后运行：

```bash
python robot_4dof/view_pybullet.py
```

也可以带初始角度：

```bash
python robot_4dof/view_pybullet.py --angles 0 10 -20 15
```

启动后界面里会出现四个滑块，可以直接拖动查看四个关节的效果。
