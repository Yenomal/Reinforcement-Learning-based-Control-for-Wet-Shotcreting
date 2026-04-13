# 隧道环境

这个目录存放从 [rock_environment.html](/home/rui/桌面/3D/rock_environment.html) 提取出来的 Plotly `surface` 曲面，并转换成 PyBullet 可加载的静态环境。

## 当前处理方式

- 数据源不是原始点云，而是规则网格曲面。
- 原始网格大小是 `100 x 200`。
- 当前环境由两部分组成：
  - `tunnel_wall.obj`：内侧岩壁
  - `tunnel_shell.obj`：外部矩形壳体和后部实心岩体
- 当前参数是：
  - 外部截面 `10m x 5m`
  - 开挖长度 `3m`
  - 总长度 `5m`
  - 后部实心长度 `2m`
- 当前只做 `visual mesh`，暂时不做 collision。

## 重新生成

在工作目录执行：

```bash
python tools/build_tunnel_environment.py
```

## 查看隧道

```bash
python tunnel_environment/view_tunnel_pybullet.py
```

如果只想做快速验证，不开 GUI：

```bash
python tunnel_environment/view_tunnel_pybullet.py --headless
```

如果想加一个地面参考：

```bash
python tunnel_environment/view_tunnel_pybullet.py --show-plane
```
