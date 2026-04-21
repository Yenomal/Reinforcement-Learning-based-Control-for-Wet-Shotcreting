# 第一阶段TODO

- [x] 增加特殊工况

- [x] 项目构建，根据[项目代码结构.jpg](项目代码结构.png)构建

- [x] 增加真实环境

- [x] 当前的planner和rl区分不明显，需要重新设计rl的边界x

# 第二阶段TODO

- 由于增加了3D env，这里需要做一些系列工作，我们的整体目标是接入3D env作为关节角展示的环境

- [x] 修复原始的math env包括reward设计（当前的计算方法是delta distance这种会导致最终点附近抖动），3D点Observation（当前是在岩壁展开的2D空间做目标跟随）

- [x] 使用关节角动作空间（当前是直接末端点进入math_env，我需要设计关节角，关节参数需要参考[rock_3D](./src/rock_3D)里面的相关mesh）

- [x] 接入pybullet（我希望是这样一个空间，我输入一连串关节角，然后等我按下按键，他直接运行，也就是他是一个独立环境，不传递Observation出来）

- [x] 整理项目——删除不需要的+assets抬出+固定eval环境

# 第三阶段算法TODO

- [x] 迭代训练，critic-actor交替训练

- [] 偏好训练

# 第四阶段 TODO

- [] 并行环境