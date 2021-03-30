#! /bin/bash

cd ../build/bin/

# 四面体、弹簧质点、voxel、点云、混合、六面体
# tet_fem
# mass_spring
# voxel
# particle
# hybrid
# hex_fem


# 隐式欧拉时间积分、半隐式欧拉时间积分、显式欧拉时间积分、快速质点弹簧法(投影动力学)、几何多重网格技术、基于cage的插值加速
# implicit_euler
# explicit_euler
# semi_euler
# fast_ms
# multi
# cage

# argv 1: 模型表示格式
# argv 2: 模拟计算方法

./Release/App_Test $1 $2
