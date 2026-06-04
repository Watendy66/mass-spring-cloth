# -*- coding: utf-8 -*-
"""
质点-弹簧布料模拟 (Mass-Spring Cloth Simulation)
==================================================
基于 Taichi 的布料动态模拟，三种数值积分求解器对比 + 选做扩展：
    积分器：显式欧拉 / 半隐式欧拉 / 隐式欧拉(定点迭代)
    弹簧  ：结构(Structural) + 剪切(Shear) + 弯曲(Bending)   [选做1]
    碰撞  ：场景中的球体与布料质点碰撞处理                     [选做2]
    交互  ：GGUI 实时调节 kd / ks / ks_shear / ks_bend / 速度上限 / 碰撞球

架构要点：
    - 初始化拆分为多个 @ti.kernel 并按序调用，保证 GPU 状态同步。
    - compute_forces_on / clamp_velocity / resolve_collision 用 @ti.func，编译时内联。
    - 三个积分器各为单个 @ti.kernel，算力+更新合并，最小化 Kernel 启动次数。
    - 弹簧力用 ti.atomic_add 累加到两端，避免多线程写冲突。
"""

import taichi as ti

# arch 优先用 GPU；调试时可用环境变量切到 CPU：  CLOTH_ARCH=cpu python cloth_simulation.py
import os
_ARCH = {"cpu": ti.cpu, "gpu": ti.gpu, "cuda": ti.cuda,
         "vulkan": ti.vulkan, "metal": ti.metal}.get(
    os.environ.get("CLOTH_ARCH", "gpu").lower(), ti.gpu)
ti.init(arch=_ARCH)

# ======================================================================
# 一、全局参数
# ======================================================================
N = 20                       # 布料网格分辨率 (N x N 个质点)
NV = N * N                   # 质点总数

# --- 三类弹簧的数量（构建时按类别拼接到同一个 spring 场里）---
NE_STRUCT = 2 * N * (N - 1)          # 结构：水平 N*(N-1) + 竖直 (N-1)*N
NE_SHEAR = 2 * (N - 1) * (N - 1)     # 剪切：每个 quad 两条对角线
NE_BEND = 2 * N * (N - 2)            # 弯曲：水平/竖直方向隔一个质点相连
NE = NE_STRUCT + NE_SHEAR + NE_BEND  # 弹簧总数

NF = 2 * (N - 1) * (N - 1)   # 渲染三角面片数

CELL = 1.0 / N               # 初始网格间距（结构弹簧原长）
DT = 5e-4                    # 时间步长；显式欧拉对该值非常敏感
SUBSTEPS = 30                # 每帧子步数
MASS = 1.0                   # 单个质点质量
IMPLICIT_ITERS = 15          # 隐式欧拉定点迭代次数

GRAVITY = ti.Vector([0.0, -9.8, 0.0])

# 运行时可调参数，用 0 维 field 承载（Python 标量会在编译时被烧死，无法实时改）
ks = ti.field(dtype=ti.f32, shape=())        # 结构弹簧劲度
ks_shear = ti.field(dtype=ti.f32, shape=())  # 剪切弹簧劲度（=0 即关闭）
ks_bend = ti.field(dtype=ti.f32, shape=())   # 弯曲弹簧劲度（=0 即关闭）
kd = ti.field(dtype=ti.f32, shape=())        # 阻尼系数
max_vel = ti.field(dtype=ti.f32, shape=())   # 速度钳制上限

# 碰撞球
ball_center = ti.Vector.field(3, dtype=ti.f32, shape=1)
ball_radius = ti.field(dtype=ti.f32, shape=())
ball_on = ti.field(dtype=ti.i32, shape=())   # 1=开启碰撞

# ======================================================================
# 二、状态场 (Fields)
# ======================================================================
x = ti.Vector.field(3, dtype=ti.f32, shape=NV)   # 位置
v = ti.Vector.field(3, dtype=ti.f32, shape=NV)   # 速度
f = ti.Vector.field(3, dtype=ti.f32, shape=NV)   # 受力累加器
fixed = ti.field(dtype=ti.i32, shape=NV)         # 是否固定 (1=钉住)

# 隐式欧拉定点迭代用：保存一步开始时的 (x_t, v_t)
x_prev = ti.Vector.field(3, dtype=ti.f32, shape=NV)
v_prev = ti.Vector.field(3, dtype=ti.f32, shape=NV)

# 弹簧拓扑
spring = ti.Vector.field(2, dtype=ti.i32, shape=NE)   # 两端质点全局索引
rest_length = ti.field(dtype=ti.f32, shape=NE)        # 原长
spring_type = ti.field(dtype=ti.i32, shape=NE)        # 0=结构 1=剪切 2=弯曲

# 渲染三角网格索引
indices = ti.field(dtype=ti.i32, shape=NF * 3)


# ======================================================================
# 三、初始化 Kernel（任务1：拆分 + 顺序调用保证同步）
# ======================================================================
@ti.kernel
def init_positions():
    """质点位置/速度/受力初始化，并钉住两个角。"""
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * CELL, 0.6, j * CELL])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        fixed[idx] = 0
    # 钉住整条 i=0 上边，布料像窗帘一样铺开下垂
    # （想改回"只钉两个角"的旗帜效果，把下面循环换成两行 fixed[...] = 1 即可）
    for j in range(N):
        fixed[0 * N + j] = 1


@ti.kernel
def init_springs():
    """三类弹簧拓扑初始化。依赖 init_positions 写好的 x（算原长），
    因此必须在其后调用 —— Kernel 边界即隐式同步屏障。"""
    # ---- 结构弹簧 type=0 ----
    # 水平 (i,j)-(i,j+1)
    for i, j in ti.ndrange(N, N - 1):
        eid = i * (N - 1) + j
        a = i * N + j
        b = i * N + (j + 1)
        spring[eid] = ti.Vector([a, b]); spring_type[eid] = 0
        rest_length[eid] = (x[a] - x[b]).norm()
    # 竖直 (i,j)-(i+1,j)
    off = N * (N - 1)
    for i, j in ti.ndrange(N - 1, N):
        eid = off + i * N + j
        a = i * N + j
        b = (i + 1) * N + j
        spring[eid] = ti.Vector([a, b]); spring_type[eid] = 0
        rest_length[eid] = (x[a] - x[b]).norm()

    # ---- 剪切弹簧 type=1（每个 quad 两条对角线）----
    base_s = NE_STRUCT
    for i, j in ti.ndrange(N - 1, N - 1):
        q = i * (N - 1) + j
        # 对角线1: (i,j)-(i+1,j+1)
        e1 = base_s + q * 2 + 0
        a1 = i * N + j
        b1 = (i + 1) * N + (j + 1)
        spring[e1] = ti.Vector([a1, b1]); spring_type[e1] = 1
        rest_length[e1] = (x[a1] - x[b1]).norm()
        # 对角线2: (i,j+1)-(i+1,j)
        e2 = base_s + q * 2 + 1
        a2 = i * N + (j + 1)
        b2 = (i + 1) * N + j
        spring[e2] = ti.Vector([a2, b2]); spring_type[e2] = 1
        rest_length[e2] = (x[a2] - x[b2]).norm()

    # ---- 弯曲弹簧 type=2（隔一个质点相连）----
    base_b = NE_STRUCT + NE_SHEAR
    # 水平 (i,j)-(i,j+2)
    for i, j in ti.ndrange(N, N - 2):
        eid = base_b + i * (N - 2) + j
        a = i * N + j
        b = i * N + (j + 2)
        spring[eid] = ti.Vector([a, b]); spring_type[eid] = 2
        rest_length[eid] = (x[a] - x[b]).norm()
    # 竖直 (i,j)-(i+2,j)
    base_b2 = base_b + N * (N - 2)
    for i, j in ti.ndrange(N - 2, N):
        eid = base_b2 + i * N + j
        a = i * N + j
        b = (i + 2) * N + j
        spring[eid] = ti.Vector([a, b]); spring_type[eid] = 2
        rest_length[eid] = (x[a] - x[b]).norm()


@ti.kernel
def init_mesh_indices():
    """渲染索引初始化：每个 quad 拆两个三角形。"""
    for i, j in ti.ndrange(N - 1, N - 1):
        quad = i * (N - 1) + j
        v00 = i * N + j
        v10 = (i + 1) * N + j
        v01 = i * N + (j + 1)
        v11 = (i + 1) * N + (j + 1)
        indices[quad * 6 + 0] = v00
        indices[quad * 6 + 1] = v10
        indices[quad * 6 + 2] = v11
        indices[quad * 6 + 3] = v00
        indices[quad * 6 + 4] = v11
        indices[quad * 6 + 5] = v01


# ======================================================================
# 四、力学计算 / 防爆 / 碰撞（任务2：@ti.func，编译时内联）
# ======================================================================
@ti.func
def compute_forces_on():
    """合力 = 重力 + 阻尼 + 三类弹簧力。按 spring_type 选用对应劲度，
    某类劲度设为 0 即等效关闭该类弹簧。"""
    for i in range(NV):
        f[i] = MASS * GRAVITY - kd[None] * v[i]
    for e in range(NE):
        # 按类型选劲度
        k = ks[None]
        if spring_type[e] == 1:
            k = ks_shear[None]
        elif spring_type[e] == 2:
            k = ks_bend[None]
        a = spring[e][0]
        b = spring[e][1]
        d = x[a] - x[b]
        L = d.norm(1e-12)
        force = -k * (L - rest_length[e]) * d / L
        ti.atomic_add(f[a], force)
        ti.atomic_add(f[b], -force)


@ti.func
def clamp_velocity():
    """速度钳制，防数值爆炸。"""
    for i in range(NV):
        sp = v[i].norm()
        if sp > max_vel[None]:
            v[i] = v[i] * (max_vel[None] / sp)


@ti.func
def resolve_collision():
    """布料质点 vs 球体的简单碰撞：进入球内则投影回球面，并消去法向(指向球心)速度。"""
    if ball_on[None] == 1:
        c = ball_center[0]
        r = ball_radius[None]
        for i in range(NV):
            if fixed[i] == 0:
                rel = x[i] - c
                dist = rel.norm(1e-9)
                if dist < r:
                    n = rel / dist
                    x[i] = c + n * r * 1.001    # 推到球面外侧一点，避免穿插抖动
                    vn = v[i].dot(n)
                    if vn < 0.0:                 # 只消去朝向球心的分量（无摩擦）
                        v[i] = v[i] - vn * n


# ======================================================================
# 五、三种积分求解器（任务3：各一个 @ti.kernel，算力+更新合并）
# ======================================================================
@ti.kernel
def step_explicit():
    """显式欧拉：x_{t+1}=x_t+v_t dt ;  v_{t+1}=v_t+a_t dt"""
    compute_forces_on()
    for i in range(NV):
        if fixed[i] == 0:
            a = f[i] / MASS
            x[i] = x[i] + v[i] * DT      # 先用旧速度更新位置
            v[i] = v[i] + a * DT
    clamp_velocity()
    resolve_collision()


@ti.kernel
def step_semi_implicit():
    """半隐式欧拉：v_{t+1}=v_t+a_t dt ;  x_{t+1}=x_t+v_{t+1} dt"""
    compute_forces_on()
    for i in range(NV):
        if fixed[i] == 0:
            a = f[i] / MASS
            v[i] = v[i] + a * DT         # 先更新速度
            x[i] = x[i] + v[i] * DT      # 再用新速度更新位置
    clamp_velocity()
    resolve_collision()


@ti.kernel
def step_implicit_iter():
    """隐式欧拉(定点迭代)：用 ti.static 编译期展开，单次 Kernel 启动完成全部迭代。"""
    for i in range(NV):
        x_prev[i] = x[i]
        v_prev[i] = v[i]
    for _ in ti.static(range(IMPLICIT_ITERS)):
        compute_forces_on()
        for i in range(NV):
            if fixed[i] == 0:
                a = f[i] / MASS
                v[i] = v_prev[i] + a * DT
                x[i] = x_prev[i] + v[i] * DT
    clamp_velocity()
    resolve_collision()


# ======================================================================
# 六、Python 侧辅助
# ======================================================================
def reset():
    """三个 init Kernel 必须按此顺序调用（springs 依赖 positions）。"""
    init_positions()
    init_springs()
    init_mesh_indices()


METHOD_NAMES = ["Explicit Euler", "Semi-Implicit Euler", "Implicit Euler (iter)"]


def main():
    # 默认参数
    ks[None] = 1500.0
    ks_shear[None] = 600.0
    ks_bend[None] = 200.0
    kd[None] = 2.0
    max_vel[None] = 50.0
    # 碰撞球（放在窗帘下垂会压到的位置；默认关闭，按钮开启即接触）
    ball_center[0] = ti.Vector([0.3, 0.0, 0.5])
    ball_radius[None] = 0.2
    ball_on[None] = 0
    reset()

    window = ti.ui.Window("质点-弹簧布料模拟 Mass-Spring Cloth", (1024, 768), vsync=True)
    canvas = window.get_canvas()
    try:
        scene = window.get_scene()          # 新版推荐
    except AttributeError:
        scene = ti.ui.Scene()               # 旧版回退
    camera = ti.ui.Camera()
    camera.position(2.0, 0.8, 2.0)
    camera.lookat(0.3, 0.0, 0.5)
    camera.up(0, 1, 0)

    method = 1          # 默认半隐式
    paused = False

    while window.running:
        # ---------- GUI 控制面板（imgui 默认字体无中文，标签用英文）----------
        gui = window.GUI
        gui.begin("Control Panel", 0.02, 0.02, 0.32, 0.94)
        gui.text("Integrator: " + METHOD_NAMES[method])
        if gui.button("1. Explicit Euler"):
            method = 0; print("切换 -> 显式欧拉")
        if gui.button("2. Semi-Implicit Euler"):
            method = 1; print("切换 -> 半隐式欧拉")
        if gui.button("3. Implicit Euler (iter)"):
            method = 2; print("切换 -> 隐式欧拉(定点迭代)")
        if gui.button("Pause / Resume"):
            paused = not paused; print("暂停" if paused else "继续")
        if gui.button("Reset"):
            reset(); paused = False; print("重置布料")

        gui.text("--- Spring stiffness ---")
        ks[None] = gui.slider_float("Structural ks", ks[None], 100.0, 5000.0)
        ks_shear[None] = gui.slider_float("Shear ks", ks_shear[None], 0.0, 3000.0)
        ks_bend[None] = gui.slider_float("Bending ks", ks_bend[None], 0.0, 2000.0)

        gui.text("--- Dynamics ---")
        kd[None] = gui.slider_float("Damping kd", kd[None], 0.0, 20.0)
        max_vel[None] = gui.slider_float("Max Velocity", max_vel[None], 1.0, 200.0)

        gui.text("--- Collision ball ---")
        if gui.button("Toggle Collision"):
            ball_on[None] = 1 - ball_on[None]
            print("碰撞球", "开" if ball_on[None] == 1 else "关")
        ball_radius[None] = gui.slider_float("Ball Radius", ball_radius[None], 0.05, 0.4)
        gui.text("Collision: " + ("ON" if ball_on[None] == 1 else "OFF"))
        gui.text("Status: " + ("PAUSED" if paused else "RUNNING"))
        gui.end()

        # ---------- 物理推进 ----------
        if not paused:
            for _ in range(SUBSTEPS):
                if method == 0:
                    step_explicit()
                elif method == 1:
                    step_semi_implicit()
                else:
                    step_implicit_iter()

        # ---------- 渲染 ----------
        camera.track_user_inputs(window, movement_speed=0.02, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.4, 0.4, 0.4))
        scene.point_light(pos=(1.5, 2.0, 2.0), color=(1.0, 1.0, 1.0))
        scene.mesh(x, indices=indices, color=(0.25, 0.55, 0.9), two_sided=True)
        scene.particles(x, radius=0.006, color=(0.95, 0.5, 0.2))
        if ball_on[None] == 1:
            scene.particles(ball_center, radius=ball_radius[None], color=(0.8, 0.8, 0.3))
        canvas.set_background_color((0.1, 0.1, 0.12))
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
