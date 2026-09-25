# -*- coding: utf-8 -*-
"""
common.py —— 四个问题共用的“公共物理规则”模块
=================================================
这里集中放置：
  1. 读取附件数据（节点、货箱、无人机、中继、通信参数）
  2. 读取 30 米 DEM，并提供“查某点地面高程”的函数
  3. 航段计算：水平距离、巡航海拔、爬升/下降高度、飞行时间、能耗
  4. 电池两阶段充电时间
  5. 通信链路：视线遮挡判断、自由空间损耗、链路是否可用
所有问题都 import 这个文件，保证“单位、编号、计算口径”完全一致。

【题目未明确给出、由我们补充的假设】（论文中需写明）
  (a) 水平巡航能耗 E_hor = E_use * d / L_g(q)
      含义：电池可用能量 E_use 恰好能让无人机以载荷 q 飞完等效航程 L_g(q)，
      所以飞 d 米要消耗 d / L_g(q) 比例的电量。
  (b) 爬升附加能耗 E_up = (m_空 + q) * g * h_up / (η_up * 3.6e6)
      含义：把总质量抬高 h_up 米需要的势能 m*g*h（焦耳），除以爬升能耗效率 η，
      再除以 3.6e6 把焦耳换算成 kWh。
  (c) 爬升/下降都在航段端点“垂直”完成，巡航段为水平直线（与时间公式一致）。
  (d) 视线遮挡判定时，忽略两个端点周围 30 米（约 1 个像元）内的地形，
      因为 DEM 是数字表面模型（含树木、房屋），端点像元本身不应遮挡自己。
"""

import os
import math
import numpy as np
import pandas as pd
import scipy.io as sio

# ---------------------------------------------------------------
# 0. 路径设置：代码放在 “D题/代码/” 下，数据在 “D题/数据/” 下
# ---------------------------------------------------------------
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "数据", "无人机应急物资运输基础数据")
DEM_FILE = os.path.join(BASE_DIR, "数据", "镇龙乡地理空间数据", "镇龙乡及周边地理数据",
                        "数字高程模型数据（DEM）", "镇龙乡及周边30米DEM.mat")
RESULT_DIR = os.path.join(BASE_DIR, "结果")
FIG_DIR = os.path.join(RESULT_DIR, "图")
os.makedirs(FIG_DIR, exist_ok=True)       # 结果文件夹不存在就自动创建

GRAVITY = 9.81            # 重力加速度 g（m/s²）
EARTH_R = 6371000.0       # 地球平均半径（m），用于经纬度换算成米
KWH_J = 3.6e6             # 1 kWh = 3.6×10^6 焦耳


def read_raw(file_name, sheet=0):
    """读取附件 Excel（不把第一行当表头），返回原始表格，后面按行号取值"""
    return pd.read_excel(os.path.join(DATA_DIR, file_name), sheet_name=sheet, header=None)


# ---------------------------------------------------------------
# 1. 读取节点（调度中心 O01 + 15 个服务区）
# ---------------------------------------------------------------
def load_nodes():
    raw = read_raw("调度中心与服务区.xlsx")
    nodes = {}
    r = raw.iloc[2]                                    # 第 3 行是 O01
    nodes["O01"] = {"name": r[1], "lon": float(r[2]), "lat": float(r[3]),
                    "alt": float(r[4]), "work_alt": float(r[4])}   # O01 作业高度 = 地面海拔
    for k in range(6, 21):                             # 第 7~21 行是 S001~S015
        r = raw.iloc[k]
        alt = float(r[4])
        nodes[r[0]] = {"name": r[1], "lon": float(r[2]), "lat": float(r[3]), "alt": alt,
                       "work_alt": alt + 30.0,         # 服务区作业高度 = 地面海拔 + 30 m
                       "people": int(r[5])}
    return nodes


# ---------------------------------------------------------------
# 2. 读取逐箱货箱清单
# ---------------------------------------------------------------
KIND_CODE = {"医疗物资": "MED", "饮用水": "WAT", "应急食品": "FOD", "生活卫生用品": "HYG"}


def load_boxes():
    df = pd.read_excel(os.path.join(DATA_DIR, "物资需求与配送时限.xlsx"), sheet_name="逐箱货箱清单")
    boxes = {}
    for _, r in df.iterrows():
        first = (r["是否首批保障"] == "是")
        kind = KIND_CODE[r["物资类型"]]
        due = float(r["期望送达时间（s）"])
        # “硬时限”：首批箱必须在首批截止时间前送达；医疗物资必须在期望送达时间前送达
        hard = None
        if first:
            hard = float(r["首批截止时间（s）"])
        if kind == "MED":
            hard = due if hard is None else min(hard, due)
        boxes[r["货箱编号"]] = {
            "id": r["货箱编号"], "area": r["服务区编号"], "kind": kind,
            "mass": float(r["单箱质量（kg）"]), "vol": float(r["单箱体积（m³）"]),
            "first": first, "due": due, "hard": hard, "coef": float(r["应急优先系数"]),
        }
    return boxes


# ---------------------------------------------------------------
# 3. 读取运输无人机（机型参数、实体无人机、共享电池）
# ---------------------------------------------------------------
def load_uav():
    raw = read_raw("运输无人机数据.xlsx")
    types = {}
    for k in range(2, 5):                              # A、B、C 三种机型
        r = raw.iloc[k]
        types[r[0]] = {
            "name": r[1], "empty": float(r[2]), "Q": float(r[3]), "V": float(r[4]),
            "v_c": float(r[5]), "L0": float(r[6]), "LF": float(r[7]), "E": float(r[8]),
            "rho": float(r[9]) / 100.0,                # 返航电量下限 20% → 0.20
            "prep": float(r[10]), "load": float(r[11]),
            "hand_base": float(r[12]), "hand_box": float(r[13]),
            "v_up": float(r[14]), "v_down": float(r[15]), "eta": float(r[16]),
        }
    drones = []                                        # [(无人机编号, 机型), ...]
    for k in range(8, 16):
        drones.append((raw.iloc[k][0], raw.iloc[k][1]))
    batteries = {}                                     # {机型: {"count": 数量, "full": 满充时间}}
    for k in range(19, 22):
        batteries[raw.iloc[k][0]] = {"count": int(raw.iloc[k][1]), "full": float(raw.iloc[k][2])}
    return types, drones, batteries


# ---------------------------------------------------------------
# 4. 读取中继无人机
# ---------------------------------------------------------------
def load_relay():
    raw = read_raw("中继无人机数据.xlsx")
    r = raw.iloc[2]
    relay = {
        "mass": float(r[4]),                           # 计划起飞总质量 23.5 kg
        "v_c": float(r[5]), "P_cruise": float(r[6]), "E": float(r[7]),
        "rho": float(r[8]) / 100.0, "prep": float(r[9]), "link": float(r[10]),
        "turn": float(r[11]), "v_up": float(r[12]), "v_down": float(r[13]),
        "eta": float(r[14]), "P_hover": float(r[16]), "P_comm": float(r[17]),
        "max_agl": float(r[18]),
    }
    relay_ids = [raw.iloc[6][0], raw.iloc[7][0]]       # R01、R02
    modules = {"count": int(raw.iloc[11][1]), "full": float(raw.iloc[11][2])}
    return relay, relay_ids, modules


# ---------------------------------------------------------------
# 5. 读取通信参数，并算出三种链路的“最大允许传播损耗”
# ---------------------------------------------------------------
def load_comm():
    raw = read_raw("通信链路参数.xlsx")
    v = [float(raw.iloc[k][4]) for k in range(2, 16)]  # 第 3~16 行第 5 列是参数值
    comm = {"f": v[0], "Lsys": v[1], "Lobs": v[2], "Psens": v[3], "M": v[4],
            "uav": {"Pt": v[5], "G": v[6]},
            "acc": {"Pt": v[7], "G": v[8]},            # 中继接入端（面向运输无人机）
            "bh": {"Pt": v[9], "G": v[10]},            # 中继回传端（面向 G01）
            "g01": {"Pt": v[11], "G": v[12]}, "hG": v[13]}
    p_th = comm["Psens"] + comm["M"]                   # 接收门限 = 灵敏度 + 衰落裕量

    def l_max(tx, rx):
        """单方向最大允许损耗 = 发射功率 + 发射增益 + 接收增益 - 系统损耗 - 接收门限"""
        return comm[tx]["Pt"] + comm[tx]["G"] + comm[rx]["G"] - comm["Lsys"] - p_th

    # 双向链路取两个方向中较小的那个
    comm["Lmax_direct"] = min(l_max("uav", "g01"), l_max("g01", "uav"))   # 运输机 ↔ G01
    comm["Lmax_access"] = min(l_max("uav", "acc"), l_max("acc", "uav"))   # 运输机 ↔ 中继
    comm["Lmax_back"] = min(l_max("bh", "g01"), l_max("g01", "bh"))       # 中继 ↔ G01
    return comm


# 模块被 import 时就把数据读好，其他文件直接用这些全局变量
NODES = load_nodes()
BOXES = load_boxes()
TYPES, DRONES, BATTERIES = load_uav()
RELAY, RELAY_IDS, MODULES = load_relay()
COMM = load_comm()
AREAS = [k for k in NODES if k != "O01"]              # ["S001", ..., "S015"]
LON0, LAT0 = NODES["O01"]["lon"], NODES["O01"]["lat"]
G01_POS = (LON0, LAT0, NODES["O01"]["alt"] + COMM["hG"])   # 网关天线三维位置


# ---------------------------------------------------------------
# 6. DEM：读取高程矩阵，按经纬度查询像元高程
# ---------------------------------------------------------------
_dem = sio.loadmat(DEM_FILE)
DEM_Z = _dem["dem"].astype(float)                     # 1309 行 × 1486 列 的高程矩阵
_tr = _dem["transform"].ravel()                       # [dx, 0, 左边界经度, 0, -dy, 上边界纬度]
DEM_X0, DEM_DX, DEM_Y0, DEM_DY = _tr[2], _tr[0], _tr[5], -_tr[4]
DEM_NROW, DEM_NCOL = DEM_Z.shape
DEM_Z[DEM_Z <= -32767] = np.nan                       # NoData 值换成 nan（本数据实际没有）
DEM_LON_MIN, DEM_LON_MAX = DEM_X0, DEM_X0 + DEM_NCOL * DEM_DX
DEM_LAT_MIN, DEM_LAT_MAX = DEM_Y0 - DEM_NROW * DEM_DY, DEM_Y0


def dem_at(lon, lat):
    """查询经纬度 (lon, lat) 所在像元的地面高程；支持 numpy 数组批量查询"""
    col = np.floor((np.asarray(lon) - DEM_X0) / DEM_DX).astype(int)   # 第几列
    row = np.floor((DEM_Y0 - np.asarray(lat)) / DEM_DY).astype(int)   # 第几行（纬度从上往下减小）
    col = np.clip(col, 0, DEM_NCOL - 1)
    row = np.clip(row, 0, DEM_NROW - 1)
    return DEM_Z[row, col]


def to_xy(lon, lat):
    """把经纬度换成以 O01 为原点的平面坐标（米）。10 km 范围内误差可忽略"""
    x = np.radians(np.asarray(lon) - LON0) * EARTH_R * math.cos(math.radians(LAT0))
    y = np.radians(np.asarray(lat) - LAT0) * EARTH_R
    return x, y


def hdist(lon1, lat1, lon2, lat2):
    """两点水平距离（米）"""
    x1, y1 = to_xy(lon1, lat1)
    x2, y2 = to_xy(lon2, lat2)
    return float(math.hypot(x2 - x1, y2 - y1))


# ---------------------------------------------------------------
# 7. 航段几何：巡航海拔 = 航段经过像元的最高地面高程 + 50 m
# ---------------------------------------------------------------
def segment_geometry(lon1, lat1, alt1, lon2, lat2, alt2):
    """
    输入：起点经纬度和作业高度 alt1，终点经纬度和作业高度 alt2
    输出：字典 {d: 水平距离, cruise: 巡航海拔, h_up: 爬升高度, h_down: 下降高度}
    """
    d = hdist(lon1, lat1, lon2, lat2)
    n = max(2, int(d / 5.0) + 2)                      # 每 5 m 取一个点，保证不漏掉 30 m 像元
    lons = np.linspace(lon1, lon2, n)
    lats = np.linspace(lat1, lat2, n)
    h_max = float(np.nanmax(dem_at(lons, lats)))      # 沿线最高地面高程
    cruise = max(h_max + 50.0, alt1, alt2)            # 中继悬停可能高于它，所以取较大者
    return {"d": d, "cruise": cruise, "h_up": cruise - alt1, "h_down": cruise - alt2}


_SEG_CACHE = {}


def node_segment(a, b):
    """两个节点（如 'O01'→'S003'）之间航段的几何信息，结果缓存起来避免重复计算"""
    if (a, b) not in _SEG_CACHE:
        na, nb = NODES[a], NODES[b]
        _SEG_CACHE[(a, b)] = segment_geometry(na["lon"], na["lat"], na["work_alt"],
                                              nb["lon"], nb["lat"], nb["work_alt"])
    return _SEG_CACHE[(a, b)]


def flight_time(seg, v_up, v_c, v_down):
    """航段飞行时间 t = 爬升高度/爬升速度 + 水平距离/巡航速度 + 下降高度/下降速度"""
    return seg["h_up"] / v_up + seg["d"] / v_c + seg["h_down"] / v_down


def equiv_range(g, q):
    """机型 g 带载荷 q 时的等效航程：L(q) = L0 - (L0 - LF) * (q / Q)^(3/2)"""
    t = TYPES[g]
    return t["L0"] - (t["L0"] - t["LF"]) * (q / t["Q"]) ** 1.5


def seg_energy(g, seg, q):
    """运输无人机航段能耗（kWh）= 水平巡航能耗 + 爬升附加能耗"""
    t = TYPES[g]
    e_hor = t["E"] * seg["d"] / equiv_range(g, q)
    e_up = (t["empty"] + q) * GRAVITY * seg["h_up"] / (t["eta"] * KWH_J)
    return e_hor + e_up


def seg_time(g, seg):
    t = TYPES[g]
    return flight_time(seg, t["v_up"], t["v_c"], t["v_down"])


def charge_time(soc, t_full):
    """两阶段充电：SOC < 90% 快充（占 65%），≥ 90% 慢充（占 35%），按增量线性折算"""
    if soc < 0.90:
        return t_full * (0.65 * (0.90 - soc) / 0.90 + 0.35)
    return t_full * 0.35 * (1.0 - soc) / 0.10


# ---------------------------------------------------------------
# 8. 通信链路计算
# ---------------------------------------------------------------
def los_blocked(p1, p2):
    """
    判断两点 p1=(经度, 纬度, 海拔)、p2 之间的视线是否被地形挡住。
    做法：在连线上每 15 m 取一个点，若某点的地面高程 ≥ 视线高度，就算遮挡。
    """
    d = hdist(p1[0], p1[1], p2[0], p2[1])
    if d < 60.0:                                      # 两点几乎在正上下方，不会被遮挡
        return False
    n = int(d / 15.0) + 2
    f = np.linspace(0.0, 1.0, n)                      # 0 表示 p1，1 表示 p2
    keep = (f * d > 30.0) & ((1 - f) * d > 30.0)      # 去掉两端 30 m 以内的点（假设 d）
    f = f[keep]
    lons = p1[0] + f * (p2[0] - p1[0])
    lats = p1[1] + f * (p2[1] - p1[1])
    line_h = p1[2] + f * (p2[2] - p1[2])              # 视线在各点的海拔
    return bool(np.any(dem_at(lons, lats) >= line_h))


def dist3d(p1, p2):
    """三维直线距离（米）"""
    return math.sqrt(hdist(p1[0], p1[1], p2[0], p2[1]) ** 2 + (p2[2] - p1[2]) ** 2)


def path_loss(p1, p2):
    """总传播损耗 = 自由空间损耗 + 遮挡附加损耗（若遮挡）"""
    d_km = max(dist3d(p1, p2), 1.0) / 1000.0
    fspl = 32.45 + 20 * math.log10(COMM["f"]) + 20 * math.log10(d_km)
    return fspl + (COMM["Lobs"] if los_blocked(p1, p2) else 0.0)


def link_ok(p1, p2, kind):
    """kind 取 'direct'（运输机-G01）、'access'（运输机-中继）、'back'（中继-G01）"""
    return path_loss(p1, p2) <= COMM["Lmax_" + kind]


# ---------------------------------------------------------------
# 9. 航段采样点（通信判定用）：按“垂直爬升 → 水平巡航 → 垂直下降”生成三维位置
# ---------------------------------------------------------------
def segment_samples(a, b):
    """
    返回航段 a→b 上一系列采样点，每个点记录：
      phase：0 爬升 / 1 巡航 / 2 下降；val：该阶段已完成的高度或距离；pos：三维位置
    """
    na, nb = NODES[a], NODES[b]
    seg = node_segment(a, b)
    pts = []
    for h in np.linspace(0, seg["h_up"], max(2, int(seg["h_up"] / 10) + 2)):      # 爬升每 10 m
        pts.append((0, h, (na["lon"], na["lat"], na["work_alt"] + h)))
    for x in np.linspace(0, seg["d"], max(2, int(seg["d"] / 50) + 2)):           # 巡航每 50 m
        f = x / seg["d"] if seg["d"] > 0 else 0
        pts.append((1, x, (na["lon"] + f * (nb["lon"] - na["lon"]),
                           na["lat"] + f * (nb["lat"] - na["lat"]), seg["cruise"])))
    for h in np.linspace(0, seg["h_down"], max(2, int(seg["h_down"] / 10) + 2)):  # 下降每 10 m
        pts.append((2, h, (nb["lon"], nb["lat"], seg["cruise"] - h)))
    return pts


def sample_time(g, seg, phase, val):
    """把采样点换算成“从航段起飞开始经过的秒数”（不同机型速度不同）"""
    t = TYPES[g]
    if phase == 0:
        return val / t["v_up"]
    if phase == 1:
        return seg["h_up"] / t["v_up"] + val / t["v_c"]
    return seg["h_up"] / t["v_up"] + seg["d"] / t["v_c"] + val / t["v_down"]


# ---------------------------------------------------------------
# 10. 中继无人机：往返某悬停点的时间与能耗
# ---------------------------------------------------------------
def relay_legs(lon, lat, alt):
    """返回中继从 O01 飞到悬停点、再飞回 O01 的时间和能耗（kWh）"""
    o = NODES["O01"]
    R = RELAY
    out_seg = segment_geometry(o["lon"], o["lat"], o["work_alt"], lon, lat, alt)
    back_seg = segment_geometry(lon, lat, alt, o["lon"], o["lat"], o["work_alt"])

    def leg(seg):
        t = flight_time(seg, R["v_up"], R["v_c"], R["v_down"])
        e = R["P_cruise"] * (seg["d"] / R["v_c"]) / 3600.0 \
            + R["mass"] * GRAVITY * seg["h_up"] / (R["eta"] * KWH_J)
        return t, e

    t_out, e_out = leg(out_seg)
    t_back, e_back = leg(back_seg)
    return {"t_out": t_out, "e_out": e_out, "t_back": t_back, "e_back": e_back}


def fmt_hms(sec):
    """秒 → “时:分:秒” 字符串，方便阅读"""
    sec = int(round(sec))
    return f"{sec // 3600:d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


if __name__ == "__main__":
    # 直接运行本文件时，打印一些基本检查信息
    print("链路门限(dB)：直连 %.1f，接入 %.1f，回传 %.1f" %
          (COMM["Lmax_direct"], COMM["Lmax_access"], COMM["Lmax_back"]))
    for s in AREAS:
        seg = node_segment("O01", s)
        print(s, "距离 %.0f m, 巡航海拔 %.1f, 爬升 %.1f, 下降 %.1f, 节点像元高程 %.1f/表格 %.1f"
              % (seg["d"], seg["cruise"], seg["h_up"], seg["h_down"],
                 dem_at(NODES[s]["lon"], NODES[s]["lat"]), NODES[s]["alt"]))
