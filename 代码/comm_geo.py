# -*- coding: utf-8 -*-
"""
comm_geo.py —— 问题三、问题四共用的“通信几何”预处理
=====================================================
1. 向量化的视线遮挡 / 链路预算判定（与 common.los_blocked / link_ok 口径完全一致：
   视线上每 15 m 取点、忽略两端 30 m、地面高程 ≥ 视线高度即遮挡）
2. 运输航段采样：对任意有向航段 a→b（16 个节点共 240 条）和每个服务区的投送悬停点，
   生成三维采样点，判定与 G01 的直连是否可用 → 得到“通信盲区采样点”
3. 候选中继悬停点：在 DEM 覆盖范围内按网格布点，悬停海拔 = 地面 + 最大离地高度 300 m，
   只保留回传链路（中继—G01）可用的点
4. 覆盖矩阵 COV[k, j]：候选点 k 能否为第 j 个盲区采样点提供接入链路（运输机—中继）
结果缓存到 结果/comm_geo.pkl，后续问题直接读取。
"""
import time
import pickle
import numpy as np
from common import *

CACHE_FILE = os.path.join(RESULT_DIR, "comm_geo.pkl")
FSPL0 = 32.45 + 20 * math.log10(COMM["f"])            # 自由空间损耗常数项（d 以 km 计）


def xy_arr(lon, lat):
    x, y = to_xy(lon, lat)
    return np.asarray(x, float), np.asarray(y, float)


def los_blocked_many(p, Q, step=15.0, chunk=4000):
    """p=(lon,lat,alt) 与 Q（N×3 数组）中每个点之间的视线是否被地形遮挡（布尔数组）"""
    Q = np.asarray(Q, float)
    N = len(Q)
    px, py = xy_arr(p[0], p[1])
    qx, qy = xy_arr(Q[:, 0], Q[:, 1])
    d = np.hypot(qx - px, qy - py)
    out = np.zeros(N, bool)
    idx_all = np.where(d >= 60.0)[0]                   # 与 common.los_blocked 一致：<60 m 不遮挡
    for s in range(0, len(idx_all), chunk):
        idx = idx_all[s:s + chunk]
        di = d[idx]
        n = (di / step).astype(int) + 2
        K = int(n.max())
        k = np.arange(K)[None, :]
        f = k / (n[:, None] - 1.0)
        valid = (k < n[:, None]) & (f * di[:, None] > 30.0) & ((1 - f) * di[:, None] > 30.0)
        lon = p[0] + f * (Q[idx, 0][:, None] - p[0])
        lat = p[1] + f * (Q[idx, 1][:, None] - p[1])
        h = p[2] + f * (Q[idx, 2][:, None] - p[2])
        g = dem_at(lon, lat)
        out[idx] = np.any(valid & (g >= h), axis=1)
    return out


def dist3d_many(p, Q):
    Q = np.asarray(Q, float)
    px, py = xy_arr(p[0], p[1])
    qx, qy = xy_arr(Q[:, 0], Q[:, 1])
    return np.sqrt((qx - px) ** 2 + (qy - py) ** 2 + (Q[:, 2] - p[2]) ** 2)


def link_ok_many(p, Q, kind):
    """p 与 Q 中每个点的双向链路是否可用（先用距离快速筛选，只对“遮挡与否决定成败”的点做视线计算）"""
    Q = np.asarray(Q, float)
    lmax = COMM["Lmax_" + kind]
    d = np.maximum(dist3d_many(p, Q), 1.0) / 1000.0
    fspl = FSPL0 + 20 * np.log10(d)
    ok = fspl + COMM["Lobs"] <= lmax                   # 即便遮挡也可用
    maybe = (~ok) & (fspl <= lmax)                     # 仅在无遮挡时可用
    if maybe.any():
        idx = np.where(maybe)[0]
        ok[idx] = ~los_blocked_many(p, Q[idx])
    return ok


# ---------------------------------------------------------------
# 运输航段采样
# ---------------------------------------------------------------
def leg_samples(a, b):
    """航段 a→b 的采样点：返回 (phase 数组, val 数组, pos N×3 数组)"""
    pts = segment_samples(a, b)
    ph = np.array([p[0] for p in pts], int)
    val = np.array([p[1] for p in pts], float)
    pos = np.array([p[2] for p in pts], float)
    return ph, val, pos


def sample_offsets(g, a, b, ph, val):
    """把采样点换算成相对航段起飞时刻的秒数（机型速度不同，时间不同）"""
    t = TYPES[g]
    seg = node_segment(a, b)
    out = np.where(ph == 0, val / t["v_up"],
                   np.where(ph == 1, seg["h_up"] / t["v_up"] + val / t["v_c"],
                            seg["h_up"] / t["v_up"] + seg["d"] / t["v_c"] + val / t["v_down"]))
    return out


def build(grid_step=0.0025, margin=0.035, verbose=True):
    t0 = time.time()
    names = list(NODES)
    legs = {}
    need_pos = []                                      # 所有盲区采样点位置（去重前按航段顺序）
    for a in names:
        for b in names:
            if a == b or (a == "O01" and b == "O01"):
                continue
            ph, val, pos = leg_samples(a, b)
            dok = link_ok_many(G01_POS, pos, "direct")
            nidx = np.where(~dok)[0]
            legs[(a, b)] = {"ph": ph, "val": val, "pos": pos, "direct": dok,
                            "need_local": nidx, "need_global": np.arange(len(need_pos), len(need_pos) + len(nidx))}
            need_pos.extend(pos[nidx].tolist())
    hovers = {}
    for s in AREAS:
        pos = np.array([[NODES[s]["lon"], NODES[s]["lat"], NODES[s]["work_alt"]]])
        dok = bool(link_ok_many(G01_POS, pos, "direct")[0])
        hovers[s] = {"pos": pos[0], "direct": dok,
                     "need_global": np.array([], int) if dok else np.array([len(need_pos)])}
        if not dok:
            need_pos.append(pos[0].tolist())
    need_pos = np.array(need_pos, float)
    if verbose:
        n_all = sum(len(v["pos"]) for v in legs.values())
        print(f"   航段 {len(legs)} 条，采样点 {n_all} 个，直连盲区点 {len(need_pos)} 个；用时 {time.time() - t0:.1f}s")

    # 候选中继悬停点（DEM 范围内）
    lons = [NODES[k]["lon"] for k in NODES]
    lats = [NODES[k]["lat"] for k in NODES]
    glon = np.arange(max(DEM_LON_MIN + 0.001, min(lons) - margin), min(DEM_LON_MAX - 0.001, max(lons) + margin), grid_step)
    glat = np.arange(max(DEM_LAT_MIN + 0.001, min(lats) - margin), min(DEM_LAT_MAX - 0.001, max(lats) + margin), grid_step)
    LO, LA = np.meshgrid(glon, glat)
    cand = np.column_stack([LO.ravel(), LA.ravel()])
    cand = np.column_stack([cand, dem_at(cand[:, 0], cand[:, 1]) + RELAY["max_agl"]])
    back = np.array([link_ok_many(G01_POS, c[None, :], "back")[0] for c in cand])
    cand = cand[back]
    if verbose:
        print(f"   候选网格 {LO.size} 个，回传可用 {len(cand)} 个")
    cov = np.zeros((len(cand), len(need_pos)), bool)
    t1 = time.time()
    for k, c in enumerate(cand):
        cov[k] = link_ok_many(c, need_pos, "access")
        if verbose and k % 200 == 0:
            print(f"   覆盖计算 {k}/{len(cand)}  用时 {time.time() - t1:.0f}s")
    data = {"legs": legs, "hovers": hovers, "need_pos": need_pos, "cand": cand, "cov": cov,
            "grid_step": grid_step}
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)
    if verbose:
        print(f"   预处理完成，总用时 {time.time() - t0:.0f}s，已缓存到 {CACHE_FILE}")
    return data


def load(rebuild=False):
    if not rebuild and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return build()


if __name__ == "__main__":
    # 一致性自检：向量化判定与 common.link_ok 逐点比较
    rng = np.random.default_rng(0)
    P = (109.24, 23.05, float(dem_at(109.24, 23.05)) + 300)
    Q = np.column_stack([rng.uniform(109.17, 109.29, 300), rng.uniform(23.0, 23.08, 300), rng.uniform(150, 600, 300)])
    v = link_ok_many(P, Q, "access")
    s = np.array([link_ok(tuple(q), P, "access") for q in Q])
    print("向量化与逐点判定一致：", bool((v == s).all()), f"（{v.sum()}/{len(v)} 可用）")
    load(rebuild=True)
