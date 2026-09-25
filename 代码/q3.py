# -*- coding: utf-8 -*-
"""
q3.py —— 问题三：通信约束下的运输与中继联合调度
=================================================
【建模思路】
 1. 通信盲区：comm_geo.py 对 240 条有向航段和 15 个投送点做三维采样，判定与 G01 的直连是否可用；
    直连不可用的采样点必须由“恰好一架”中继同时满足接入链路（运输机—中继）与回传链路（中继—G01）。
 2. 中继悬停点：在 DEM 范围内网格布点（离地 300 m，回传可用），得到覆盖矩阵 COV[候选点, 盲区采样点]。
    分析发现：任意两点都无法覆盖全部 12 个需中继的服务区 → 必须“多点位 + 分时段”部署，
    而中继实体只有 2 架，因此选择 K 个点位（K=3/4），由 2 架中继多个架次轮换值守。
 3. 通信剖面：给定点位集合，把每个运输架次的盲区采样序列切分为若干“点位-时段”需求 (k, a, b)，
    允许在不同点位之间切换（任一时刻只由一架中继保障）；按“最少切换”贪心切分，并生成
    “优先使用点位 k”的备选剖面，交给调度器挑选。
 4. 联合表调度（内层）：运输架次按紧急度排序，依次确定无人机、电池与开始时刻；
    对该架次的每个 (k, a, b) 需求，要么“延长”点位 k 上已有中继架次的服务窗口，
    要么“新开”中继架次（选中继实体 + 能源组件），取新增代价（能耗 + 架次）最小的可行方案；
    需满足：中继能量余量 ≥ 20%、中继架次周转 300 s、能源组件充满后才能再用、同一资源不重叠。
    若当前时刻无法保障，则在“资源释放事件时刻”集合中搜索最早可行开始时刻（运输等待中继）。
 5. 联合模拟退火（外层）：邻域 = 问题二的 8 种运输邻域（组批/机型/顺序/合并/拆分/优先级）
    + 中继点位邻域（点位局部平移 / 替换为高覆盖候选点），目标同时计入两类无人机的代价。
 6. 逐秒校验：对最终方案每架运输机逐秒插值三维位置，用 common.link_ok 精确复核通信状态。
"""
import time
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vrp
import comm_geo
from common import *

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False

GEO = comm_geo.load()
COV = GEO["cov"]
CAND = GEO["cand"]
LEGS = GEO["legs"]
HOVS = GEO["hovers"]
R = RELAY
E_LIMIT = (1 - R["rho"]) * R["E"]                     # 中继单架次可用能量（kWh）
P_SERVE = R["P_hover"] + R["P_comm"]                  # 悬停 + 通信附加功率（kW）

# 问题三目标权重（与问题二同一口径，另加中继的架次与能耗）
W3 = {"hard": 1000.0, "tard": 1.0, "T": 2.0, "E": 5.0, "N": 3.0, "NR": 3.0, "ER": 5.0}


# ===============================================================
# 1. 中继点位参数
# ===============================================================
_SITE_CACHE = {}


def site_info(ci):
    """候选点 ci 的往返时间与能耗、从起飞到建链完成的提前量、最长服务时长"""
    if ci not in _SITE_CACHE:
        lon, lat, alt = CAND[ci]
        lg = relay_legs(lon, lat, alt)
        lead = R["prep"] + lg["t_out"] + R["link"]
        e_fix = lg["e_out"] + lg["e_back"] + P_SERVE * R["link"] / 3600.0
        _SITE_CACHE[ci] = {"ci": ci, "pos": (float(lon), float(lat), float(alt)), "lead": lead,
                           "t_out": lg["t_out"], "t_back": lg["t_back"], "e_fix": e_fix,
                           "e_out": lg["e_out"], "e_back": lg["e_back"],
                           "max_serve": (E_LIMIT - e_fix) * 3600.0 / P_SERVE}
    return _SITE_CACHE[ci]


# ===============================================================
# 2. 运输架次的通信剖面
# ===============================================================
def trip_timeline(ev):
    """把架次的全部采样点按时间展开：返回 (相对时刻 T, 盲区编号 gid[-1 表示直连可用], 阶段标签)"""
    g = ev["g"]
    T, gid, lab = [], [], []
    hov = {s: (a, b) for s, a, b in ev["hovers"]}
    for a, b, t_off in ev["legs"]:
        L = LEGS[(a, b)]
        ts = t_off + comm_geo.sample_offsets(g, a, b, L["ph"], L["val"])
        gg = np.full(len(ts), -1, int)
        gg[L["need_local"]] = L["need_global"]
        name = {0: "爬升", 1: "巡航", 2: "下降"}
        T.extend(ts.tolist()); gid.extend(gg.tolist())
        lab.extend([f"{a}→{b} {name[p]}" for p in L["ph"]])
        if b in hov:
            h = HOVS[b]
            hid = int(h["need_global"][0]) if len(h["need_global"]) else -1
            T.extend(hov[b]); gid.extend([hid, hid]); lab.extend([f"{b} 投送悬停"] * 2)
    return np.array(T), np.array(gid), lab


_PROF_CACHE = {}


def trip_profiles(ev, sites):
    """
    给定点位集合 sites（候选点编号元组），返回该架次的备选通信剖面列表：
      每个剖面 = [(点位序号 k, 相对开始 a, 相对结束 b, 采样段 (i, j)), ...]
    若存在任何点位都覆盖不到的盲区采样点，返回 None（该架次在此点位集合下不可行）。
    """
    key = (vrp.trip_key(ev["g"], ev["stops"]), sites)
    if key in _PROF_CACHE:
        return _PROF_CACHE[key]
    T, gid, _ = trip_timeline(ev)
    need = gid >= 0
    if not need.any():
        _PROF_CACHE[key] = []
        return []
    idx = np.where(need)[0]
    C = COV[np.array(sites)][:, gid[idx]]              # K × (盲区点数)
    if not C.any(axis=0).all():
        _PROF_CACHE[key] = None
        return None
    K = len(sites)
    # 盲区连续段（run）
    runs = []
    start = idx[0]
    for u, v in zip(idx[:-1], idx[1:]):
        if v != u + 1:
            runs.append((start, u)); start = v
    runs.append((start, idx[-1]))
    pos_in = {int(i): n for n, i in enumerate(idx)}
    # reach[k, p]：点位 k 从第 p 个盲区点起连续覆盖（不跨 run）到 reach-1
    n_need = len(idx)
    run_last = np.zeros(n_need, bool)
    for _, i1 in runs:
        run_last[pos_in[int(i1)]] = True
    reach = np.zeros((K, n_need), int)
    for k in range(K):
        for p in range(n_need - 1, -1, -1):
            if not C[k, p]:
                reach[k, p] = p
            elif run_last[p]:
                reach[k, p] = p + 1
            else:
                reach[k, p] = reach[k, p + 1]

    def segment(prefer):
        plan = []
        for i0, i1 in runs:
            p = pos_in[int(i0)]
            p_end = pos_in[int(i1)]
            while p <= p_end:
                if prefer is not None and C[prefer, p]:
                    k = prefer
                else:
                    k = max(range(K), key=lambda kk: (reach[kk, p], -kk))
                q = min(reach[k, p], p_end + 1) - 1      # 该点位覆盖 [p, q]
                i, j = int(idx[p]), int(idx[q])
                # 两端各向外延伸一个采样间隔：进出盲区与点位交接时，前后保障方式有重叠，采样点之间不留空档
                a = T[i - 1] if i > 0 else T[i]
                b = T[j + 1] if j + 1 < len(T) else T[j]
                plan.append((k, float(a), float(b), i, j))
                p = q + 1
        # 同一点位相邻片段合并（间隔很短时视为同一需求）
        merged = []
        for seg in plan:
            if merged and merged[-1][0] == seg[0] and seg[1] - merged[-1][2] < 120.0:
                m = merged[-1]
                merged[-1] = (m[0], m[1], seg[2], m[3], seg[4])
            else:
                merged.append(seg)
        return tuple(merged)

    alts = {segment(None)}
    for k in range(K):
        alts.add(segment(k))
    alts = sorted(alts, key=lambda pl: (len(pl), len({s[0] for s in pl})))
    _PROF_CACHE[key] = alts
    return alts


# ===============================================================
# 3. 中继资源状态与“延长 / 新开”操作
# ===============================================================
def sortie_derived(s, S):
    """由服务窗口 [ws, we] 推出起飞、返回、能耗、SOC、充满时刻"""
    si = S[s["k"]]
    launch = s["ws"] - si["lead"]
    ret = s["we"] + si["t_back"]
    e = si["e_fix"] + P_SERVE * (s["we"] - s["ws"]) / 3600.0
    soc = 1 - e / R["E"]
    return launch, ret, e, soc, ret + charge_time(soc, MODULES["full"])


def sortie_ok(s, others, S):
    launch, ret, e, soc, cend = sortie_derived(s, S)
    if launch < -1e-6 or e > E_LIMIT + 1e-9:
        return False
    for o in others:
        if o is s:
            continue
        ol, orr, _, _, oc = o["_d"]
        if o["r"] == s["r"] and launch < orr + R["turn"] - 1e-6 and ol < ret + R["turn"] - 1e-6:
            return False
        if o["m"] == s["m"] and launch < oc - 1e-6 and ol < cend - 1e-6:
            return False
    return True


MODULE_IDS = [f"M{k + 1:02d}" for k in range(MODULES["count"])]


def apply_need(state, k, a, b, S, W):
    """
    把点位 k 在 [a, b] 的保障需求加入中继状态 state（列表，元素为中继架次字典）。
    返回 (新增代价, 使用的中继架次 uid)；不可行返回 None。state 会被原地修改。
    """
    best = None
    # 方案一：延长点位 k 上已有架次
    for s in state:
        if s["k"] != k:
            continue
        ws, we = min(s["ws"], a), max(s["we"], b)
        if ws == s["ws"] and we == s["we"]:
            return 0.0, s["uid"]
        old = (s["ws"], s["we"], s["_d"])
        s["ws"], s["we"] = ws, we
        s["_d"] = sortie_derived(s, S)
        if sortie_ok(s, state, S):
            dc = W["ER"] * (s["_d"][2] - old[2][2])
            if best is None or dc < best[0]:
                best = (dc, "ext", s, ws, we)
        s["ws"], s["we"], s["_d"] = old
    # 方案二：新开一个中继架次（选中继实体与能源组件）
    si = S[k]
    if b - a <= si["max_serve"] + 1e-6:
        probe = {"k": k, "ws": a, "we": b, "r": None, "m": None}
        probe["_d"] = sortie_derived(probe, S)
        dc = W["ER"] * probe["_d"][2] + W["NR"]
        if (best is None or dc < best[0]) and probe["_d"][0] >= -1e-6:
            found = None
            for r in RELAY_IDS:
                probe["r"] = r
                for m in MODULE_IDS:
                    probe["m"] = m
                    if sortie_ok(probe, state, S):
                        found = (r, m)
                        break
                if found:
                    break
            if found:
                best = (dc, "new", found, a, b)
    if best is None:
        return None
    if best[1] == "ext":
        s = best[2]
        s["ws"], s["we"] = best[3], best[4]
        s["_d"] = sortie_derived(s, S)
        return best[0], s["uid"]
    uid = max((s["uid"] for s in state), default=-1) + 1
    s = {"uid": uid, "k": k, "ws": best[3], "we": best[4], "r": best[2][0], "m": best[2][1]}
    s["_d"] = sortie_derived(s, S)
    state.append(s)
    return best[0], uid


def copy_state(state):
    return [dict(s) for s in state]


def try_profile(state, prof, t, S, W):
    """在开始时刻 t 尝试满足整个剖面；成功返回 (新状态, 代价, 分配)，否则 None"""
    st = copy_state(state)
    cost = 0.0
    assign = []
    for k, a, b, i, j in prof:
        r = apply_need(st, k, t + a, t + b, S, W)
        if r is None:
            return None
        cost += r[0]
        assign.append((k, a, b, i, j, r[1]))
    return st, cost, assign


def candidate_starts(state, profs, t0, S):
    """最早开始时刻候选集合：t0 + 各类资源释放事件对齐到需求开始的时刻 + 兜底网格"""
    ev_times = [0.0]
    for s in state:
        l, r, _, _, c = s["_d"]
        ev_times += [r + R["turn"], c]
    cands = set()
    for prof in profs:
        for k, a, b, _, _ in prof:
            lead = S[k]["lead"]
            for e in ev_times:
                cands.add(e + lead - a)
            for s in state:
                if s["k"] == k:
                    cands.add(s["ws"] - a)
    out = sorted(c for c in cands if c > t0 + 1e-6)[:80]
    grid = [t0 + 120.0 * i for i in range(1, 150)]
    return [t0] + sorted(set(out + grid))


# ===============================================================
# 4. 联合表调度
# ===============================================================
def joint_schedule(sol, sites, W=None, want_detail=False):
    W = W or W3
    S = [site_info(ci) for ci in sites]
    evals = []
    for idx, tr in enumerate(sol):
        ev = vrp.eval_trip(tr["g"], tr["stops"])
        if ev is None:
            return None
        profs = trip_profiles(ev, sites)
        if profs is None:
            return None
        evals.append((ev["slack"] + tr["bias"], idx, ev, profs))
    evals.sort(key=lambda x: (x[0], x[1]))
    drone_free = {u: 0.0 for u, g in DRONES}
    batt_ready = {f"{g}-B{k + 1:02d}": 0.0 for g, info in BATTERIES.items() for k in range(info["count"])}
    state = []
    plan = []
    for _, idx, ev, profs in evals:
        g = ev["g"]
        u = min((x for x, gg in DRONES if gg == g), key=lambda x: (drone_free[x], x))
        b = min((x for x in batt_ready if x.startswith(g + "-")), key=lambda x: (batt_ready[x], x))
        t0 = max(drone_free[u], batt_ready[b])
        start, assign = t0, []
        if profs:
            done = None
            for t in candidate_starts(state, profs, t0, S):
                best = None
                for prof in profs:
                    r = try_profile(state, prof, t, S, W)
                    if r is not None and (best is None or r[1] < best[1]):
                        best = r
                if best is not None:
                    done = (t, best)
                    break
            if done is None:
                return None
            start = done[0]
            state, assign = done[1][0], done[1][2]
        end = start + ev["dur"]
        drone_free[u] = end
        batt_ready[b] = end + ev["charge"]
        plan.append({"idx": idx, "ev": ev, "uav": u, "batt": b, "start": start, "end": end,
                     "charge_end": end + ev["charge"], "wait": start - t0, "assign": assign})
    plan.sort(key=lambda p: (p["start"], p["uav"]))
    return plan, state, S


def evaluate(sol, sites, W=None, detail=False):
    W = W or W3
    res = joint_schedule(sol, sites, W)
    if res is None:
        return (float("inf"), None) if detail else float("inf")
    plan, state, S = res
    hard_v = tard = 0.0
    on_time = 0
    for p in plan:
        for i, off in p["ev"]["deliver"].items():
            tt = p["start"] + off
            bx = BOXES[i]
            if bx["hard"] is not None:
                hard_v += max(0.0, tt - bx["hard"])
            late = max(0.0, tt - bx["due"])
            tard += bx["coef"] * late / 60.0
            on_time += late <= 1e-6
    t_ms = float(max(p["end"] for p in plan))
    r_ms = float(max((s["_d"][1] for s in state), default=0.0))
    makespan = max(t_ms, r_ms)
    e_t = float(sum(p["ev"]["E"] for p in plan))
    e_r = float(sum(s["_d"][2] for s in state))
    on_time = int(on_time)
    cost = (W["hard"] * hard_v / 60.0 + W["tard"] * tard + W["T"] * makespan / 60.0
            + W["E"] * e_t + W["N"] * len(plan) + W["ER"] * e_r + W["NR"] * len(state))
    if not detail:
        return cost
    for k, p in enumerate(plan, 1):
        p["tid"] = f"Q3-T{k:02d}"
    for k, s in enumerate(sorted(state, key=lambda s: (s["_d"][0], s["r"])), 1):
        s["rid"] = f"Q3-R{k:02d}"
    return cost, {"cost": cost, "plan": plan, "state": state, "S": S, "hard_violation_s": hard_v,
                  "weighted_tardiness": tard, "on_time": on_time, "makespan": makespan,
                  "transport_makespan": t_ms, "relay_makespan": r_ms, "energy_t": e_t, "energy_r": e_r,
                  "trips": len(plan), "relay_sorties": len(state)}


# ===============================================================
# 5. 点位选择：初始解（样本级贪心集合覆盖）+ 邻域
# ===============================================================
def need_ids_of(sol):
    ids = []
    for tr in sol:
        ev = vrp.eval_trip(tr["g"], tr["stops"])
        _, gid, _ = trip_timeline(ev)
        ids.extend(gid[gid >= 0].tolist())
    return np.unique(np.array(ids, int))


def round_trip_ids():
    ids = []
    for s in AREAS:
        for key in (("O01", s), (s, "O01")):
            ids.extend(LEGS[key]["need_global"].tolist())
        ids.extend(HOVS[s]["need_global"].tolist())
    return np.unique(np.array(ids, int))


def greedy_sites(ids, K):
    """贪心：每次选能覆盖最多“尚未覆盖盲区点”的候选点；覆盖完后继续按“单点覆盖整段”补足 K 个"""
    left = np.ones(len(ids), bool)
    chosen = []
    C = COV[:, ids]
    while len(chosen) < K:
        gain = (C & left).sum(axis=1)
        gain[chosen] = -1
        k = int(np.argmax(gain))
        if gain[k] <= 0:
            tot = C.sum(axis=1).astype(float)
            tot[chosen] = -1
            k = int(np.argmax(tot))
        chosen.append(k)
        left &= ~C[k]
    return tuple(chosen), int(left.sum())


def exact_cover(ids, time_limit=120):
    """最小集合覆盖（0-1 整数规划，HiGHS 求解）：最少用几个候选点能覆盖 ids 中的全部盲区采样点"""
    from scipy.optimize import milp, LinearConstraint, Bounds
    A = np.unique(COV[:, ids].T.astype(float), axis=0)
    n = A.shape[1]
    r = milp(np.ones(n), constraints=LinearConstraint(A, lb=1, ub=np.inf), integrality=np.ones(n),
             bounds=Bounds(0, 1), options={"time_limit": time_limit})
    if r.x is None:
        return None
    return tuple(int(k) for k in np.where(r.x > 0.5)[0])


_GRID_NB = {}


def grid_neighbors(ci, radius):
    key = (ci, radius)
    if key not in _GRID_NB:
        step = GEO["grid_step"]
        d = np.abs(CAND[:, :2] - CAND[ci, :2])
        m = (d[:, 0] <= radius * step + 1e-9) & (d[:, 1] <= radius * step + 1e-9)
        m[ci] = False
        _GRID_NB[key] = np.where(m)[0]
    return _GRID_NB[key]


TOP_POOL = None


def site_neighbor(sites):
    sites = list(sites)
    i = random.randrange(len(sites))
    if random.random() < 0.7:
        nb = grid_neighbors(sites[i], random.choice((1, 2, 3)))
        if len(nb) == 0:
            return None
        sites[i] = int(random.choice(nb))
    else:
        sites[i] = int(random.choice(TOP_POOL))
    if len(set(sites)) < len(sites):
        return None
    return tuple(sites)


# ===============================================================
# 6. 联合模拟退火
# ===============================================================
def joint_anneal(sol, sites, iters=30000, T0=8.0, T_end=0.02, seed=0, W=None, p_site=0.15, verbose=True,
                 restarts=4):
    """联合退火；每 iters/restarts 次迭代回到历史最优解并重新升温（分段重启，避免在劣解区漂移）"""
    W = W or W3
    random.seed(seed)
    cur = (sol, sites)
    cur_c = evaluate(sol, sites, W)
    best, best_c = cur, cur_c
    t0 = time.time()
    seg = max(1, iters // restarts)
    for it in range(iters):
        if it and it % seg == 0:
            cur, cur_c = best, best_c
        T = T0 * (T_end / T0) ** ((it % seg) / seg)
        if random.random() < p_site:
            ns = site_neighbor(cur[1])
            if ns is None:
                continue
            cand = (cur[0], ns)
        else:
            nsol = vrp.neighbor(cur[0])
            if nsol is None:
                continue
            cand = (nsol, cur[1])
        c = evaluate(cand[0], cand[1], W)
        if c < cur_c or (c < float("inf") and random.random() < math.exp(-(c - cur_c) / T)):
            cur, cur_c = cand, c
            if c < best_c - 1e-9:
                best, best_c = cand, c
        if verbose and it % 5000 == 0:
            print(f"   迭代 {it:6d}  温度 {T:7.3f}  当前 {cur_c:9.2f}  最优 {best_c:9.2f}  用时 {time.time() - t0:.0f}s")
    return best[0], best[1], best_c


def add_candidate(lon, lat):
    """在线新增一个候选悬停点（离地 300 m），回传不可用返回 None；返回候选编号"""
    global CAND, COV
    lon = float(np.clip(lon, DEM_LON_MIN + 1e-4, DEM_LON_MAX - 1e-4))
    lat = float(np.clip(lat, DEM_LAT_MIN + 1e-4, DEM_LAT_MAX - 1e-4))
    p = np.array([lon, lat, float(dem_at(lon, lat)) + R["max_agl"]])
    if not link_ok(tuple(p), G01_POS, "back"):
        return None
    row = comm_geo.link_ok_many(tuple(p), GEO["need_pos"], "access")
    CAND = np.vstack([CAND, p])
    COV = np.vstack([COV, row])
    return len(CAND) - 1


def refine_sites(sol, sites, W=None, steps=(0.001, 0.0005, 0.00025), verbose=True):
    """
    点位精细化（坐标轮换的模式搜索）：对每个点位在 3×3 邻域内按逐级缩小的步长试探，
    在线计算新点的覆盖向量，接受使联合目标下降的位置。
    """
    W = W or W3
    best_c = evaluate(sol, sites, W)
    for step in steps:
        improved = True
        while improved:
            improved = False
            for i in range(len(sites)):
                lon, lat = CAND[sites[i]][:2]
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        ci = add_candidate(lon + dx * step, lat + dy * step)
                        if ci is None:
                            continue
                        ns = tuple(ci if j == i else s for j, s in enumerate(sites))
                        c = evaluate(sol, ns, W)
                        if c < best_c - 1e-6:
                            sites, best_c, improved = ns, c, True
                            lon, lat = CAND[ci][:2]
        if verbose:
            print(f"   点位精细化 步长 {step:.5f}°（约 {step * 111000:.0f} m）：目标 {best_c:.2f}")
    return sites, best_c


def polish(sol, sites, W=None, rounds=4, tries=4000):
    """退火后“只接受改进”的爬山修正（运输邻域 + 点位邻域交替）"""
    W = W or W3
    best_c = evaluate(sol, sites, W)
    for _ in range(rounds):
        improved = False
        for _ in range(tries):
            if random.random() < 0.3:
                ns = site_neighbor(sites)
                if ns is None:
                    continue
                c = evaluate(sol, ns, W)
                if c < best_c - 1e-9:
                    sites, best_c, improved = ns, c, True
            else:
                nsol = vrp.neighbor(sol)
                if nsol is None:
                    continue
                c = evaluate(nsol, sites, W)
                if c < best_c - 1e-9:
                    sol, best_c, improved = nsol, c, True
        if not improved:
            break
    return sol, sites, best_c


# ===============================================================
# 7. 逐秒校验（精确复核）
# ===============================================================
def trip_track(ev):
    """架次三维轨迹关键帧 (t, lon, lat, alt)，线性插值即可得到任意时刻位置"""
    g = ev["g"]
    tp = TYPES[g]
    o = NODES["O01"]
    kf = [(0.0, o["lon"], o["lat"], o["work_alt"])]
    hov = {s: (a, b) for s, a, b in ev["hovers"]}
    for a, b, t_off in ev["legs"]:
        na, nb = NODES[a], NODES[b]
        seg = node_segment(a, b)
        t1 = t_off + seg["h_up"] / tp["v_up"]
        t2 = t1 + seg["d"] / tp["v_c"]
        t3 = t2 + seg["h_down"] / tp["v_down"]
        kf += [(t_off, na["lon"], na["lat"], na["work_alt"]), (t1, na["lon"], na["lat"], seg["cruise"]),
               (t2, nb["lon"], nb["lat"], seg["cruise"]), (t3, nb["lon"], nb["lat"], nb["work_alt"])]
        if b in hov:
            kf.append((hov[b][1], nb["lon"], nb["lat"], nb["work_alt"]))
    return np.array(kf)


def phase_list(ev):
    """架次的通信阶段（相对时刻）：各航段的爬升/巡航/下降 + 各服务区投送悬停"""
    tp = TYPES[ev["g"]]
    hov = {s: (a, b) for s, a, b in ev["hovers"]}
    out = []
    for a, b, t_off in ev["legs"]:
        seg = node_segment(a, b)
        t1 = t_off + seg["h_up"] / tp["v_up"]
        t2 = t1 + seg["d"] / tp["v_c"]
        t3 = t2 + seg["h_down"] / tp["v_down"]
        out += [(f"{a}→{b} 爬升", t_off, t1), (f"{a}→{b} 巡航", t1, t2), (f"{a}→{b} 下降", t2, t3)]
        if b in hov:
            out.append((f"{b} 投送", hov[b][0], hov[b][1]))
    return out


def verify(info, dt=1.0):
    """
    逐秒复核：对每架运输机的每个通信阶段按 1 s 插值三维位置，用精确链路判定确定保障方式：
      直连可用 → 直连；否则在“已建链且处于服务窗口”的中继中选接入可用者（优先沿用当前中继，
      保证任一时刻只由一架中继保障）；都不满足 → 中断。结果同时写入 p["comm_rows"] 供输出。
    """
    S = info["S"]
    total = relay_sec = direct_sec = bad = 0
    bad_list = []
    for p in info["plan"]:
        ev = p["ev"]
        kf = trip_track(ev)
        rows = []
        cur = None
        for lab, a, b in phase_list(ev):
            if b - a < 1e-9:
                continue
            ts = np.unique(np.concatenate([np.arange(a, b, dt), [b]]))
            pos = np.column_stack([np.interp(ts, kf[:, 0], kf[:, c]) for c in (1, 2, 3)])
            dok = comm_geo.link_ok_many(G01_POS, pos, "direct")
            tab = p["start"] + ts
            rel = {}
            for s in info["state"]:
                act = (tab >= s["ws"] - 1e-6) & (tab <= s["we"] + 1e-6) & ~dok
                if act.any():
                    m = np.zeros(len(ts), bool)
                    m[act] = comm_geo.link_ok_many(S[s["k"]]["pos"], pos[act], "access")
                    if m.any():
                        rel[s["uid"]] = m
            mode = []
            for q in range(len(ts)):
                if dok[q]:
                    md = -1
                elif cur in rel and rel[cur][q]:
                    md = cur
                else:
                    md = next((u for u, m in rel.items() if m[q]), -2)
                if md >= 0:
                    cur = md
                mode.append(md)
            n = len(ts) - 1 if len(ts) > 1 else 1
            for q in range(n):
                t_a, t_b = tab[q], tab[min(q + 1, len(ts) - 1)]
                if rows and rows[-1][2] == lab and rows[-1][3] == mode[q]:
                    rows[-1][1] = t_b
                else:
                    rows.append([t_a, t_b, lab, mode[q]])
                span = t_b - t_a
                total += span
                if mode[q] == -1:
                    direct_sec += span
                elif mode[q] >= 0:
                    relay_sec += span
                else:
                    bad += span
                    bad_list.append((p.get("tid", p["idx"]), lab, round(t_a, 1)))
        p["comm_rows"] = rows
    back_ok = all(link_ok(S[s["k"]]["pos"], G01_POS, "back") for s in info["state"])
    return {"total_s": total, "direct_s": direct_sec, "relay_s": relay_sec, "outage_s": bad,
            "bad_trips": bad_list, "backhaul_ok": back_ok}


PH_CODE = {"爬升": 0, "巡航": 1, "下降": 2}


def augment(info, ver, half_window=1.5, step=1.0):
    """
    闭环修正（割平面思想）：对逐秒复核发现的中断时刻，在对应航段的该位置前后 ±1.5 s 飞行距离内
    按 1 m 步长加密采样，重新精确判定直连，并把新增盲区点的覆盖列并入 COV；
    随后清空剖面缓存，调度器会在新的盲区点上自动安排中继（或调整切换时机）。返回新增盲区点数。
    """
    global COV
    tid2p = {p["tid"]: p for p in info["plan"]}
    add = {}
    for tid, lab, t_a in ver["bad_trips"]:
        if "→" not in lab:
            continue
        p = tid2p[tid]
        ev = p["ev"]
        tp = TYPES[ev["g"]]
        leg, ph_name = lab.split(" ")
        a, b = leg.split("→")
        ph = PH_CODE[ph_name]
        t_off = next(o for x, y, o in ev["legs"] if x == a and y == b)
        seg = node_segment(a, b)
        t_rel = t_a - p["start"]
        t1 = t_off + seg["h_up"] / tp["v_up"]
        t2 = t1 + seg["d"] / tp["v_c"]
        v = [(t_rel - t_off) * tp["v_up"], (t_rel - t1) * tp["v_c"], (t_rel - t2) * tp["v_down"]][ph]
        spd = [tp["v_up"], tp["v_c"], tp["v_down"]][ph]
        top = [seg["h_up"], seg["d"], seg["h_down"]][ph]
        vals = np.arange(v - half_window * spd, v + (half_window + 1) * spd, step)
        add.setdefault((a, b), set()).update((ph, round(float(x), 3)) for x in vals if 0 <= x <= top)
    new_cols, new_pos = [], []
    n0 = COV.shape[1]
    for (a, b), items in add.items():
        L = LEGS[(a, b)]
        na, nb = NODES[a], NODES[b]
        seg = node_segment(a, b)
        exist = set(zip(L["ph"].tolist(), np.round(L["val"], 3).tolist()))
        items = sorted(x for x in items if x not in exist)
        if not items:
            continue
        ph = np.array([x[0] for x in items]); val = np.array([x[1] for x in items])
        f = np.where(ph == 1, val / max(seg["d"], 1e-9), np.where(ph == 2, 1.0, 0.0))
        alt = np.where(ph == 0, na["work_alt"] + val, np.where(ph == 1, seg["cruise"], seg["cruise"] - val))
        pos = np.column_stack([na["lon"] + f * (nb["lon"] - na["lon"]), na["lat"] + f * (nb["lat"] - na["lat"]), alt])
        dok = comm_geo.link_ok_many(G01_POS, pos, "direct")
        gid_old = np.full(len(L["ph"]), -1, int)
        gid_old[L["need_local"]] = L["need_global"]
        gid_new = np.full(len(ph), -1, int)
        for q in np.where(~dok)[0]:
            gid_new[q] = n0 + len(new_pos)
            new_pos.append(pos[q])
            new_cols.append(comm_geo.link_ok_many(tuple(pos[q]), CAND, "access"))
        PH = np.concatenate([L["ph"], ph]); VAL = np.concatenate([L["val"], val])
        order = np.lexsort((VAL, PH))
        L["ph"], L["val"] = PH[order], VAL[order]
        L["pos"] = np.concatenate([L["pos"], pos])[order]
        L["direct"] = np.concatenate([L["direct"], dok])[order]
        gid = np.concatenate([gid_old, gid_new])[order]
        L["need_local"] = np.where(gid >= 0)[0]
        L["need_global"] = gid[L["need_local"]]
    if new_cols:
        COV = np.hstack([COV, np.stack(new_cols, axis=1)])
        GEO["need_pos"] = np.vstack([GEO["need_pos"], np.array(new_pos)])
    _PROF_CACHE.clear()
    return len(new_pos), sum(len(v) for v in add.values())


# ===============================================================
# 8. 输出
# ===============================================================
def relay_rows(info):
    S = info["S"]
    rows = []
    st = sorted(info["state"], key=lambda s: s["rid"])
    for s in st:
        l, r, e, soc, c = s["_d"]
        pos = S[s["k"]]["pos"]
        rows.append({"中继架次编号": s["rid"], "中继无人机编号": s["r"], "能源组件编号": s["m"],
                     "开始时刻（s）": round(l, 1), "悬停经度（°）": round(pos[0], 6), "悬停纬度（°）": round(pos[1], 6),
                     "悬停海拔（m）": round(pos[2], 1), "建链完成时刻（s）": round(s["ws"], 1),
                     "服务结束时刻（s）": round(s["we"], 1), "返回O01时刻（s）": round(r, 1),
                     "架次能耗（kWh）": round(e, 4), "悬停点位": f"P{s['k'] + 1}",
                     "悬停离地高度(m)": round(pos[2] - float(dem_at(pos[0], pos[1])), 1),
                     "返航SOC(%)": round(soc * 100, 2), "组件充满时刻(s)": round(c, 1)})
    return pd.DataFrame(rows)


def transport_rows(info):
    trips, boxes, comm = [], [], []
    by_uid = {s["uid"]: s for s in info["state"]}
    for p in info["plan"]:
        ev = p["ev"]
        trips.append({"架次编号": p["tid"], "无人机编号": p["uav"], "机型编号": ev["g"], "电池编号": p["batt"],
                      "开始时刻（s）": round(p["start"], 1),
                      "访问服务区顺序": "O01→" + "→".join(s for s, _ in ev["stops"]) + "→O01",
                      "返回O01时刻（s）": round(p["end"], 1), "架次能耗（kWh）": round(ev["E"], 4),
                      "载重(kg)": ev["mass"], "体积(m³)": round(ev["vol"], 4),
                      "返航SOC(%)": round(ev["soc"] * 100, 2), "电池充满时刻(s)": round(p["charge_end"], 1),
                      "等待中继(s)": round(p["wait"], 1),
                      "中继保障时长(s)": round(sum(r[1] - r[0] for r in p["comm_rows"] if r[3] >= 0), 1),
                      "依托中继架次": ",".join(sorted({by_uid[r[3]]["rid"] for r in p["comm_rows"] if r[3] >= 0}))
                      or "无（全程直连）"})
        for s, ids in ev["stops"]:
            for i in ids:
                t = p["start"] + ev["deliver"][i]
                bx = BOXES[i]
                boxes.append({"货箱编号": i, "架次编号": p["tid"], "服务区编号": s, "交付完成时刻（s）": round(t, 1),
                              "期望送达时间(s)": bx["due"], "硬时限(s)": bx["hard"],
                              "是否准时": "是" if t <= bx["due"] + 1e-6 else "否",
                              "延误(s)": round(max(0, t - bx["due"]), 1)})
        for t_a, t_b, lab, m in p["comm_rows"]:
            comm.append({"运输架次编号": p["tid"], "通信阶段": lab, "开始时刻（s）": round(t_a, 1),
                         "结束时刻（s）": round(t_b, 1),
                         "保障方式": {-1: "直连G01", -2: "中断"}.get(m, "中继"),
                         "中继架次编号": by_uid[m]["rid"] if m >= 0 else ""})
    boxes.sort(key=lambda r: r["货箱编号"])
    return pd.DataFrame(trips), pd.DataFrame(boxes), pd.DataFrame(comm)


def resource_rows(info):
    plan, state = info["plan"], info["state"]
    rows = []
    for u, g in DRONES:
        ps = [p for p in plan if p["uav"] == u]
        rows.append({"资源": u, "类型": f"{g}型运输无人机", "使用次数": len(ps),
                     "占用时长(s)": round(sum(p["end"] - p["start"] for p in ps), 1),
                     "承担架次": ",".join(p["tid"] for p in ps)})
    for g, inf in BATTERIES.items():
        for k in range(inf["count"]):
            b = f"{g}-B{k + 1:02d}"
            ps = [p for p in plan if p["batt"] == b]
            rows.append({"资源": b, "类型": f"{g}型共享电池", "使用次数": len(ps),
                         "占用时长(s)": round(sum(p["end"] - p["start"] for p in ps), 1),
                         "承担架次": ",".join(p["tid"] for p in ps),
                         "充电时长(s)": round(sum(p["ev"]["charge"] for p in ps), 1)})
    for r in RELAY_IDS:
        ss = [s for s in state if s["r"] == r]
        rows.append({"资源": r, "类型": "中继无人机", "使用次数": len(ss),
                     "占用时长(s)": round(sum(s["_d"][1] - s["_d"][0] for s in ss), 1),
                     "承担架次": ",".join(s["rid"] for s in sorted(ss, key=lambda s: s["_d"][0]))})
    for m in MODULE_IDS:
        ss = [s for s in state if s["m"] == m]
        rows.append({"资源": m, "类型": "中继能源组件", "使用次数": len(ss),
                     "占用时长(s)": round(sum(s["_d"][1] - s["_d"][0] for s in ss), 1),
                     "承担架次": ",".join(s["rid"] for s in sorted(ss, key=lambda s: s["_d"][0])),
                     "充电时长(s)": round(sum(s["_d"][4] - s["_d"][1] for s in ss), 1)})
    return pd.DataFrame(rows)


def check_rows(info, ver):
    plan, state, S = info["plan"], info["state"], info["S"]
    checks = vrp.check_plan(plan)
    ok_e = all(s["_d"][2] <= E_LIMIT + 1e-9 for s in state)
    checks.append(("中继架次返航电量≥20%", ok_e,
                   f"最低返航SOC {min(s['_d'][3] for s in state) * 100:.2f}%" if state else "无中继架次"))
    ok = True
    for key, busy in (("r", lambda s: s["_d"][1] + R["turn"]), ("m", lambda s: s["_d"][4])):
        grp = {}
        for s in state:
            grp.setdefault(s[key], []).append(s)
        for ss in grp.values():
            ss.sort(key=lambda s: s["_d"][0])
            for x, y in zip(ss, ss[1:]):
                if y["_d"][0] < busy(x) - 1e-6:
                    ok = False
    checks.append(("中继无人机周转≥300s、能源组件占用+充电不重叠", ok,
                   f"使用中继 {len({s['r'] for s in state})} 架，能源组件 {len({s['m'] for s in state})} 组"))
    agl = [S[s["k"]]["pos"][2] - float(dem_at(*S[s["k"]]["pos"][:2])) for s in state]
    inside = all(DEM_LON_MIN <= S[s["k"]]["pos"][0] <= DEM_LON_MAX and
                 DEM_LAT_MIN <= S[s["k"]]["pos"][1] <= DEM_LAT_MAX for s in state)
    checks.append(("悬停点位于DEM范围内、离地高度≤300m", inside and all(a <= R["max_agl"] + 1e-6 for a in agl),
                   f"离地高度 {min(agl):.0f}~{max(agl):.0f} m" if agl else "-"))
    checks.append(("中继回传链路（中继—G01）可用", ver["backhaul_ok"], ""))
    checks.append(("逐秒复核：运输机飞行与投送全程通信不中断", ver["outage_s"] == 0,
                   f"飞行+投送共 {ver['total_s']:.0f} s：直连 {ver['direct_s']:.0f} s，"
                   f"中继 {ver['relay_s']:.0f} s，中断 {ver['outage_s']:.0f} s"))
    return pd.DataFrame(checks, columns=["检验内容", "是否通过", "说明"])


def draw_map(info, fname):
    S = info["S"]
    lons = [NODES[k]["lon"] for k in NODES] + [s["pos"][0] for s in S]
    lats = [NODES[k]["lat"] for k in NODES] + [s["pos"][1] for s in S]
    x0, x1 = min(lons) - 0.015, max(lons) + 0.015
    y0, y1 = min(lats) - 0.015, max(lats) + 0.015
    c0, c1 = int((x0 - DEM_X0) / DEM_DX), int((x1 - DEM_X0) / DEM_DX)
    r0, r1 = int((DEM_Y0 - y1) / DEM_DY), int((DEM_Y0 - y0) / DEM_DY)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(DEM_Z[r0:r1, c0:c1], extent=[x0, x1, y0, y1], cmap="terrain", origin="upper")
    plt.colorbar(im, ax=ax, label="地面高程 (m)", shrink=0.7)
    by_uid = {s["uid"]: s for s in info["state"]}
    for p in info["plan"]:
        T, gid, _ = trip_timeline(p["ev"])
        kf = trip_track(p["ev"])
        pts = ["O01"] + [s for s, _ in p["ev"]["stops"]] + ["O01"]
        ax.plot([NODES[k]["lon"] for k in pts], [NODES[k]["lat"] for k in pts], "-", color="0.25", lw=0.8, alpha=0.6)
    colors = plt.get_cmap("Set1")
    for k, s in enumerate(S):
        used = [x for x in info["state"] if x["k"] == k]
        ax.plot(s["pos"][0], s["pos"][1], "^", color=colors(k), ms=15, mec="k")
        ax.text(s["pos"][0] + 0.002, s["pos"][1] - 0.004,
                f"P{k + 1}（{len(used)}个中继架次）", color=colors(k), fontsize=10, weight="bold")
    # 盲区采样点：按保障点位着色
    for p in info["plan"]:
        T, gid, _ = trip_timeline(p["ev"])
        for k, a, b, i, j, uid in p["assign"]:
            sel = np.arange(i, j + 1)
            sel = sel[gid[sel] >= 0]
            pos = GEO["need_pos"][gid[sel]]
            ax.scatter(pos[:, 0], pos[:, 1], s=2, color=colors(k), alpha=0.5)
    for k, n in NODES.items():
        ax.plot(n["lon"], n["lat"], "k*" if k == "O01" else "ro", ms=14 if k == "O01" else 6)
        ax.text(n["lon"] + 0.002, n["lat"] + 0.002, k, fontsize=9, weight="bold")
    ax.set_xlabel("经度 (°)"); ax.set_ylabel("纬度 (°)")
    ax.set_title("问题三 运输路线、直连盲区（彩色点=由对应中继点位保障）与中继悬停点")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, fname), dpi=150); plt.close()


def draw_gantt(info, fname):
    plan, state = info["plan"], info["state"]
    rows = [u for u, _ in DRONES] + [f"{g}-B{k + 1:02d}" for g, inf in BATTERIES.items() for k in range(inf["count"])]
    rows += list(RELAY_IDS) + MODULE_IDS
    fig, ax = plt.subplots(figsize=(16, 0.36 * len(rows) + 1.8))
    color = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
    for p in plan:
        g = p["ev"]["g"]
        y = rows.index(p["uav"])
        ax.barh(y, p["end"] - p["start"], left=p["start"], color=color[g], edgecolor="k", height=0.6)
        ax.text(p["start"] + 20, y, p["tid"][3:], va="center", fontsize=6, color="w")
        for k, a, b, i, j, uid in p["assign"]:
            ax.barh(y + 0.33, b - a, left=p["start"] + a, color="m", height=0.12)
        y = rows.index(p["batt"])
        ax.barh(y, p["end"] - p["start"], left=p["start"], color=color[g], edgecolor="k", height=0.6)
        ax.barh(y, p["charge_end"] - p["end"], left=p["end"], color=color[g], alpha=0.3, height=0.6)
    cm = plt.get_cmap("Set1")
    for s in state:
        l, r, e, soc, c = s["_d"]
        y = rows.index(s["r"])
        ax.barh(y, r - l, left=l, color=cm(s["k"]), alpha=0.45, edgecolor="k", height=0.6)
        ax.barh(y, s["we"] - s["ws"], left=s["ws"], color=cm(s["k"]), edgecolor="k", height=0.6)
        ax.text(l + 20, y, f"{s['rid'][3:]}@P{s['k'] + 1}", va="center", fontsize=6)
        y = rows.index(s["m"])
        ax.barh(y, r - l, left=l, color=cm(s["k"]), edgecolor="k", height=0.6)
        ax.barh(y, c - r, left=r, color=cm(s["k"]), alpha=0.3, height=0.6)
    for d in (3600, 7200, 10800):
        ax.axvline(d, color="r", ls="--", lw=0.8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows); ax.invert_yaxis()
    ax.set_xlabel("时间 (s)（深色=任务/服务窗口，浅色=往返飞行或充电；紫色细条=运输机依托中继的时段）")
    ax.set_title("问题三 运输无人机、电池、中继无人机与能源组件联合甘特图")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, fname), dpi=150); plt.close()


def export(info, sol):
    """把最终联合方案整理成与求解过程无关的纯数据（问题四直接读取，不需要重新求解）"""
    trips = []
    for p in info["plan"]:
        ev = p["ev"]
        trips.append({"tid": p["tid"], "g": ev["g"], "stops": [(s, list(ids)) for s, ids in ev["stops"]],
                      "areas": [s for s, _ in ev["stops"]], "uav": p["uav"], "batt": p["batt"],
                      "start": float(p["start"]), "end": float(p["end"]), "charge_end": float(p["charge_end"]),
                      "E": float(ev["E"]), "soc": float(ev["soc"]), "n_box": len(ev["ids"]),
                      "deliver": {i: float(p["start"] + o) for i, o in ev["deliver"].items()},
                      "comm": [(float(a), float(b), lab, m) for a, b, lab, m in p["comm_rows"]]})
    uid2rid = {s["uid"]: s["rid"] for s in info["state"]}
    for t in trips:
        t["comm"] = [(a, b, lab, uid2rid[m] if m >= 0 else ("直连" if m == -1 else "中断")) for a, b, lab, m in t["comm"]]
    relays = []
    for s in sorted(info["state"], key=lambda s: s["rid"]):
        l, r, e, soc, c = s["_d"]
        relays.append({"rid": s["rid"], "relay": s["r"], "module": s["m"], "site": f"P{s['k'] + 1}",
                       "launch": float(l), "ws": float(s["ws"]), "we": float(s["we"]), "ret": float(r),
                       "E": float(e), "soc": float(soc), "charge_end": float(c)})
    sites = [{"name": f"P{k + 1}", **{kk: s[kk] for kk in ("pos", "lead", "t_out", "t_back", "e_fix", "e_out", "e_back")}}
             for k, s in enumerate(info["S"])]
    return {"sol": sol, "trips": trips, "relays": relays, "sites": sites,
            "metrics": {k: info[k] for k in ("cost", "makespan", "transport_makespan", "relay_makespan",
                                             "energy_t", "energy_r", "trips", "relay_sorties", "on_time")}}


def metrics(name, info):
    return {"方案": name, "运输架次": info["trips"], "中继架次": info["relay_sorties"],
            "运输能耗(kWh)": round(info["energy_t"], 3), "中继能耗(kWh)": round(info["energy_r"], 3),
            "总能耗(kWh)": round(info["energy_t"] + info["energy_r"], 3),
            "运输完成(s)": round(info["transport_makespan"], 1), "联合完成(s)": round(info["makespan"], 1),
            "加权延误(系数·min)": round(info["weighted_tardiness"], 2), "准时箱数(/80)": info["on_time"],
            "硬时限违反(s)": round(info["hard_violation_s"], 1), "目标值": round(info["cost"], 2)}


# ===============================================================
# 主程序
# ===============================================================
if __name__ == "__main__":
    t_all = time.time()
    print("===== 问题三：通信约束下的运输与中继联合调度 =====")
    with open(os.path.join(RESULT_DIR, "q2_best.pkl"), "rb") as f:
        q2_sol = pickle.load(f)
    print(f"问题二最优运输方案：{len(q2_sol)} 个架次")

    # ---- 通信盲区统计 ----
    n_blind = 0
    for tr in q2_sol:
        ev = vrp.eval_trip(tr["g"], tr["stops"])
        _, gid, _ = trip_timeline(ev)
        n_blind += (gid >= 0).any()
    print(f"问题二方案中含直连盲区的架次：{n_blind}/{len(q2_sol)}（不加中继将无法执行）")
    rows_dir = []
    for s in AREAS:
        rows_dir.append({"服务区": s, "投送点直连": "可用" if HOVS[s]["direct"] else "不可用",
                         "往返航段盲区采样点": int(len(LEGS[("O01", s)]["need_global"]) + len(LEGS[(s, "O01")]["need_global"])),
                         "单点可完整保障的候选中继点数": int(COV[:, np.concatenate(
                             [LEGS[("O01", s)]["need_global"], HOVS[s]["need_global"], LEGS[(s, "O01")]["need_global"]]
                         ).astype(int)].all(axis=1).sum()) if not (HOVS[s]["direct"] and len(LEGS[("O01", s)]["need_global"]) == 0) else "-"})
    df_dir = pd.DataFrame(rows_dir)
    print(df_dir.to_string(index=False))

    # ---- 候选点池：按对往返盲区的覆盖量排序 ----
    rt_ids = round_trip_ids()
    score = COV[:, rt_ids].sum(axis=1)
    TOP_POOL = np.argsort(-score)[:300]

    # ---- 点位下界分析：最小集合覆盖（整数规划） ----
    all_ids = np.arange(COV.shape[1])
    cover_rt = exact_cover(rt_ids)
    cover_all = exact_cover(all_ids)
    print(f"\n覆盖全部“单点往返”盲区的最少点位数：{len(cover_rt)}  {cover_rt}")
    print(f"覆盖全部 240 条航段 + 投送点盲区的最少点位数：{len(cover_all)}  {cover_all}")
    rt_area = {s: np.concatenate([LEGS[("O01", s)]["need_global"], HOVS[s]["need_global"],
                                  LEGS[(s, "O01")]["need_global"]]).astype(int) for s in AREAS}
    rt_area = {s: v for s, v in rt_area.items() if len(v)}
    single = np.stack([COV[:, v].all(axis=1) for v in rt_area.values()], axis=1)
    pair_cnt = np.zeros((len(CAND), len(CAND)), int)
    for v in rt_area.values():
        U = (~COV[:, v]).astype(np.float32)
        pair_cnt += (U @ U.T) == 0                     # 两点的并集覆盖该服务区往返全部盲区点
    print(f"单个点位最多可完整保障 {single.sum(axis=1).max()} 个服务区的往返；"
          f"任意两点位（允许切换）最多 {pair_cnt.max()} 个，需中继的服务区共 {len(rt_area)} 个")
    del pair_cnt

    # ---- 不同点位数 K 的比较：精确覆盖初始点位 + 多种子联合退火 + 爬山 + 点位精细化 ----
    results = {}
    for K in (3, 4):
        sites0 = cover_all
        if K == 4:
            extra = [k for k in np.argsort(-single.sum(axis=1)) if k not in sites0][0]
            sites0 = tuple(sites0) + (int(extra),)
        c0 = evaluate(q2_sol, sites0)
        print(f"\nK={K} 初始点位 {sites0}，问题二路线直接做中继调度的目标 {c0:.2f}")
        best = None
        for sd in (11, 12, 13):
            sol, sites, c = joint_anneal(q2_sol, sites0, iters=30000, seed=sd + K)
            sol, sites, c = polish(sol, sites)
            print(f"   K={K} 种子 {sd}: 目标 {c:.2f}")
            if best is None or c < best[2]:
                best = (sol, sites, c)
        sol, sites, c = best
        sites, c = refine_sites(sol, sites)
        sol, sites, c = polish(sol, sites)
        print(f"   K={K} 精细化 + 修正后目标 {c:.2f}")
        results[K] = (sol, sites, c)
    K_best = min(results, key=lambda k: results[k][2])
    sol, sites, c = results[K_best]
    print(f"\n最优点位数 K={K_best}，目标 {c:.2f}")

    # ---- 权衡分析：改变权重重新优化；各情景解再按主权重“接力”修正，取全局最好者为最终方案 ----
    print("\n===== 基线与权衡分析 =====")
    rows = []
    c_b, inf_b = evaluate(q2_sol, cover_all, detail=True)
    if inf_b is not None:
        rows.append(metrics("基线：问题二运输方案+精确覆盖点位（仅做中继调度）", inf_b))
    for K in sorted(results):
        _, inf_k = evaluate(results[K][0], results[K][1], detail=True)
        rows.append(metrics(f"K={K}个中继点位（联合优化）", inf_k))
    scenarios = [("时效优先", {"tard": 6.0, "T": 8.0, "E": 1.0, "ER": 1.0, "N": 1.0, "NR": 1.0}),
                 ("能耗优先", {"tard": 0.3, "T": 0.3, "E": 40.0, "ER": 40.0}),
                 ("架次优先", {"tard": 0.3, "T": 0.3, "E": 2.0, "ER": 2.0, "N": 40.0, "NR": 40.0})]
    pool = [(sol, sites)]
    for name, w in scenarios:
        ww = dict(W3); ww.update(w)
        s2, st2, _ = joint_anneal(sol, sites, iters=12000, seed=77, W=ww, verbose=False)
        s2, st2, _ = polish(s2, st2, W=ww, rounds=2, tries=3000)
        _, inf2 = evaluate(s2, st2, W=ww, detail=True)
        inf2["cost"] = evaluate(s2, st2)                # 统一按主权重计算目标值，便于横向比较
        rows.append(metrics(name, inf2))
        pool.append((s2, st2))
        print("  ", rows[-1])
    for s2, st2 in pool[1:]:
        s3, st3, c3 = polish(s2, st2, rounds=3, tries=3000)
        if c3 < c - 1e-9:
            sol, sites, c = s3, st3, c3
            print(f"   情景解接力修正得到更优主方案：目标 {c:.2f}")

    # ---- 闭环修正：逐秒复核 → 中断处局部加密采样 → 重新调度 + 爬山，直到全程无中断 ----
    print("\n===== 逐秒复核闭环修正 =====")
    for rnd in range(1, 9):
        c, info = evaluate(sol, sites, detail=True)
        ver = verify(info)
        print(f"   第 {rnd} 轮：目标 {c:.2f}，通信中断 {ver['outage_s']:.0f} s")
        if ver["outage_s"] <= 1e-9:
            break
        n_need, n_add = augment(info, ver)
        print(f"      局部加密采样 {n_add} 个，其中新增直连盲区点 {n_need} 个")
        if evaluate(sol, sites) == float("inf"):
            sites, _ = refine_sites(sol, sites, verbose=False)
        sol, sites, c = polish(sol, sites, rounds=2, tries=2000)
    c, info = evaluate(sol, sites, detail=True)
    rows.append(metrics("最终主方案（均衡权重，闭环修正后）", info))
    df_trade = pd.DataFrame(rows)
    print(df_trade.to_string(index=False))

    df_relay = relay_rows(info)
    ver = verify(info)
    print(f"最终逐秒复核：中断 {ver['outage_s']:.0f} s")
    df_trip, df_box, df_comm = transport_rows(info)
    df_res = resource_rows(info)
    df_chk = check_rows(info, ver)
    print(df_trip.to_string(index=False))
    print(df_relay.to_string(index=False))
    print(df_chk.to_string(index=False))
    df_site = pd.DataFrame([{"点位": f"P{k + 1}", "经度(°)": round(s["pos"][0], 6), "纬度(°)": round(s["pos"][1], 6),
                             "悬停海拔(m)": round(s["pos"][2], 1), "地面高程(m)": round(float(dem_at(*s["pos"][:2])), 1),
                             "距O01水平距离(m)": round(hdist(LON0, LAT0, s["pos"][0], s["pos"][1]), 0),
                             "单程飞行时间(s)": round(s["t_out"], 1), "往返飞行能耗(kWh)": round(s["e_out"] + s["e_back"], 4),
                             "单架次最长服务(s)": round(s["max_serve"], 0),
                             "可完整保障往返的服务区": ",".join(a for a, v in rt_area.items() if COV[s["ci"], v].all()),
                             "使用中继架次": ",".join(x["rid"] for x in info["state"] if x["k"] == k)}
                            for k, s in enumerate(info["S"])])
    print(df_site.to_string(index=False))

    with pd.ExcelWriter(os.path.join(RESULT_DIR, "Q3_结果.xlsx")) as w:
        df_relay.to_excel(w, sheet_name="Q3_中继架次", index=False)
        df_comm.to_excel(w, sheet_name="Q3_通信保障", index=False)
        df_trip.to_excel(w, sheet_name="Q3_运输架次", index=False)
        df_box.to_excel(w, sheet_name="Q3_逐箱交付", index=False)
        df_res.to_excel(w, sheet_name="资源使用", index=False)
        df_chk.to_excel(w, sheet_name="可行性检验", index=False)
        df_trade.to_excel(w, sheet_name="权衡分析", index=False)
        df_dir.to_excel(w, sheet_name="直连盲区统计", index=False)
        df_site.to_excel(w, sheet_name="中继点位", index=False)
    with open(os.path.join(RESULT_DIR, "q3_best.pkl"), "wb") as f:
        pickle.dump(export(info, sol), f)
    draw_map(info, "Q3_运输路线与中继点位.png")
    draw_gantt(info, "Q3_联合资源甘特图.png")
    print(f"\n问题三完成，总用时 {time.time() - t_all:.0f}s。")
