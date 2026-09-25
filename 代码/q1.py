# -*- coding: utf-8 -*-
"""
q1.py —— 问题一：单点往返运输能力与货箱组批
==========================================
思路：
 (1) 最大安全载荷：往返能耗 E(q) = 去程(带载 q) + 返程(空载)，随 q 单调增加，
     用“二分法”找到满足 E(q) ≤ (1-ρ)·E_use 的最大 q，再与最大载货质量 Q 取小。
 (2) 组批：同一服务区内同一类物资的货箱完全相同（质量、体积一样），
     所以“还剩几箱医疗/饮用水/食品/卫生用品”就能完整描述剩余任务。
     把它当作动态规划(DP)的“状态”，每一步选一个可行批次（机型 + 各类箱数），
     状态总数最多 3×9×4×3 = 324 个，DP 可以求出“精确最优解”。
 (3) 多目标：架次数、总能耗、累计作业时间。用“字典序”比较（先比架次，再比能耗，再比时间），
     同时给出“能耗优先”“时间优先”两种字典序结果，以及贪心基线，对比说明权衡关系。
 (4) 灵敏度：改变返航安全余量 ρ，重新计算最大载荷和组批结果。
"""
import itertools
from functools import lru_cache
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # 不弹窗口，直接保存图片
import matplotlib.pyplot as plt
from common import *

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]   # 图中显示中文
plt.rcParams["axes.unicode_minus"] = False

KINDS = ["MED", "WAT", "FOD", "HYG"]


# ---------------------------------------------------------------
# 1. 单点往返的能耗、时间、最大安全载荷
# ---------------------------------------------------------------
def round_trip_energy(g, s, q):
    """机型 g 给服务区 s 送载荷 q 的往返能耗：去程带 q，返程空载"""
    return seg_energy(g, node_segment("O01", s), q) + seg_energy(g, node_segment(s, "O01"), 0.0)


def round_trip_time(g, s, n_box):
    """往返作业时间 = 准备 + 装载 + 去程飞行 + 交接 + 返程飞行"""
    t = TYPES[g]
    return (t["prep"] + t["load"] * n_box + seg_time(g, node_segment("O01", s))
            + t["hand_base"] + t["hand_box"] * n_box + seg_time(g, node_segment(s, "O01")))


def max_safe_payload(g, s, rho):
    """二分法求最大安全载荷"""
    limit = (1 - rho) * TYPES[g]["E"]                 # 允许消耗的最大能量
    Q = TYPES[g]["Q"]
    if round_trip_energy(g, s, 0.0) > limit:          # 空载都飞不回来
        return 0.0
    if round_trip_energy(g, s, Q) <= limit:           # 满载也够，最大载荷就是 Q
        return Q
    lo, hi = 0.0, Q
    for _ in range(60):                               # 二分 60 次，精度远小于 0.001 kg
        mid = (lo + hi) / 2
        if round_trip_energy(g, s, mid) <= limit:
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------
# 2. 某服务区的货箱信息（按类别统计箱数、单箱质量和体积）
# ---------------------------------------------------------------
def area_kinds(s):
    info = {}
    for b in BOXES.values():
        if b["area"] == s:
            k = b["kind"]
            if k not in info:
                info[k] = {"n": 0, "mass": b["mass"], "vol": b["vol"], "ids": []}
            info[k]["n"] += 1
            info[k]["ids"].append(b["id"])
    return info


# ---------------------------------------------------------------
# 3. 枚举所有“可行批次”
# ---------------------------------------------------------------
def feasible_batches(s, rho, types_allowed=("A", "B", "C")):
    info = area_kinds(s)
    kinds = [k for k in KINDS if k in info]
    batches = []
    for g in types_allowed:
        qmax = max_safe_payload(g, s, rho)
        # itertools.product 枚举每类取 0..n 箱的所有组合
        for combo in itertools.product(*[range(info[k]["n"] + 1) for k in kinds]):
            n_box = sum(combo)
            if n_box == 0:
                continue
            mass = sum(c * info[k]["mass"] for c, k in zip(combo, kinds))
            vol = sum(c * info[k]["vol"] for c, k in zip(combo, kinds))
            if mass > qmax + 1e-9 or vol > TYPES[g]["V"] + 1e-9:
                continue                               # 超重、超体积或能量不够 → 不可行
            e = round_trip_energy(g, s, mass)
            batches.append({"g": g, "combo": combo, "n": n_box, "mass": mass, "vol": vol,
                            "E": e, "T": round_trip_time(g, s, n_box),
                            "soc": 1 - e / TYPES[g]["E"]})
    return kinds, info, batches


# ---------------------------------------------------------------
# 4. 动态规划求最优组批
# ---------------------------------------------------------------
def optimize_area(s, rho, order="NET", types_allowed=("A", "B", "C")):
    """
    order 决定字典序：'NET' = 架次→能耗→时间，'ENT' = 能耗→架次→时间，'TNE' = 时间→架次→能耗
    返回：批次列表（None 表示无可行方案）
    """
    kinds, info, batches = feasible_batches(s, rho, types_allowed)
    start = tuple(info[k]["n"] for k in kinds)        # 初始状态：全部箱子都没送

    def key(n, e, t):                                 # 把三个指标按指定顺序排成元组
        d = {"N": n, "E": round(e, 9), "T": round(t, 6)}
        return tuple(d[c] for c in order)

    @lru_cache(maxsize=None)                          # 记忆化：同一状态只算一次
    def solve(state):
        if sum(state) == 0:
            return (key(0, 0, 0), 0, 0.0, 0.0, ())
        best = None
        for idx, bt in enumerate(batches):
            if any(c > r for c, r in zip(bt["combo"], state)):
                continue                               # 这批要的箱子比剩下的还多，跳过
            nxt = tuple(r - c for c, r in zip(bt["combo"], state))
            sub = solve(nxt)
            if sub is None:
                continue
            n, e, t = sub[1] + 1, sub[2] + bt["E"], sub[3] + bt["T"]
            cand = (key(n, e, t), n, e, t, (idx,) + sub[4])
            if best is None or cand[0] < best[0]:
                best = cand
        return best

    res = solve(start)
    if res is None:
        return None
    return [dict(batches[i], kinds=kinds) for i in res[4]], info


def greedy_area(s, rho):
    """
    基线方法（用来对比）：首次适应递减（FFD）+ 固定用该服务区“单箱能耗最低”的机型。
    即：先装重箱子，装不下就开新架次。
    """
    best_plan = None
    for g in ("A", "B", "C"):
        qmax = max_safe_payload(g, s, rho)
        items = sorted([b for b in BOXES.values() if b["area"] == s], key=lambda b: -b["mass"])
        bins = []
        ok = True
        for b in items:
            if b["mass"] > qmax or b["vol"] > TYPES[g]["V"]:
                ok = False
                break
            for bn in bins:
                if bn["mass"] + b["mass"] <= qmax and bn["vol"] + b["vol"] <= TYPES[g]["V"]:
                    bn["mass"] += b["mass"]; bn["vol"] += b["vol"]; bn["n"] += 1
                    break
            else:
                bins.append({"mass": b["mass"], "vol": b["vol"], "n": 1})
        if not ok:
            continue
        n = len(bins)
        e = sum(round_trip_energy(g, s, bn["mass"]) for bn in bins)
        t = sum(round_trip_time(g, s, bn["n"]) for bn in bins)
        if best_plan is None or e < best_plan[1]:
            best_plan = (n, e, t)
    return best_plan


def summarize(rho, order="NET"):
    """对 15 个服务区全部求解，返回 (总架次, 总能耗, 总时间, 每个服务区方案)"""
    total_n, total_e, total_t, plans = 0, 0.0, 0.0, {}
    for s in AREAS:
        out = optimize_area(s, rho, order)
        if out is None:
            return None
        plan, info = out
        plans[s] = (plan, info)
        total_n += len(plan)
        total_e += sum(p["E"] for p in plan)
        total_t += sum(p["T"] for p in plan)
    return total_n, total_e, total_t, plans


# ---------------------------------------------------------------
# 5. 主程序
# ---------------------------------------------------------------
if __name__ == "__main__":
    RHO = TYPES["A"]["rho"]                            # 三种机型返航下限都是 20%

    # ---- (1) 最大安全载荷表 ----
    rows = []
    for s in AREAS:
        row = {"服务区": s}
        for g in "ABC":
            row[f"{g}型最大安全载荷(kg)"] = round(max_safe_payload(g, s, RHO), 3)
            row[f"{g}型满载往返SOC(%)"] = round(
                100 * (1 - round_trip_energy(g, s, TYPES[g]["Q"]) / TYPES[g]["E"]), 2)
        rows.append(row)
    df_cap = pd.DataFrame(rows)
    print("\n===== 最大安全载荷（ρ=20%） =====")
    print(df_cap.to_string(index=False))

    # ---- (2)(3) 组批优化：三种字典序 + 贪心基线 ----
    compare = []
    for order, name in [("NET", "架次优先(主方案)"), ("ENT", "能耗优先"), ("TNE", "时间优先")]:
        n, e, t, plans = summarize(RHO, order)
        compare.append({"方案": name, "总架次": n, "总能耗(kWh)": round(e, 4), "累计作业时间(s)": round(t, 1)})
        if order == "NET":
            main_plans = plans
    g_n = g_e = g_t = 0
    for s in AREAS:
        n, e, t = greedy_area(s, RHO)
        g_n += n; g_e += e; g_t += t
    compare.append({"方案": "贪心FFD基线", "总架次": g_n, "总能耗(kWh)": round(g_e, 4), "累计作业时间(s)": round(g_t, 1)})
    df_cmp = pd.DataFrame(compare)
    print("\n===== 组批方案比较 =====")
    print(df_cmp.to_string(index=False))

    # ---- 把主方案展开成逐架次表（货箱编号：首批/医疗箱优先放进前面的架次）----
    trip_rows = []
    tid = 0
    for s in AREAS:
        plan, info = main_plans[s]
        # 让含医疗和饮用水多的架次排前面（首批保障箱也就先出发）
        plan = sorted(plan, key=lambda p: (-p["combo"][0], -p["mass"]))
        pointer = {k: 0 for k in info}                 # 每类箱子已分配到第几个
        for p in plan:
            tid += 1
            ids = []
            for c, k in zip(p["combo"], p["kinds"]):
                ids += info[k]["ids"][pointer[k]: pointer[k] + c]
                pointer[k] += c
            trip_rows.append({"架次编号": f"Q1-T{tid:02d}", "服务区编号": s, "机型编号": p["g"],
                              "货箱编号列表": ",".join(ids), "总质量（kg）": round(p["mass"], 3),
                              "总体积（m³）": round(p["vol"], 4), "往返时间（s）": round(p["T"], 1),
                              "架次能耗（kWh）": round(p["E"], 4), "返航SOC（%）": round(100 * p["soc"], 2)})
    df_q1 = pd.DataFrame(trip_rows)
    print("\n===== 主方案逐架次组批 =====")
    print(df_q1.to_string(index=False))

    # ---- (4) 返航安全余量灵敏度 ----
    sens = []
    rhos = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    for r in rhos:
        row = {"ρ": r}
        for g in "ABC":
            caps = [max_safe_payload(g, s, r) for s in AREAS]
            row[f"{g}最小载荷"] = round(min(caps), 2)
            row[f"{g}平均载荷"] = round(sum(caps) / len(caps), 2)
            row[f"{g}满载可达服务区数"] = sum(1 for c in caps if c >= TYPES[g]["Q"] - 1e-6)
        res = summarize(r, "NET")
        if res is None:
            row.update({"总架次": "不可行", "总能耗": "-", "C型架次": "-"})
        else:
            row.update({"总架次": res[0], "总能耗": round(res[1], 3),
                        "C型架次": sum(1 for s in AREAS for p in res[3][s][0] if p["g"] == "C"),
                        "A型架次": sum(1 for s in AREAS for p in res[3][s][0] if p["g"] == "A"),
                        "B型架次": sum(1 for s in AREAS for p in res[3][s][0] if p["g"] == "B")})
        sens.append(row)
    df_sens = pd.DataFrame(sens)
    print("\n===== 返航安全余量灵敏度 =====")
    print(df_sens.to_string(index=False))

    # ---- 保存结果 ----
    with pd.ExcelWriter(os.path.join(RESULT_DIR, "Q1_结果.xlsx")) as w:
        df_q1.to_excel(w, sheet_name="Q1_单点组批", index=False)
        df_cap.to_excel(w, sheet_name="最大安全载荷", index=False)
        df_cmp.to_excel(w, sheet_name="方案比较", index=False)
        df_sens.to_excel(w, sheet_name="余量灵敏度", index=False)
    df_q1.to_pickle(os.path.join(RESULT_DIR, "q1_trips.pkl"))

    # ---- 画图：最大安全载荷柱状图 + 灵敏度曲线 ----
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    x = range(len(AREAS))
    for i, g in enumerate("ABC"):
        ax[0].bar([v + (i - 1) * 0.27 for v in x], df_cap[f"{g}型最大安全载荷(kg)"], width=0.27, label=f"{g}型")
    ax[0].set_xticks(list(x)); ax[0].set_xticklabels(AREAS, rotation=60)
    ax[0].set_ylabel("最大安全载荷 (kg)"); ax[0].set_title("ρ=20% 时各服务区最大安全载荷"); ax[0].legend()
    for g in "ABC":
        ys = [[max_safe_payload(g, s, r) for s in AREAS] for r in rhos]
        ax[1].plot(rhos, [min(y) for y in ys], "o-", label=f"{g}型 最小值")
        ax[1].plot(rhos, [sum(y) / len(y) for y in ys], "s--", label=f"{g}型 平均值")
    ax[1].set_xlabel("返航安全余量 ρ"); ax[1].set_ylabel("最大安全载荷 (kg)")
    ax[1].set_title("返航安全余量对最大安全载荷的影响"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "Q1_最大安全载荷与灵敏度.png"), dpi=150)
    print("\n问题一完成，结果已保存。")
