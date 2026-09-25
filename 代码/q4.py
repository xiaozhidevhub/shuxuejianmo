# -*- coding: utf-8 -*-
"""
q4.py —— 问题四：救援任务分区与资源配置优化
=============================================
【输入】问题三的联合调度方案（结果/q3_best.pkl）：运输架次（组批、访问顺序、时刻、机型）、
        中继架次（点位、时段、实体、组件）以及逐段“运输架次 ↔ 中继架次”的通信保障关系。
【规则】分区时上述安排全部保持不变；同一运输架次访问的多个服务区必须同组；
        各组只承担本组服务区的运输任务及为这些任务提供保障的中继任务，资源不得跨组调配。
【建模】
 1. 服务区“块”：以多点架次为边做并查集，连通分量为不可拆分的块 → 分区 = 块的集合划分。
 2. 组资源需求：任务时刻固定时，某类资源的最少数量 = 该组任务占用区间的最大重叠数
    （区间图着色数 = 最大团，按开始时刻贪心分配即可达到，属于精确值）：
      运输无人机 [开始, 返回)、共享电池 [开始, 充满)、中继无人机 [起飞, 返回+周转300s)、能源组件 [起飞, 充满)。
 3. 中继任务归属：若某中继架次同时保障了两个组的运输架次，则两组各自需要一份该中继任务
    （口径A：原中继架次整体复制，时段完全不变——主口径；
      口径B：各组副本的服务窗口收缩到本组实际使用区间，点位与保障关系不变——用于分析冗余来源）。
 4. 评价指标：资源配置规模（各组资源之和）、资源冗余（相对不分区的增加量）、
    组间工作量均衡（各组无人机作业时长的变异系数 CV、最大/最小比）、与库存的资源缺口。
 5. 求解：块数不多，对 2 组、3 组的全部划分（第二类 Stirling 数）穷举 → 全局最优，无启发式误差。
    多目标处理用 ε-约束法：先要求组间工作量 CV ≤ ε（推荐 ε=0.2，即最大/最小约 1.4 倍以内，
    保证各组是“规模相当的独立执行单元”），再按“资源缺口 → 资源规模 → CV”字典序取最优；
    同时给出两个极端（只省资源 / 只求均衡）与完整的 ε-权衡曲线、帕累托前沿。
"""
import os
import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False

with open(os.path.join(RESULT_DIR, "q3_best.pkl"), "rb") as f:
    Q3 = pickle.load(f)
TRIPS = Q3["trips"]
RELAYS = {r["rid"]: r for r in Q3["relays"]}
SITES = {s["name"]: s for s in Q3["sites"]}
P_SERVE = RELAY["P_hover"] + RELAY["P_comm"]

STOCK = {"U_A": 0, "U_B": 0, "U_C": 0}
for _, g in DRONES:
    STOCK[f"U_{g}"] += 1
STOCK.update({f"B_{g}": BATTERIES[g]["count"] for g in "ABC"})
STOCK.update({"R": len(RELAY_IDS), "M": MODULES["count"]})
RES_KEYS = ["U_A", "U_B", "U_C", "B_A", "B_B", "B_C", "R", "M"]
RES_NAME = {"U_A": "A型运输无人机", "U_B": "B型运输无人机", "U_C": "C型运输无人机",
            "B_A": "A型电池组", "B_B": "B型电池组", "B_C": "C型电池组", "R": "中继无人机", "M": "中继能源组件"}

# 每个运输架次使用了哪些中继架次、各自的使用区间
USAGE = {}
for t in TRIPS:
    u = {}
    for a, b, lab, m in t["comm"]:
        if m not in ("直连", "中断"):
            lo, hi = u.get(m, (a, b))
            u[m] = (min(lo, a), max(hi, b))
    USAGE[t["tid"]] = u


# ---------------------------------------------------------------
# 1. 服务区块（同一架次的服务区必须同组）
# ---------------------------------------------------------------
def area_blocks():
    parent = {s: s for s in AREAS}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for t in TRIPS:
        for a, b in zip(t["areas"], t["areas"][1:]):
            parent[find(a)] = find(b)
    blocks = {}
    for s in AREAS:
        blocks.setdefault(find(s), []).append(s)
    return sorted((sorted(v) for v in blocks.values()), key=lambda v: v[0])


BLOCKS = area_blocks()
AREA_BLOCK = {s: k for k, blk in enumerate(BLOCKS) for s in blk}
TRIP_BLOCK = {t["tid"]: AREA_BLOCK[t["areas"][0]] for t in TRIPS}


# ---------------------------------------------------------------
# 2. 资源需求：区间最大重叠数（并给出一个达到该下界的具体分配）
# ---------------------------------------------------------------
def max_overlap(intervals):
    ev = []
    for a, b in intervals:
        ev.append((a, 1)); ev.append((b, -1))
    ev.sort(key=lambda x: (x[0], x[1]))              # 同一时刻先释放再占用
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return best


def interval_assign(items):
    """items = [(名称, 开始, 结束)]，按开始时刻贪心分配到最早空闲的资源，返回 {名称: 资源序号}"""
    free = []
    out = {}
    for name, a, b in sorted(items, key=lambda x: (x[1], x[2])):
        k = next((i for i, f in enumerate(free) if f <= a + 1e-6), None)
        if k is None:
            free.append(b); k = len(free) - 1
        else:
            free[k] = b
        out[name] = k + 1
    return out


def relay_copy(rid, tids, mode):
    """某组对中继架次 rid 的需求副本；mode='A' 整体复制，'B' 收缩到本组使用区间"""
    r = RELAYS[rid]
    if mode == "A":
        return dict(r)
    lo = min(USAGE[t][rid][0] for t in tids)
    hi = max(USAGE[t][rid][1] for t in tids)
    ws, we = max(r["ws"], lo), min(r["we"], hi)
    si = SITES[r["site"]]
    e = si["e_fix"] + P_SERVE * (we - ws) / 3600.0
    soc = 1 - e / RELAY["E"]
    ret = we + si["t_back"]
    return {**r, "ws": ws, "we": we, "launch": ws - si["lead"], "ret": ret, "E": e, "soc": soc,
            "charge_end": ret + charge_time(soc, MODULES["full"])}


def group_detail(trips, mode="A"):
    """一组任务（运输架次列表）的资源需求与工作量"""
    need = {}
    for g in "ABC":
        ts = [t for t in trips if t["g"] == g]
        need[f"U_{g}"] = max_overlap([(t["start"], t["end"]) for t in ts])
        need[f"B_{g}"] = max_overlap([(t["start"], t["charge_end"]) for t in ts])
    serve = {}
    for t in trips:
        for rid in USAGE[t["tid"]]:
            serve.setdefault(rid, []).append(t["tid"])
    rel = [relay_copy(rid, tids, mode) for rid, tids in sorted(serve.items())]
    need["R"] = max_overlap([(r["launch"], r["ret"] + RELAY["turn"]) for r in rel])
    need["M"] = max_overlap([(r["launch"], r["charge_end"]) for r in rel])
    work_t = sum(t["end"] - t["start"] for t in trips)
    work_r = sum(r["ret"] - r["launch"] for r in rel)
    return {"need": need, "relays": rel, "serve": serve, "n_trips": len(trips), "n_relay": len(rel),
            "E_t": sum(t["E"] for t in trips), "E_r": sum(r["E"] for r in rel),
            "work_t": work_t, "work_r": work_r, "work": work_t + work_r,
            "makespan": max([t["end"] for t in trips] + [r["ret"] for r in rel]),
            "n_box": sum(t["n_box"] for t in trips)}


def trips_of(blocks_in_group):
    return [t for t in TRIPS if TRIP_BLOCK[t["tid"]] in blocks_in_group]


BASE = group_detail(TRIPS, "A")                      # 不分区时（问题三原方案）的资源需求


def evaluate_partition(labels, K, mode="A"):
    """labels[k] = 第 k 个块所属任务组（0..K-1）"""
    groups = []
    for g in range(K):
        blks = {k for k, l in enumerate(labels) if l == g}
        groups.append({"blocks": blks, "areas": sorted(s for k in blks for s in BLOCKS[k]),
                       **group_detail(trips_of(blks), mode)})
    total = {k: sum(gr["need"][k] for gr in groups) for k in RES_KEYS}
    gap = {k: max(0, total[k] - STOCK[k]) for k in RES_KEYS}
    redund = {k: total[k] - BASE["need"][k] for k in RES_KEYS}
    work = np.array([gr["work"] for gr in groups])
    cv = float(work.std() / work.mean()) if work.mean() > 0 else 0.0
    return {"labels": tuple(labels), "groups": groups, "total": total, "gap": gap, "redund": redund,
            "gap_sum": sum(gap.values()), "res_sum": sum(total.values()), "red_sum": sum(redund.values()),
            "cv": cv, "ratio": float(work.max() / max(work.min(), 1e-9)),
            "ms_spread": max(gr["makespan"] for gr in groups) - min(gr["makespan"] for gr in groups)}


def partitions(n, K):
    """n 个块划分为恰好 K 个非空组的全部方案（限制增长串，组编号无重复计数）"""
    def rec(i, labels, m):
        if i == n:
            if m == K:
                yield list(labels)
            return
        for g in range(min(m + 1, K)):
            if n - i - 1 < K - max(m, g + 1):
                continue
            labels.append(g)
            yield from rec(i + 1, labels, max(m, g + 1))
            labels.pop()
    yield from rec(0, [], 0)


CV_MAX = 0.20                                       # ε-约束：组间工作量变异系数上限
EPS_LIST = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, float("inf"))


def lex_key(r):
    return (r["gap_sum"], r["res_sum"], round(r["cv"], 6), r["ms_spread"])


def best_under(rs, eps):
    ok = [r for r in rs if r["cv"] <= eps + 1e-12]
    return min(ok, key=lex_key) if ok else None


def pareto(results):
    """在（资源缺口, 资源规模, 工作量CV）三目标下的非支配解"""
    pts = [(r["gap_sum"], r["res_sum"], r["cv"]) for r in results]
    front = []
    for i, p in enumerate(pts):
        if not any(all(q[k] <= p[k] for k in range(3)) and any(q[k] < p[k] for k in range(3))
                   for j, q in enumerate(pts) if j != i):
            front.append(results[i])
    return front


# ---------------------------------------------------------------
# 3. 输出
# ---------------------------------------------------------------
def config_rows(K, res):
    rows = []
    for g, gr in enumerate(res["groups"], 1):
        n = gr["need"]
        rows.append({"K（2或3）": K, "任务组编号": f"G{g}", "服务区列表": ",".join(gr["areas"]),
                     "A型运输无人机数": n["U_A"], "B型运输无人机数": n["U_B"], "C型运输无人机数": n["U_C"],
                     "A型电池组数": n["B_A"], "B型电池组数": n["B_B"], "C型电池组数": n["B_C"],
                     "中继无人机数": n["R"], "中继能源组件数": n["M"]})
    return rows


def workload_rows(K, res):
    rows = []
    for g, gr in enumerate(res["groups"], 1):
        rows.append({"K": K, "任务组": f"G{g}", "服务区数": len(gr["areas"]), "货箱数": gr["n_box"],
                     "运输架次": gr["n_trips"], "中继架次（含复制）": gr["n_relay"],
                     "运输能耗(kWh)": round(gr["E_t"], 3), "中继能耗(kWh)": round(gr["E_r"], 3),
                     "运输作业时长(s)": round(gr["work_t"], 1), "中继作业时长(s)": round(gr["work_r"], 1),
                     "总作业时长(s)": round(gr["work"], 1), "组完成时刻(s)": round(gr["makespan"], 1),
                     "资源总数": sum(gr["need"].values()),
                     "运输架次列表": ",".join(t["tid"] for t in trips_of(gr["blocks"])),
                     "依托中继架次": ",".join(sorted(gr["serve"]))})
    return rows


def compare_rows(K, res, name):
    row = {"方案": name, "K": K, "资源配置总数": res["res_sum"], "资源冗余(相对不分区)": res["red_sum"],
           "资源缺口合计": res["gap_sum"], "工作量CV": round(res["cv"], 4), "工作量最大/最小": round(res["ratio"], 3),
           "组完成时刻极差(s)": round(res["ms_spread"], 1)}
    for k in RES_KEYS:
        row[f"{RES_NAME[k]}(需求/库存)"] = f"{res['total'][k]}/{STOCK[k]}"
    return row


def gap_reasons(K, res):
    """逐类资源解释缺口/冗余的来源：找出各组峰值时段，以及被多组复制的中继架次"""
    rows = []
    groups = res["groups"]
    shared = {}
    for g, gr in enumerate(groups, 1):
        for rid in gr["serve"]:
            shared.setdefault(rid, []).append(f"G{g}")
    for k in RES_KEYS:
        if res["total"][k] <= BASE["need"][k] and res["gap"][k] == 0:
            continue
        parts = []
        for g, gr in enumerate(groups, 1):
            n = gr["need"][k]
            if n == 0:
                continue
            parts.append(f"G{g}需{n}")
        why = []
        if k in ("R", "M"):
            dup = [f"{rid}({'/'.join(v)})" for rid, v in shared.items() if len(v) > 1]
            if dup:
                why.append("同一中继架次同时保障多个组的运输架次，分区后需各组独立派出：" + "、".join(dup))
            why.append("不同组的中继任务时段重叠，无法再由同一中继实体/组件先后执行")
        else:
            why.append("各组的高峰时段相互重叠，不分区时可错峰共用的无人机/电池被“锁定”在各自组内")
            if k.startswith("B"):
                why.append("电池返回后需充电周转（占用至充满），组内峰值被放大")
        rows.append({"K": K, "资源": RES_NAME[k], "不分区需求": BASE["need"][k], "分区后需求合计": res["total"][k],
                     "库存": STOCK[k], "冗余": res["redund"][k], "缺口": res["gap"][k],
                     "各组需求": "，".join(parts), "原因": "；".join(why)})
    return rows


def assignment_rows(K, res):
    """给出每组达到最少资源数的具体分配（组内编号），证明配置可执行"""
    rows = []
    for g, gr in enumerate(res["groups"], 1):
        trips = trips_of(gr["blocks"])
        for tp in "ABC":
            ts = [t for t in trips if t["g"] == tp]
            ua = interval_assign([(t["tid"], t["start"], t["end"]) for t in ts])
            ba = interval_assign([(t["tid"], t["start"], t["charge_end"]) for t in ts])
            for t in ts:
                rows.append({"K": K, "任务组": f"G{g}", "任务": t["tid"], "类型": f"{tp}型运输",
                             "开始(s)": round(t["start"], 1), "结束(s)": round(t["end"], 1),
                             "组内无人机": f"G{g}-{tp}{ua[t['tid']]}", "组内电池": f"G{g}-{tp}B{ba[t['tid']]}"})
        ra = interval_assign([(r["rid"], r["launch"], r["ret"] + RELAY["turn"]) for r in gr["relays"]])
        ma = interval_assign([(r["rid"], r["launch"], r["charge_end"]) for r in gr["relays"]])
        for r in gr["relays"]:
            rows.append({"K": K, "任务组": f"G{g}", "任务": r["rid"], "类型": f"中继@{r['site']}",
                         "开始(s)": round(r["launch"], 1), "结束(s)": round(r["ret"], 1),
                         "组内无人机": f"G{g}-R{ra[r['rid']]}", "组内电池": f"G{g}-M{ma[r['rid']]}"})
    return rows


def draw_partition(res, K, fname):
    lons = [NODES[k]["lon"] for k in NODES] + [s["pos"][0] for s in SITES.values()]
    lats = [NODES[k]["lat"] for k in NODES] + [s["pos"][1] for s in SITES.values()]
    x0, x1 = min(lons) - 0.015, max(lons) + 0.015
    y0, y1 = min(lats) - 0.015, max(lats) + 0.015
    c0, c1 = int((x0 - DEM_X0) / DEM_DX), int((x1 - DEM_X0) / DEM_DX)
    r0, r1 = int((DEM_Y0 - y1) / DEM_DY), int((DEM_Y0 - y0) / DEM_DY)
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(DEM_Z[r0:r1, c0:c1], extent=[x0, x1, y0, y1], cmap="Greys", origin="upper", alpha=0.55)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for g, gr in enumerate(res["groups"]):
        for t in trips_of(gr["blocks"]):
            pts = ["O01"] + t["areas"] + ["O01"]
            ax.plot([NODES[k]["lon"] for k in pts], [NODES[k]["lat"] for k in pts], "-", color=colors[g], lw=1.0, alpha=0.5)
        for s in gr["areas"]:
            n = NODES[s]
            ax.plot(n["lon"], n["lat"], "o", color=colors[g], ms=11, mec="k")
            ax.text(n["lon"] + 0.002, n["lat"] + 0.002, s, fontsize=9, weight="bold", color=colors[g])
        nd = gr["need"]
        ax.plot([], [], "o", color=colors[g], ms=10, mec="k",
                label=f"G{g + 1}：{len(gr['areas'])}个服务区  运输机A/B/C={nd['U_A']}/{nd['U_B']}/{nd['U_C']}  "
                      f"电池={nd['B_A']}/{nd['B_B']}/{nd['B_C']}  中继={nd['R']}  组件={nd['M']}")
    for name, s in SITES.items():
        ax.plot(s["pos"][0], s["pos"][1], "m^", ms=13, mec="k")
        ax.text(s["pos"][0] + 0.002, s["pos"][1] - 0.004, f"中继点位{name}", color="m", fontsize=9)
    o = NODES["O01"]
    ax.plot(o["lon"], o["lat"], "k*", ms=18)
    ax.text(o["lon"] + 0.002, o["lat"] + 0.002, "O01", fontsize=10, weight="bold")
    ax.set_xlabel("经度 (°)"); ax.set_ylabel("纬度 (°)")
    ax.set_title(f"问题四 {K} 组任务分区（资源缺口 {res['gap_sum']}，资源总数 {res['res_sum']}，工作量CV {res['cv']:.3f}）")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, fname), dpi=150); plt.close()


def draw_compare(best, all_res, fname):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    ax = axes[0]
    x = np.arange(len(RES_KEYS))
    w = 0.27
    ax.bar(x - w, [BASE["need"][k] for k in RES_KEYS], w, label="不分区（问题三）", color="0.6")
    ax.bar(x, [best[2]["total"][k] for k in RES_KEYS], w, label="2组", color="tab:blue")
    ax.bar(x + w, [best[3]["total"][k] for k in RES_KEYS], w, label="3组", color="tab:orange")
    ax.plot(x, [STOCK[k] for k in RES_KEYS], "r_", ms=30, mew=3, label="现有库存")
    ax.set_xticks(x); ax.set_xticklabels([RES_NAME[k] for k in RES_KEYS], rotation=35, ha="right")
    ax.set_ylabel("数量"); ax.set_title("资源配置规模与库存对比"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    for K, col in ((2, "tab:blue"), (3, "tab:orange")):
        for g, gr in enumerate(best[K]["groups"]):
            lab = f"K={K}-G{g + 1}"
            ax.bar(lab, gr["work_t"] / 3600, color=col)
            ax.bar(lab, gr["work_r"] / 3600, bottom=gr["work_t"] / 3600, color=col, alpha=0.45)
    ax.set_ylabel("作业时长 (h)（深色=运输，浅色=中继）"); ax.set_title("组间工作量"); ax.grid(axis="y", alpha=0.3)
    ax = axes[2]
    for K, mk in ((2, "o"), (3, "s")):
        rs = all_res[K]
        sc = ax.scatter([r["res_sum"] + (0.12 if K == 3 else -0.12) for r in rs], [r["cv"] for r in rs],
                        c=[r["gap_sum"] for r in rs], cmap="RdYlGn_r", marker=mk, s=14, alpha=0.6, label=f"K={K} 全部划分")
        ax.scatter([best[K]["res_sum"]], [best[K]["cv"]], marker="*", s=300, c="k")
        ax.annotate(f"K={K} 推荐", (best[K]["res_sum"], best[K]["cv"]), textcoords="offset points", xytext=(8, 8))
    ax.axhline(CV_MAX, color="k", ls="--", lw=0.8)
    ax.text(ax.get_xlim()[0], CV_MAX, f" ε = {CV_MAX}", va="bottom", fontsize=8)
    plt.colorbar(sc, ax=ax, label="资源缺口合计")
    ax.set_xlabel("资源配置总数"); ax.set_ylabel("组间工作量变异系数 CV"); ax.set_title("穷举全部分区：规模-均衡-缺口")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, fname), dpi=150); plt.close()


def draw_group_gantt(res, K, fname):
    rows, bars = [], []
    for g, gr in enumerate(res["groups"], 1):
        trips = trips_of(gr["blocks"])
        for tp in "ABC":
            ts = [t for t in trips if t["g"] == tp]
            ua = interval_assign([(t["tid"], t["start"], t["end"]) for t in ts])
            for t in ts:
                name = f"G{g}-{tp}{ua[t['tid']]}"
                if name not in rows:
                    rows.append(name)
                bars.append((name, t["start"], t["end"], t["tid"][3:], g))
        ra = interval_assign([(r["rid"], r["launch"], r["ret"] + RELAY["turn"]) for r in gr["relays"]])
        for r in gr["relays"]:
            name = f"G{g}-R{ra[r['rid']]}"
            if name not in rows:
                rows.append(name)
            bars.append((name, r["launch"], r["ret"], r["rid"][3:] + "@" + r["site"], g))
    rows.sort(key=lambda s: (s.split("-")[0], s.split("-")[1][0] == "R", s))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    fig, ax = plt.subplots(figsize=(15, 0.34 * len(rows) + 1.6))
    for name, a, b, lab, g in bars:
        y = rows.index(name)
        ax.barh(y, b - a, left=a, color=colors[g - 1], edgecolor="k", height=0.6,
                hatch="//" if "@" in lab else None, alpha=0.85)
        ax.text(a + 15, y, lab, va="center", fontsize=6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows); ax.invert_yaxis()
    ax.set_xlabel("时间 (s)（斜线=中继架次）"); ax.set_title(f"问题四 {K} 组独立执行的资源占用（组内编号）")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, fname), dpi=150); plt.close()


# ---------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("===== 问题四：救援任务分区与资源配置 =====")
    print(f"问题三方案：运输架次 {len(TRIPS)}，中继架次 {len(RELAYS)}")
    print("不可拆分的服务区块（同一架次访问的服务区合并）：")
    for k, blk in enumerate(BLOCKS):
        print(f"  块{k + 1}: {blk}  运输架次 {sum(1 for t in TRIPS if TRIP_BLOCK[t['tid']] == k)}")
    print("不分区（问题三原方案）资源需求：", BASE["need"], " 库存：", STOCK)
    multi = {rid: sorted({BLOCKS[TRIP_BLOCK[t]][0] for t in USAGE if rid in USAGE[t]}) for rid in RELAYS}
    for rid, v in multi.items():
        print(f"  {rid}（点位 {RELAYS[rid]['site']}）保障的块：{v}")

    all_res, best = {}, {}
    cmp_rows, conf_rows, work_rows, gap_rows, asg_rows, top_rows, modeB_rows, eps_rows = [], [], [], [], [], [], [], []
    for K in (2, 3):
        rs = [evaluate_partition(lb, K) for lb in partitions(len(BLOCKS), K)]
        rs.sort(key=lex_key)
        all_res[K] = rs
        best[K] = best_under(rs, CV_MAX)
        front = pareto(rs)
        print(f"\nK={K}：共穷举 {len(rs)} 种分区，帕累托非支配解 {len(front)} 个")
        for eps in EPS_LIST:
            b = best_under(rs, eps)
            if b is None:
                continue
            eps_rows.append({"K": K, "ε（CV上限）": "不限" if eps == float("inf") else eps,
                             "满足约束的分区数": sum(1 for r in rs if r["cv"] <= eps + 1e-12),
                             "资源缺口": b["gap_sum"], "资源总数": b["res_sum"], "冗余": b["red_sum"],
                             "工作量CV": round(b["cv"], 4), "最大/最小": round(b["ratio"], 3),
                             "分区": " | ".join(",".join(g["areas"]) for g in b["groups"])})
            print(f"   ε={eps}: 缺口 {b['gap_sum']}  资源 {b['res_sum']}  CV {b['cv']:.3f}  "
                  f"{[g['areas'] for g in b['groups']]}")
        ranked = sorted([r for r in rs if r["cv"] <= CV_MAX + 1e-12], key=lex_key)
        for rank, r in enumerate(ranked[:10], 1):
            top_rows.append({"K": K, "排名": rank, "分区": " | ".join(",".join(g["areas"]) for g in r["groups"]),
                             "资源缺口": r["gap_sum"], "资源总数": r["res_sum"], "冗余": r["red_sum"],
                             "工作量CV": round(r["cv"], 4), "最大/最小": round(r["ratio"], 3),
                             "是否帕累托": any(r is f for f in front)})
        # 两个极端作对照，说明“省资源”与“均衡”之间的权衡
        lean = rs[0]
        bal = min(rs, key=lambda r: (r["cv"], r["gap_sum"], r["res_sum"]))
        cmp_rows.append(compare_rows(K, best[K], f"{K}组-推荐（CV≤{CV_MAX}，缺口→规模→均衡）"))
        cmp_rows.append(compare_rows(K, lean, f"{K}组-极端①只省资源（不限均衡）"))
        cmp_rows.append(compare_rows(K, bal, f"{K}组-极端②只求均衡"))
        conf_rows += config_rows(K, best[K])
        work_rows += workload_rows(K, best[K])
        gap_rows += gap_reasons(K, best[K])
        asg_rows += assignment_rows(K, best[K])
        rb = evaluate_partition(list(best[K]["labels"]), K, mode="B")
        modeB_rows.append({"K": K, "口径": "A 原中继架次整体复制（主口径）", "中继无人机": best[K]["total"]["R"],
                           "中继能源组件": best[K]["total"]["M"], "中继架次（含复制）": sum(g["n_relay"] for g in best[K]["groups"]),
                           "中继能耗(kWh)": round(sum(g["E_r"] for g in best[K]["groups"]), 3)})
        modeB_rows.append({"K": K, "口径": "B 副本收缩到本组使用区间", "中继无人机": rb["total"]["R"],
                           "中继能源组件": rb["total"]["M"], "中继架次（含复制）": sum(g["n_relay"] for g in rb["groups"]),
                           "中继能耗(kWh)": round(sum(g["E_r"] for g in rb["groups"]), 3)})
    base_row = {"方案": "不分区（问题三原方案）", "K": 1, "资源配置总数": sum(BASE["need"].values()),
                "资源冗余(相对不分区)": 0, "资源缺口合计": 0, "工作量CV": 0.0, "工作量最大/最小": 1.0, "组完成时刻极差(s)": 0.0}
    for k in RES_KEYS:
        base_row[f"{RES_NAME[k]}(需求/库存)"] = f"{BASE['need'][k]}/{STOCK[k]}"
    cmp_rows.insert(0, base_row)
    modeB_rows.insert(0, {"K": 1, "口径": "不分区", "中继无人机": BASE["need"]["R"], "中继能源组件": BASE["need"]["M"],
                          "中继架次（含复制）": len(RELAYS), "中继能耗(kWh)": round(BASE["E_r"], 3)})

    df_conf = pd.DataFrame(conf_rows)
    df_cmp = pd.DataFrame(cmp_rows)
    df_work = pd.DataFrame(work_rows)
    df_gap = pd.DataFrame(gap_rows) if gap_rows else pd.DataFrame([{"说明": "无资源缺口与冗余"}])
    df_top = pd.DataFrame(top_rows)
    df_asg = pd.DataFrame(asg_rows)
    df_mb = pd.DataFrame(modeB_rows)
    df_blk = pd.DataFrame([{"块": f"块{k + 1}", "服务区": ",".join(b),
                            "运输架次": ",".join(t["tid"] for t in TRIPS if TRIP_BLOCK[t["tid"]] == k)}
                           for k, b in enumerate(BLOCKS)])
    print("\n" + df_conf.to_string(index=False))
    print("\n" + df_cmp.to_string(index=False))
    print("\n" + df_work.drop(columns=["运输架次列表"]).to_string(index=False))
    print("\n" + df_gap.to_string(index=False))
    print("\n" + df_mb.to_string(index=False))
    print("\n" + pd.DataFrame(eps_rows).drop(columns=["分区"]).to_string(index=False))

    with pd.ExcelWriter(os.path.join(RESULT_DIR, "Q4_结果.xlsx")) as w:
        df_conf.to_excel(w, sheet_name="Q4_分区配置", index=False)
        df_cmp.to_excel(w, sheet_name="方案比较", index=False)
        df_work.to_excel(w, sheet_name="组间工作量", index=False)
        df_gap.to_excel(w, sheet_name="冗余与缺口原因", index=False)
        df_mb.to_excel(w, sheet_name="中继复制口径对比", index=False)
        pd.DataFrame(eps_rows).to_excel(w, sheet_name="ε约束权衡曲线", index=False)
        df_top.to_excel(w, sheet_name="候选分区Top10", index=False)
        df_asg.to_excel(w, sheet_name="组内资源分配", index=False)
        df_blk.to_excel(w, sheet_name="服务区块", index=False)
    draw_partition(best[2], 2, "Q4_2组分区地图.png")
    draw_partition(best[3], 3, "Q4_3组分区地图.png")
    draw_compare(best, all_res, "Q4_方案对比.png")
    draw_group_gantt(best[2], 2, "Q4_2组资源甘特图.png")
    draw_group_gantt(best[3], 3, "Q4_3组资源甘特图.png")
    print("\n问题四完成，结果已保存到 结果/Q4_结果.xlsx 与 结果/图/。")
