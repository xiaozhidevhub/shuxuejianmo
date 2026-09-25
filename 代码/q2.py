# -*- coding: utf-8 -*-
"""
q2.py —— 问题二：异构无人机多点多架次运输调度（不考虑通信）
=========================================================
求解流程：
  1. 初始解：问题一的单点最优组批（每个服务区单独成批）
  2. 多个随机种子分别做模拟退火，取最好的解，再用爬山法“修正”
  3. 改变目标权重（时效优先 / 能耗优先 / 架次优先）重新优化，分析权衡关系
  4. 输出运输架次表、逐箱交付表、资源使用情况、可行性检验和图
"""
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vrp
from common import *

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def optimize(weights, seeds=(1, 2, 3), iters=40000, init=None, verbose=True):
    """在给定权重下，多种子退火 + 爬山修正，返回最优解"""
    vrp.WEIGHTS.update(weights)
    best, best_cost = None, float("inf")
    for sd in seeds:
        start = init if init is not None else vrp.initial_from_q1()
        t0 = time.time()
        sol, c = vrp.anneal(start, iters=iters, seed=sd, verbose=verbose)
        sol, c = vrp.local_polish(sol)
        if verbose:
            print(f"  种子 {sd}: 目标值 {c:.2f}，用时 {time.time() - t0:.1f}s")
        if c < best_cost:
            best, best_cost = sol, c
    return best, best_cost


def metrics_row(name, info):
    return {"方案": name, "架次数": info["trips"], "总能耗(kWh)": round(info["energy"], 3),
            "完成时间(s)": round(info["makespan"], 1), "加权延误(系数·min)": round(info["weighted_tardiness"], 2),
            "准时箱数(/80)": info["on_time"], "硬时限违反(s)": round(info["hard_violation_s"], 1)}


def plan_tables(plan, prefix="T"):
    """把调度结果整理成“运输架次表”和“逐箱交付表”"""
    trips, boxes = [], []
    for k, p in enumerate(plan, 1):
        tid = f"{prefix}{k:02d}"
        p["tid"] = tid
        ev = p["ev"]
        trips.append({"架次编号": tid, "无人机编号": p["uav"], "机型编号": ev["g"], "电池编号": p["batt"],
                      "开始时刻（s）": round(p["start"], 1),
                      "访问服务区顺序": "O01→" + "→".join(s for s, _ in ev["stops"]) + "→O01",
                      "返回O01时刻（s）": round(p["end"], 1), "架次能耗（kWh）": round(ev["E"], 4),
                      "载重(kg)": ev["mass"], "体积(m³)": round(ev["vol"], 4),
                      "返航SOC(%)": round(ev["soc"] * 100, 2), "电池充满时刻(s)": round(p["charge_end"], 1)})
        for s, ids in ev["stops"]:
            for i in ids:
                t = p["start"] + ev["deliver"][i]
                b = BOXES[i]
                boxes.append({"货箱编号": i, "架次编号": tid, "服务区编号": s, "交付完成时刻（s）": round(t, 1),
                              "期望送达时间(s)": b["due"], "硬时限(s)": b["hard"],
                              "是否准时": "是" if t <= b["due"] + 1e-6 else "否",
                              "延误(s)": round(max(0, t - b["due"]), 1)})
    boxes.sort(key=lambda r: r["货箱编号"])
    return pd.DataFrame(trips), pd.DataFrame(boxes)


def resource_table(plan):
    """统计每架无人机、每组电池的使用情况"""
    rows = []
    for u, g in DRONES:
        ps = [p for p in plan if p["uav"] == u]
        busy = sum(p["end"] - p["start"] for p in ps)
        rows.append({"资源": u, "类型": f"{g}型运输无人机", "使用次数": len(ps),
                     "占用时长(s)": round(busy, 1), "承担架次": ",".join(p["tid"] for p in ps)})
    for g, info in BATTERIES.items():
        for k in range(info["count"]):
            b = f"{g}-B{k + 1:02d}"
            ps = [p for p in plan if p["batt"] == b]
            rows.append({"资源": b, "类型": f"{g}型共享电池", "使用次数": len(ps),
                         "占用时长(s)": round(sum(p["end"] - p["start"] for p in ps), 1),
                         "承担架次": ",".join(p["tid"] for p in ps),
                         "充电时长(s)": round(sum(p["ev"]["charge"] for p in ps), 1)})
    return pd.DataFrame(rows)


def draw_routes(plan, title, fname, relay_sorties=None):
    """在 DEM 底图上画出所有运输路线（和中继悬停点）"""
    lons = [NODES[k]["lon"] for k in NODES]
    lats = [NODES[k]["lat"] for k in NODES]
    x0, x1 = min(lons) - 0.02, max(lons) + 0.02
    y0, y1 = min(lats) - 0.02, max(lats) + 0.02
    c0, c1 = int((x0 - DEM_X0) / DEM_DX), int((x1 - DEM_X0) / DEM_DX)
    r0, r1 = int((DEM_Y0 - y1) / DEM_DY), int((DEM_Y0 - y0) / DEM_DY)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(DEM_Z[r0:r1, c0:c1], extent=[x0, x1, y0, y1], cmap="terrain", origin="upper")
    plt.colorbar(im, ax=ax, label="地面高程 (m)", shrink=0.7)
    cmap = plt.get_cmap("tab10")
    uav_ids = [u for u, _ in DRONES]
    for p in plan:
        pts = ["O01"] + [s for s, _ in p["ev"]["stops"]] + ["O01"]
        xs = [NODES[k]["lon"] for k in pts]
        ys = [NODES[k]["lat"] for k in pts]
        jit = (uav_ids.index(p["uav"]) - 3.5) * 0.0004            # 轻微错开，避免线条重叠
        ax.plot(np.array(xs) + jit, np.array(ys) + jit, "-", color=cmap(uav_ids.index(p["uav"])),
                lw=1.3, alpha=0.8)
    for u in uav_ids:
        ax.plot([], [], color=cmap(uav_ids.index(u)), label=f"{u}({dict(DRONES)[u]}型)")
    for k, n in NODES.items():
        ax.plot(n["lon"], n["lat"], "k*" if k == "O01" else "ro", ms=14 if k == "O01" else 6)
        ax.text(n["lon"] + 0.002, n["lat"] + 0.002, k, fontsize=9, weight="bold")
    if relay_sorties:
        for r in relay_sorties:
            ax.plot(r["lon"], r["lat"], "m^", ms=14)
            ax.text(r["lon"] + 0.002, r["lat"] - 0.004, "中继悬停点", color="m", fontsize=10)
    ax.set_xlabel("经度 (°)"); ax.set_ylabel("纬度 (°)"); ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, fname), dpi=150); plt.close()


def draw_gantt(plan, title, fname, relay=None):
    """甘特图：每架无人机、每组电池的时间占用（电池的充电时段用浅色表示）"""
    rows = [u for u, _ in DRONES]
    for g, info in BATTERIES.items():
        rows += [f"{g}-B{k + 1:02d}" for k in range(info["count"])]
    if relay:
        rows += RELAY_IDS + [f"M{k + 1:02d}" for k in range(MODULES["count"])]
    fig, ax = plt.subplots(figsize=(15, 0.36 * len(rows) + 1.5))
    color = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
    for p in plan:
        g = p["ev"]["g"]
        y = rows.index(p["uav"])
        ax.barh(y, p["end"] - p["start"], left=p["start"], color=color[g], edgecolor="k", height=0.6)
        ax.text(p["start"] + 20, y, p.get("tid", ""), va="center", fontsize=6, color="w")
        y = rows.index(p["batt"])
        ax.barh(y, p["end"] - p["start"], left=p["start"], color=color[g], edgecolor="k", height=0.6)
        ax.barh(y, p["charge_end"] - p["end"], left=p["end"], color=color[g], alpha=0.3, height=0.6)
    if relay:
        for r in relay["sorties"]:
            y = rows.index(r["relay"])
            ax.barh(y, r["ret"] - r["launch"], left=r["launch"], color="tab:purple", edgecolor="k", height=0.6)
            ax.barh(y, r["end"] - r["link_done"], left=r["link_done"], color="m", height=0.3)
            ax.text(r["launch"] + 20, y, r["rid"], va="center", fontsize=6, color="w")
            y = rows.index(r["module"])
            ax.barh(y, r["ret"] - r["launch"], left=r["launch"], color="tab:purple", edgecolor="k", height=0.6)
            ax.barh(y, r["charge_end"] - r["ret"], left=r["ret"], color="tab:purple", alpha=0.3, height=0.6)
    for d in (3600, 7200, 10800, 14400):
        ax.axvline(d, color="r", ls="--", lw=0.8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.invert_yaxis(); ax.set_xlabel("时间 (s)（红色虚线为 1h/2h/3h/4h 时限）"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, fname), dpi=150); plt.close()


if __name__ == "__main__":
    print("===== 问题二：主方案（均衡权重）优化 =====")
    base_w = dict(vrp.WEIGHTS)
    best, c = optimize(base_w, seeds=(1, 2, 3, 4, 5), iters=200000)
    _, info = vrp.evaluate(best, detail=True)
    plan = info["plan"]
    df_trip, df_box = plan_tables(plan, "Q2-T")
    df_res = resource_table(plan)
    checks = vrp.check_plan(plan)
    df_chk = pd.DataFrame(checks, columns=["检验内容", "是否通过", "说明"])

    print("\n主方案指标：", metrics_row("均衡(主方案)", info))
    print(df_trip.to_string(index=False))
    print(df_chk.to_string(index=False))

    # ---- 权衡分析：改变权重重新优化 ----
    print("\n===== 权衡分析 =====")
    scenarios = [
        ("时效优先", {"tard": 6.0, "T": 4.0, "E": 1.0, "N": 1.0}),
        ("能耗优先", {"tard": 0.3, "T": 0.3, "E": 40.0, "N": 3.0}),
        ("架次优先", {"tard": 0.3, "T": 0.3, "E": 2.0, "N": 40.0}),
        ("完成时间优先", {"tard": 0.5, "T": 20.0, "E": 1.0, "N": 1.0}),
    ]
    rows = [metrics_row("均衡(主方案)", info)]
    for name, w in scenarios:
        ww = dict(base_w); ww.update(w)
        sol, _ = optimize(ww, seeds=(7, 8), iters=120000, init=best, verbose=False)
        vrp.WEIGHTS.update(ww)
        _, inf2 = vrp.evaluate(sol, detail=True)
        rows.append(metrics_row(name, inf2))
        print("  ", rows[-1])
    vrp.WEIGHTS.update(base_w)
    df_tradeoff = pd.DataFrame(rows)
    print(df_tradeoff.to_string(index=False))

    # ---- 保存 ----
    with pd.ExcelWriter(os.path.join(RESULT_DIR, "Q2_结果.xlsx")) as w:
        df_trip.to_excel(w, sheet_name="Q2_运输架次", index=False)
        df_box.to_excel(w, sheet_name="Q2_逐箱交付", index=False)
        df_res.to_excel(w, sheet_name="资源使用", index=False)
        df_chk.to_excel(w, sheet_name="可行性检验", index=False)
        df_tradeoff.to_excel(w, sheet_name="权衡分析", index=False)
    with open(os.path.join(RESULT_DIR, "q2_best.pkl"), "wb") as f:
        pickle.dump(best, f)
    draw_routes(plan, "问题二 运输路线（颜色=执行无人机）", "Q2_运输路线.png")
    draw_gantt(plan, "问题二 无人机与电池资源占用甘特图", "Q2_资源甘特图.png")
    print("\n问题二完成。")
