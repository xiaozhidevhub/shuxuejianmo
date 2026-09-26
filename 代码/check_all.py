# -*- coding: utf-8 -*-
"""
check_all.py —— 结果独立复核
=================================================
只读取 结果/Q1~Q4_结果.xlsx 中“提交出去的表格”，从原始数据与附录规则重新推算，
不调用任何求解器内部函数（vrp / q3 / q4 / comm_geo），用来发现求解代码自身的口径错误：
  Q1  货箱全覆盖、单服务区、载重/体积、往返时间与能耗、返航 SOC
  Q2  同上 + 逐箱交付时刻、硬时限、无人机/电池不重叠（含两阶段充电）、机型不混用
  Q3  同 Q2 + 中继架次时间链（准备→飞行→建链→服务→返航）、能耗、SOC、离地高度、DEM 范围、
      回传链路、中继周转 300 s、能源组件充电不重叠；
      通信保障表按 0.1 s 步长逐点复核（直连 / 指定中继架次的接入+回传同时可用且处于服务窗口），
      并检查表格是否无缝覆盖每个架次的全部爬升、巡航、下降与投送时段
  Q4  分区完整性、同架次服务区同组、各组资源数（按区间最大重叠）与提交表一致
"""
import sys
import numpy as np
import pandas as pd
from common import *

TOL = 0.15          # 表格时刻保留 1 位小数，比对容差（s）
DT = 0.1            # 通信复核步长（s）
REPORT = []


def check(name, ok, msg=""):
    REPORT.append((name, bool(ok), msg))
    print(f"  [{'通过' if ok else '失败'}] {name}  {msg}")


def read(q, sheet):
    return pd.read_excel(os.path.join(RESULT_DIR, f"{q}_结果.xlsx"), sheet_name=sheet)


# ---------------------------------------------------------------
# 架次时间线（独立实现附录 2）
# ---------------------------------------------------------------
def trip_timeline(g, stops):
    """stops = [(服务区, [货箱...]), ...]；返回 dict：能耗、时长、各箱送达偏移、阶段列表、关键帧"""
    t = TYPES[g]
    ids = [i for _, b in stops for i in b]
    q = sum(BOXES[i]["mass"] for i in ids)
    clock = t["prep"] + t["load"] * len(ids)
    E = 0.0
    cur = "O01"
    deliver, phases = {}, []
    kf = [(0.0,) + (NODES["O01"]["lon"], NODES["O01"]["lat"], NODES["O01"]["work_alt"])]
    route = [s for s, _ in stops] + ["O01"]
    boxes_at = dict(stops)
    for s in route:
        seg = node_segment(cur, s)
        E += seg_energy(g, seg, q)
        na, nb = NODES[cur], NODES[s]
        t1 = clock + seg["h_up"] / t["v_up"]
        t2 = t1 + seg["d"] / t["v_c"]
        t3 = t2 + seg["h_down"] / t["v_down"]
        kf += [(clock, na["lon"], na["lat"], na["work_alt"]), (t1, na["lon"], na["lat"], seg["cruise"]),
               (t2, nb["lon"], nb["lat"], seg["cruise"]), (t3, nb["lon"], nb["lat"], nb["work_alt"])]
        phases += [(f"{cur}→{s} 爬升", clock, t1), (f"{cur}→{s} 巡航", t1, t2), (f"{cur}→{s} 下降", t2, t3)]
        clock = t3
        if s != "O01":
            n = len(boxes_at[s])
            end = clock + t["hand_base"] + t["hand_box"] * n
            phases.append((f"{s} 投送", clock, end))
            kf.append((end, nb["lon"], nb["lat"], nb["work_alt"]))
            for i in boxes_at[s]:
                deliver[i] = end
            clock = end
            q -= sum(BOXES[i]["mass"] for i in boxes_at[s])
            q = max(q, 0.0)
        cur = s
    mass = sum(BOXES[i]["mass"] for i in ids)
    vol = sum(BOXES[i]["vol"] for i in ids)
    return {"E": E, "dur": clock, "deliver": deliver, "phases": phases, "kf": np.array(kf),
            "mass": mass, "vol": vol, "soc": 1 - E / t["E"], "ids": ids}


def overlap_ok(items):
    """items = [(资源, 开始, 占用结束)]；同一资源区间不得重叠"""
    by = {}
    for r, a, b in items:
        by.setdefault(r, []).append((a, b))
    bad = []
    for r, iv in by.items():
        iv.sort()
        for (a1, b1), (a2, b2) in zip(iv, iv[1:]):
            if a2 < b1 - TOL:
                bad.append((r, round(b1, 1), round(a2, 1)))
    return bad


def max_overlap(iv):
    """同时占用数的最大值；区间右端减去 TOL，使“前一任务结束即下一任务开始”（表格已四舍五入）不被误判为重叠"""
    iv = [(a, b - TOL) for a, b in iv]
    ev = sorted([(a, 1) for a, b in iv] + [(b, -1) for a, b in iv], key=lambda x: (x[0], x[1]))
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return best


# ---------------------------------------------------------------
# Q1
# ---------------------------------------------------------------
def check_q1():
    print("\n==== 问题一 ====")
    df = read("Q1", "Q1_单点组批")
    seen = []
    bad_phys, bad_num = [], []
    for _, r in df.iterrows():
        ids = str(r["货箱编号列表"]).split(",")
        seen += ids
        g, s = r["机型编号"], r["服务区编号"]
        if any(BOXES[i]["area"] != s for i in ids):
            bad_phys.append((r["架次编号"], "跨服务区"))
        tl = trip_timeline(g, [(s, ids)])
        tp = TYPES[g]
        if tl["mass"] > tp["Q"] + 1e-9 or tl["vol"] > tp["V"] + 1e-9 or tl["soc"] < tp["rho"] - 1e-9:
            bad_phys.append((r["架次编号"], f"载重{tl['mass']}/体积{tl['vol']:.3f}/SOC{tl['soc']:.3f}"))
        if (abs(tl["E"] - r["架次能耗（kWh）"]) > 1e-3 or abs(tl["dur"] - r["往返时间（s）"]) > TOL
                or abs(tl["mass"] - r["总质量（kg）"]) > 1e-6 or abs(tl["soc"] * 100 - r["返航SOC（%）"]) > 0.01):
            bad_num.append((r["架次编号"], round(tl["E"], 4), r["架次能耗（kWh）"], round(tl["dur"], 1), r["往返时间（s）"]))
    cap = read("Q1", "最大安全载荷")
    bad_cap = []
    for _, r in cap.iterrows():
        s = r["服务区"]
        for g in "ABC":
            tp = TYPES[g]
            e = lambda q: seg_energy(g, node_segment("O01", s), q) + seg_energy(g, node_segment(s, "O01"), 0.0)
            lim = (1 - tp["rho"]) * tp["E"]
            if e(tp["Q"]) <= lim:
                qmax = tp["Q"]
            elif e(0.0) > lim:
                qmax = 0.0
            else:
                lo, hi = 0.0, tp["Q"]
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if e(mid) <= lim else (lo, mid)
                qmax = lo
            if abs(qmax - r[f"{g}型最大安全载荷(kg)"]) > 1e-3:
                bad_cap.append((s, g, round(qmax, 3), r[f"{g}型最大安全载荷(kg)"]))
    check("Q1 最大安全载荷（返航余量20%约束下二分求解）与独立重算一致", not bad_cap, str(bad_cap[:3]))
    check("Q1 80 箱全覆盖且每箱一次", sorted(seen) == sorted(BOXES), f"{len(seen)} 箱 / {len(df)} 架次")
    check("Q1 单服务区、载重、体积、返航SOC≥20%", not bad_phys, str(bad_phys[:3]))
    check("Q1 往返时间/能耗/SOC 与独立重算一致", not bad_num, str(bad_num[:3]))


# ---------------------------------------------------------------
# Q2 / Q3 运输部分
# ---------------------------------------------------------------
def check_transport(q):
    trips = read(q, f"{q}_运输架次")
    box = read(q, f"{q}_逐箱交付")
    check(f"{q} 80 箱全覆盖且每箱一次", sorted(box["货箱编号"]) == sorted(BOXES) and box["货箱编号"].is_unique,
          f"{len(box)} 箱 / {len(trips)} 架次")
    drone_type = dict(DRONES)
    res = {}
    bad_phys, bad_num, bad_type, late = [], [], [], []
    tard = 0.0
    for _, r in trips.iterrows():
        tid, g = r["架次编号"], r["机型编号"]
        route = r["访问服务区顺序"].split("→")[1:-1]
        mine = box[box["架次编号"] == tid]
        stops = [(s, list(mine[mine["服务区编号"] == s]["货箱编号"])) for s in route]
        if len(set(route)) != len(route) or sum(len(b) for _, b in stops) != len(mine):
            bad_phys.append((tid, "路线与逐箱表不一致"))
        tl = trip_timeline(g, stops)
        tp = TYPES[g]
        if tl["mass"] > tp["Q"] + 1e-9 or tl["vol"] > tp["V"] + 1e-9 or tl["soc"] < tp["rho"] - 1e-9:
            bad_phys.append((tid, f"载重{tl['mass']}/体积{tl['vol']:.3f}/SOC{tl['soc']:.3f}"))
        st = float(r["开始时刻（s）"])
        end = st + tl["dur"]
        charge_end = end + charge_time(tl["soc"], BATTERIES[g]["full"])
        if abs(end - r["返回O01时刻（s）"]) > TOL or abs(tl["E"] - r["架次能耗（kWh）"]) > 1e-3:
            bad_num.append((tid, round(end, 1), r["返回O01时刻（s）"], round(tl["E"], 4), r["架次能耗（kWh）"]))
        if "电池充满时刻(s)" in r and abs(charge_end - r["电池充满时刻(s)"]) > TOL:
            bad_num.append((tid, "充满时刻", round(charge_end, 1), r["电池充满时刻(s)"]))
        if drone_type.get(r["无人机编号"]) != g or not str(r["电池编号"]).startswith(g + "-") \
                or int(str(r["电池编号"]).split("B")[-1]) > BATTERIES[g]["count"]:
            bad_type.append(tid)
        for _, b in mine.iterrows():
            i = b["货箱编号"]
            t_d = st + tl["deliver"][i]
            if abs(t_d - b["交付完成时刻（s）"]) > TOL:
                bad_num.append((tid, i, round(t_d, 1), b["交付完成时刻（s）"]))
            bx = BOXES[i]
            if bx["hard"] is not None and t_d > bx["hard"] + 1e-6:
                late.append((i, round(t_d, 1), bx["hard"]))
            tard += bx["coef"] * max(0.0, t_d - bx["due"]) / 60
        res[tid] = {"g": g, "uav": r["无人机编号"], "batt": r["电池编号"], "start": st, "end": end,
                    "charge_end": charge_end, "tl": tl, "areas": route}
    check(f"{q} 载重、体积、返航SOC≥20%、路线与逐箱表一致", not bad_phys, str(bad_phys[:3]))
    check(f"{q} 返回时刻/能耗/充满时刻/逐箱交付时刻与独立重算一致", not bad_num, str(bad_num[:3]))
    check(f"{q} 无人机与电池机型匹配、编号在库存内", not bad_type, str(bad_type[:3]))
    check(f"{q} 医疗物资与首批货箱硬时限", not late, f"违反 {len(late)} 箱；加权延误 {tard:.2f} 系数·min")
    b1 = overlap_ok([(x["uav"], x["start"], x["end"]) for x in res.values()])
    b2 = overlap_ok([(x["batt"], x["start"], x["charge_end"]) for x in res.values()])
    check(f"{q} 无人机任务不重叠、电池占用+充电不重叠", not b1 and not b2, str((b1 + b2)[:3]))
    ms = max(x["end"] for x in res.values())
    E = sum(x["tl"]["E"] for x in res.values())
    print(f"     运输完成 {ms:.1f} s，运输能耗 {E:.4f} kWh，架次 {len(res)}")
    return res


# ---------------------------------------------------------------
# Q3 中继与通信
# ---------------------------------------------------------------
def check_relay():
    rel = read("Q3", "Q3_中继架次")
    R = RELAY
    P = R["P_hover"] + R["P_comm"]
    out = {}
    bad_num, bad_phys = [], []
    for _, r in rel.iterrows():
        pos = (float(r["悬停经度（°）"]), float(r["悬停纬度（°）"]), float(r["悬停海拔（m）"]))
        lg = relay_legs(*pos)
        l, ws, we = float(r["开始时刻（s）"]), float(r["建链完成时刻（s）"]), float(r["服务结束时刻（s）"])
        ret = we + lg["t_back"]
        e = lg["e_out"] + lg["e_back"] + P * (R["link"] + we - ws) / 3600
        soc = 1 - e / R["E"]
        if abs(l + R["prep"] + lg["t_out"] + R["link"] - ws) > TOL or abs(ret - r["返回O01时刻（s）"]) > TOL \
                or abs(e - r["架次能耗（kWh）"]) > 1e-3:
            bad_num.append((r["中继架次编号"], round(l + R["prep"] + lg["t_out"] + R["link"], 1), ws,
                            round(ret, 1), r["返回O01时刻（s）"], round(e, 4), r["架次能耗（kWh）"]))
        agl = pos[2] - float(dem_at(pos[0], pos[1]))
        inside = DEM_LON_MIN <= pos[0] <= DEM_LON_MAX and DEM_LAT_MIN <= pos[1] <= DEM_LAT_MAX
        back = link_ok(pos, G01_POS, "back")
        if soc < R["rho"] - 1e-9 or agl > R["max_agl"] + 0.05 or agl < 0 or not inside or not back or l < -1e-6:
            bad_phys.append((r["中继架次编号"], f"SOC{soc:.3f} 离地{agl:.1f} 范围{inside} 回传{back}"))
        out[r["中继架次编号"]] = {"pos": pos, "r": r["中继无人机编号"], "m": r["能源组件编号"], "launch": l,
                               "ws": ws, "we": we, "ret": ret, "e": e,
                               "charge_end": ret + charge_time(soc, MODULES["full"])}
    check("Q3 中继时间链（准备180+飞行+建链30）、返航时刻、能耗与独立重算一致", not bad_num, str(bad_num[:3]))
    check("Q3 中继返航SOC≥20%、离地≤300 m、DEM 范围内、回传链路可用", not bad_phys, str(bad_phys[:3]))
    ok_id = all(x["r"] in RELAY_IDS for x in out.values()) and \
        all(int(x["m"][1:]) <= MODULES["count"] for x in out.values())
    b1 = overlap_ok([(x["r"], x["launch"], x["ret"] + R["turn"]) for x in out.values()])
    b2 = overlap_ok([(x["m"], x["launch"], x["charge_end"]) for x in out.values()])
    check("Q3 中继周转≥300 s、能源组件占用+充电不重叠、编号在库存内", ok_id and not b1 and not b2, str((b1 + b2)[:3]))
    return out


def check_comm(trips, relays):
    comm = read("Q3", "Q3_通信保障")
    comm["中继架次编号"] = comm["中继架次编号"].fillna("")
    gap_bad, mode_bad, lab_bad = [], [], []
    tot = {"直连G01": 0.0, "中继": 0.0, "中断": 0.0}
    true_out = 0.0
    n_pts = 0
    rel_list = list(relays.items())
    for tid, x in trips.items():
        rows = comm[comm["运输架次编号"] == tid].sort_values("开始时刻（s）")
        kf = x["tl"]["kf"]
        st = x["start"]
        # (1) 表格无缝覆盖所有阶段
        need = [(lab, st + a, st + b) for lab, a, b in x["tl"]["phases"] if b - a > 1e-9]
        segs = list(zip(rows["开始时刻（s）"], rows["结束时刻（s）"], rows["通信阶段"], rows["保障方式"], rows["中继架次编号"]))
        cov_t = need[0][1]
        for a, b, lab, m, rid in segs:
            if abs(a - cov_t) > TOL:
                gap_bad.append((tid, lab, round(cov_t, 1), a))
            cov_t = b
        if abs(cov_t - need[-1][2]) > TOL:
            gap_bad.append((tid, "结尾", round(cov_t, 1), round(need[-1][2], 1)))
        for lab, a, b in need:
            inside = [s for s in segs if s[2] == lab]
            if not inside or abs(min(s[0] for s in inside) - a) > TOL or abs(max(s[1] for s in inside) - b) > TOL:
                lab_bad.append((tid, lab))
        # (2) 逐 0.1 s 复核保障方式
        for a, b, lab, m, rid in segs:
            tot[m] = tot.get(m, 0.0) + (b - a)
            ts = np.unique(np.concatenate([np.arange(a, b, DT), [b]]))
            for t in ts:
                p = tuple(float(np.interp(t - st, kf[:, 0], kf[:, c])) for c in (1, 2, 3))
                n_pts += 1
                d_ok = link_ok(p, G01_POS, "direct")
                if m == "直连G01":
                    ok = d_ok
                elif m == "中继":
                    y = relays.get(rid)
                    ok = (y is not None and y["ws"] - TOL <= t <= y["we"] + TOL
                          and link_ok(p, y["pos"], "access") and link_ok(y["pos"], G01_POS, "back"))
                else:
                    ok = False
                if not ok:
                    mode_bad.append((tid, lab, round(t, 1), m, rid))
                    alt = d_ok or any(y["ws"] <= t <= y["we"] and link_ok(p, y["pos"], "access")
                                      for _, y in rel_list)
                    if not alt:
                        true_out += DT
    check("Q3 通信保障表无缝覆盖全部爬升/巡航/下降/投送时段", not gap_bad and not lab_bad,
          str((gap_bad + lab_bad)[:3]))
    check(f"Q3 通信保障逐 {DT} s 复核（共 {n_pts} 点）：所列保障方式在该时刻确实可用",
          not mode_bad, f"不符 {len(mode_bad)} 点 {mode_bad[:4]}；其中任何方式都不可用约 {true_out:.1f} s")
    print(f"     表中时长：直连 {tot['直连G01']:.1f} s，中继 {tot['中继']:.1f} s，中断 {tot.get('中断', 0):.1f} s")
    used = set(comm["中继架次编号"]) - {""}
    check("Q3 每个中继架次都被使用，且表中引用的架次均存在", used == set(relays), f"{sorted(used)}")
    return comm


# ---------------------------------------------------------------
# Q4
# ---------------------------------------------------------------
def check_q4(trips, relays, comm):
    print("\n==== 问题四 ====")
    cfg = read("Q4", "Q4_分区配置")
    uses = {tid: set(comm[(comm["运输架次编号"] == tid) & (comm["中继架次编号"] != "")]["中继架次编号"])
            for tid in trips}
    stock = {"A型运输无人机数": 4, "B型运输无人机数": 2, "C型运输无人机数": 2}
    for K in (2, 3):
        sub = cfg[cfg["K（2或3）"] == K]
        groups = {r["任务组编号"]: set(str(r["服务区列表"]).split(",")) for _, r in sub.iterrows()}
        allareas = [a for g in groups.values() for a in g]
        check(f"Q4 K={K} 每个服务区恰属一组、组数为 {K}、各组非空",
              sorted(allareas) == sorted(AREAS) and len(groups) == K and all(groups.values()))
        split = [tid for tid, x in trips.items() if len({g for g, A in groups.items() if set(x["areas"]) & A}) > 1]
        check(f"Q4 K={K} 同一运输架次涉及的服务区在同一组", not split, str(split))
        bad = []
        for _, r in sub.iterrows():
            A = groups[r["任务组编号"]]
            mine = [x for x in trips.values() if set(x["areas"]) <= A]
            rids = set().union(*[uses[t] for t, x in trips.items() if set(x["areas"]) <= A]) if mine else set()
            need = {}
            for g in "ABC":
                need[f"{g}型运输无人机数"] = max_overlap([(x["start"], x["end"]) for x in mine if x["g"] == g])
                need[f"{g}型电池组数"] = max_overlap([(x["start"], x["charge_end"]) for x in mine if x["g"] == g])
            need["中继无人机数"] = max_overlap([(relays[k]["launch"], relays[k]["ret"] + RELAY["turn"]) for k in rids])
            need["中继能源组件数"] = max_overlap([(relays[k]["launch"], relays[k]["charge_end"]) for k in rids])
            for k, v in need.items():
                if int(r[k]) != v:
                    bad.append((r["任务组编号"], k, int(r[k]), v))
        check(f"Q4 K={K} 各组资源数与按区间最大重叠独立重算一致", not bad, str(bad[:4]))
        tot = {k: int(sub[k].sum()) for k in cfg.columns[3:]}
        print(f"     K={K} 需求合计 {tot}")


if __name__ == "__main__":
    check_q1()
    print("\n==== 问题二 ====")
    check_transport("Q2")
    print("\n==== 问题三 ====")
    tr3 = check_transport("Q3")
    rel = check_relay()
    ms = max(max(x["end"] for x in tr3.values()), max(y["ret"] for y in rel.values()))
    print(f"     联合完成（含中继返航） {ms:.1f} s，中继能耗 {sum(y['e'] for y in rel.values()):.4f} kWh")
    cm = check_comm(tr3, rel)
    check_q4(tr3, rel, cm)
    n_bad = sum(not ok for _, ok, _ in REPORT)
    print(f"\n复核完成：{len(REPORT)} 项，失败 {n_bad} 项")
    pd.DataFrame(REPORT, columns=["检验内容", "是否通过", "说明"]).to_excel(
        os.path.join(RESULT_DIR, "独立复核报告.xlsx"), index=False)
    sys.exit(1 if n_bad else 0)
