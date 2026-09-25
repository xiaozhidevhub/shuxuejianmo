# -*- coding: utf-8 -*-
"""
vrp.py —— 问题二、问题三共用的“多点多架次调度”核心
=================================================
整体框架（两层）：
  外层：模拟退火（SA）决定“哪些货箱放在同一架次、用什么机型、按什么顺序访问服务区”
  内层：表调度（List Scheduling）把架次分给具体无人机和电池，算出开始时刻

一个“解”= 若干个架次组成的列表，每个架次是一个字典：
  {"g": 机型, "stops": [[服务区, [货箱编号...]], ...], "bias": 调度优先级微调量}

内层调度规则：
  1. 按“松弛时间”从小到大排序（越紧急越先排）：
     松弛时间 = min（货箱时限 - 该箱在架次内的送达偏移时间）+ bias
  2. 依次给每个架次选：同机型中最早空闲的无人机 + 同机型中最早充满的电池
     开始时刻 = max(无人机空闲时刻, 电池可用时刻, 通信约束时刻)
  3. 电池用完后要充到 100% 才能再用，充电时长由 SOC 决定（两阶段模型）
"""
import math
import random
from common import *

# 目标函数权重（可以在 q2.py / q3.py 中修改，用来做权衡分析）
WEIGHTS = {"hard": 1000.0,   # 硬时限每超 1 分钟的罚分（非常大，相当于硬约束）
           "tard": 1.0,      # 加权延误：优先系数 × 延误分钟
           "T": 2.0,         # 全部任务完成时间（分钟）
           "E": 5.0,         # 总能耗（kWh）
           "N": 3.0,         # 运输架次数
           "NR": 3.0,        # 中继架次数（问题三）
           "ER": 5.0}        # 中继能耗（问题三）

MAX_STOPS = 4               # 每个架次最多访问 4 个服务区（控制搜索规模）

# 问题三会把“通信模型”挂到这里；问题二时为 None（不考虑通信）
CTX = {"comm": None}


# ---------------------------------------------------------------
# 1. 评估一个架次：能否飞、耗多少电、每个货箱几秒送到
# ---------------------------------------------------------------
_TRIP_CACHE = {}


def trip_key(g, stops):
    return (g, tuple((s, tuple(sorted(ids))) for s, ids in stops))


def eval_trip(g, stops):
    """返回架次信息字典；不可行（超重/超体积/电量不足/通信中断）返回 None"""
    key = trip_key(g, stops)
    if key in _TRIP_CACHE:
        return _TRIP_CACHE[key]
    t = TYPES[g]
    all_ids = [i for _, ids in stops for i in ids]
    mass = sum(BOXES[i]["mass"] for i in all_ids)
    vol = sum(BOXES[i]["vol"] for i in all_ids)
    res = None
    if mass <= t["Q"] + 1e-9 and vol <= t["V"] + 1e-9:
        clock = t["prep"] + t["load"] * len(all_ids)   # 准备 + 逐箱装载
        q = mass                                       # 当前剩余载荷
        energy = 0.0
        cur = "O01"
        deliver = {}                                   # 货箱 → 送达偏移时刻
        legs, hovers = [], []                          # 记录每个航段/投送的时间，通信判定用
        for s, ids in stops:
            seg = node_segment(cur, s)
            energy += seg_energy(g, seg, q)
            legs.append((cur, s, clock))               # (起点, 终点, 航段起飞偏移时刻)
            clock += seg_time(g, seg)
            arrive = clock
            clock += t["hand_base"] + t["hand_box"] * len(ids)   # 交接时间
            hovers.append((s, arrive, clock))
            for i in ids:
                deliver[i] = clock                     # 交接完成即“交付完成”
            q = max(0.0, q - sum(BOXES[i]["mass"] for i in ids))
            cur = s
        seg = node_segment(cur, "O01")
        energy += seg_energy(g, seg, 0.0)              # 返程空载
        legs.append((cur, "O01", clock))
        clock += seg_time(g, seg)
        if energy <= (1 - t["rho"]) * t["E"] + 1e-9:   # 返航安全余量约束
            soc = 1 - energy / t["E"]
            res = {"g": g, "stops": [(s, list(ids)) for s, ids in stops], "ids": all_ids,
                   "mass": mass, "vol": vol, "E": energy, "dur": clock, "deliver": deliver,
                   "soc": soc, "charge": charge_time(soc, BATTERIES[g]["full"]),
                   "legs": legs, "hovers": hovers, "need": []}
            # 问题三：计算该架次哪些时段需要中继（相对时刻）
            if CTX["comm"] is not None:
                need, ok = CTX["comm"].trip_need(res)
                res["need"] = need
                if not ok:
                    res = None                         # 有无法保障通信的时段 → 不可行
            if res is not None:
                # 紧急程度：各箱“时限 - 送达偏移”的最小值（越小越紧急）
                res["slack"] = min((BOXES[i]["hard"] if BOXES[i]["hard"] else BOXES[i]["due"])
                                   - deliver[i] for i in all_ids)
    _TRIP_CACHE[key] = res
    return res


# ---------------------------------------------------------------
# 2. 内层：表调度（分配无人机、电池、开始时刻）
# ---------------------------------------------------------------
def schedule(sol, cover=None):
    """
    输入：解 sol（架次列表）；cover 为问题三的“中继窗口”对象（问题二时为 None）
    输出：plan 列表，每项含架次信息 + 无人机 + 电池 + 开始时刻；若某架次不可行返回 None
    """
    evals = []
    for idx, tr in enumerate(sol):
        ev = eval_trip(tr["g"], tr["stops"])
        if ev is None:
            return None
        evals.append((ev["slack"] + tr["bias"], idx, ev))
    evals.sort(key=lambda x: (x[0], x[1]))            # 最紧急的先排

    drone_free = {u: 0.0 for u, g in DRONES}           # 每架无人机的空闲时刻
    batt_ready = {}                                    # 每块电池充满可用的时刻
    for g, info in BATTERIES.items():
        for k in range(info["count"]):
            batt_ready[f"{g}-B{k + 1:02d}"] = 0.0
    plan = []
    for _, idx, ev in evals:
        g = ev["g"]
        # 同机型中最早空闲的无人机（编号小者优先）
        u = min((x for x, gg in DRONES if gg == g), key=lambda x: (drone_free[x], x))
        # 同机型中最早可用的电池
        b = min((x for x in batt_ready if x.startswith(g + "-")), key=lambda x: (batt_ready[x], x))
        start = max(drone_free[u], batt_ready[b])
        lost = 0.0
        # 通信约束（问题三）：推迟到“需要中继的时段都落在中继服务窗口内”的最早时刻
        if cover is not None and ev["need"]:
            start, lost = cover.earliest(ev, start)
        end = start + ev["dur"]
        drone_free[u] = end
        batt_ready[b] = end + ev["charge"]             # 回来后立即开始充电
        plan.append({"idx": idx, "ev": ev, "uav": u, "batt": b, "start": start, "end": end,
                     "charge_end": end + ev["charge"], "lost": lost})
    plan.sort(key=lambda p: (p["start"], p["uav"]))
    return plan


# ---------------------------------------------------------------
# 3. 目标函数
# ---------------------------------------------------------------
def evaluate(sol, detail=False, cover=None):
    plan = schedule(sol, cover)
    if plan is None:
        return (float("inf"), None) if detail else float("inf")
    hard_v = tard = 0.0
    on_time = 0
    for p in plan:
        for i, off in p["ev"]["deliver"].items():
            tt = p["start"] + off
            b = BOXES[i]
            if b["hard"] is not None:
                hard_v += max(0.0, tt - b["hard"])
            late = max(0.0, tt - b["due"])
            tard += b["coef"] * late / 60.0
            on_time += (late <= 1e-6)
    makespan = max(p["end"] for p in plan)
    energy = sum(p["ev"]["E"] for p in plan)
    W = WEIGHTS
    cost = (W["hard"] * hard_v / 60.0 + W["tard"] * tard + W["T"] * makespan / 60.0
            + W["E"] * energy + W["N"] * len(plan))
    relay = None
    if cover is not None:                              # 问题三：加上中继计划的代价
        relay = cover.finalize(plan)
        lost = sum(p["lost"] for p in plan) + relay["module_short"]
        cost += (W["hard"] * lost / 60.0 + W["ER"] * relay["energy"]
                 + W["NR"] * len(relay["sorties"]))
        makespan_joint = max(makespan, relay["last_return"])
        cost += W["T"] * (makespan_joint - makespan) / 60.0
    if not detail:
        return cost
    info = {"cost": cost, "hard_violation_s": hard_v, "weighted_tardiness": tard,
            "on_time": on_time, "makespan": makespan, "energy": energy, "trips": len(plan),
            "plan": plan, "relay": relay}
    return cost, info


# ---------------------------------------------------------------
# 4. 邻域操作（每次随机改动一点点，得到一个“邻居解”）
# ---------------------------------------------------------------
def new_trip(g, stops, bias=0.0):
    return {"g": g, "stops": [[s, list(ids)] for s, ids in stops], "bias": bias}


def trip_ok(tr):
    return len(tr["stops"]) <= MAX_STOPS and eval_trip(tr["g"], tr["stops"]) is not None


def add_box(tr, box_id, pos=None):
    """把货箱放进架次 tr：该服务区已在路线上就并入，否则作为新站点插入"""
    s = BOXES[box_id]["area"]
    for st in tr["stops"]:
        if st[0] == s:
            st[1].append(box_id)
            return
    if pos is None:
        pos = random.randint(0, len(tr["stops"]))
    tr["stops"].insert(pos, [s, [box_id]])


def remove_box(tr, box_id):
    for st in tr["stops"]:
        if box_id in st[1]:
            st[1].remove(box_id)
    tr["stops"] = [st for st in tr["stops"] if st[1]]


def clone(tr):
    return new_trip(tr["g"], tr["stops"], tr["bias"])


def neighbor(sol):
    """随机选一种操作生成新解；不可行就返回 None"""
    sol = list(sol)                                    # 浅拷贝列表，被改的架次再单独复制
    n = len(sol)
    op = random.random()
    i = random.randrange(n)
    if op < 0.30:                                      # ① 移动一个货箱到别的架次/新架次
        a = clone(sol[i])
        box = random.choice([x for st in a["stops"] for x in st[1]])
        remove_box(a, box)
        if a["stops"] and not trip_ok(a):
            return None
        if random.random() < 0.15 or n == 1:           # 放进一个新开的架次
            b = new_trip(random.choice("ABC"), [])
            add_box(b, box)
            if not trip_ok(b):
                return None
            sol[i] = a
            sol.append(b)
        else:                                          # 放进另一个已有架次
            j = random.randrange(n)
            if j == i:
                return None
            b = clone(sol[j])
            add_box(b, box)
            if not trip_ok(b):
                return None
            sol[i], sol[j] = a, b
    elif op < 0.45:                                    # ② 交换两个架次中的各一个货箱
        j = random.randrange(n)
        if i == j:
            return None
        a, b = clone(sol[i]), clone(sol[j])
        x = random.choice([x for st in a["stops"] for x in st[1]])
        y = random.choice([y for st in b["stops"] for y in st[1]])
        if BOXES[x]["kind"] == BOXES[y]["kind"] and BOXES[x]["area"] == BOXES[y]["area"]:
            return None                                # 完全等价的交换没意义
        remove_box(a, x); remove_box(b, y); add_box(a, y); add_box(b, x)
        if not (trip_ok(a) and trip_ok(b)):
            return None
        sol[i], sol[j] = a, b
    elif op < 0.55:                                    # ③ 更换架次机型
        a = clone(sol[i])
        a["g"] = random.choice([g for g in "ABC" if g != a["g"]])
        if not trip_ok(a):
            return None
        sol[i] = a
    elif op < 0.65:                                    # ④ 调整访问顺序（交换两个站点）
        a = clone(sol[i])
        if len(a["stops"]) < 2:
            return None
        p, q = random.sample(range(len(a["stops"])), 2)
        a["stops"][p], a["stops"][q] = a["stops"][q], a["stops"][p]
        if not trip_ok(a):
            return None
        sol[i] = a
    elif op < 0.75:                                    # ⑤ 合并两个架次
        j = random.randrange(n)
        if i == j:
            return None
        a = clone(sol[i])
        for s, ids in sol[j]["stops"]:
            for x in ids:
                add_box(a, x, pos=len(a["stops"]))
        a["g"] = random.choice([sol[i]["g"], sol[j]["g"], "C"])
        if not trip_ok(a):
            return None
        sol[i] = a
        sol.pop(j)
    elif op < 0.83:                                    # ⑥ 拆分一个架次
        a = clone(sol[i])
        ids = [x for st in a["stops"] for x in st[1]]
        if len(ids) < 2:
            return None
        part = random.sample(ids, random.randint(1, len(ids) - 1))
        b = new_trip(random.choice("ABC"), [])
        for x in part:
            remove_box(a, x); add_box(b, x)
        if not (trip_ok(a) and trip_ok(b)):
            return None
        sol[i] = a
        sol.append(b)
    elif op < 0.92:                                    # ⑦ 把一整个站点移到另一个架次
        j = random.randrange(n)
        if i == j:
            return None
        a, b = clone(sol[i]), clone(sol[j])
        k = random.randrange(len(a["stops"]))
        s, ids = a["stops"].pop(k)
        for x in ids:
            add_box(b, x)
        if not trip_ok(b) or (a["stops"] and not trip_ok(a)):
            return None
        sol[j] = b
        if a["stops"]:
            sol[i] = a
        else:
            sol.pop(i)
    else:                                              # ⑧ 微调调度优先级
        a = clone(sol[i])
        a["bias"] += random.gauss(0, 900)
        sol[i] = a
    return [t for t in sol if t["stops"]]


# ---------------------------------------------------------------
# 5. 模拟退火主循环
# ---------------------------------------------------------------
def anneal(sol, iters=40000, T0=40.0, T_end=0.02, seed=0, verbose=True, nb=None, ev=None):
    """nb / ev：邻域函数和目标函数，默认用本文件的 neighbor / evaluate（问题三会换成自己的）"""
    nb = nb or neighbor
    ev = ev or evaluate
    random.seed(seed)
    cur, cur_cost = sol, ev(sol)
    best, best_cost = cur, cur_cost
    for it in range(iters):
        T = T0 * (T_end / T0) ** (it / iters)          # 温度按指数规律逐步降低
        cand = nb(cur)
        if cand is None:
            continue
        c = ev(cand)
        # Metropolis 准则：更好就接受；更差也以一定概率接受（跳出局部最优）
        if c < cur_cost or random.random() < math.exp(-(c - cur_cost) / T):
            cur, cur_cost = cand, c
            if c < best_cost - 1e-9:
                best, best_cost = cand, c
        if verbose and it % 50000 == 0:
            print(f"   迭代 {it:6d}  温度 {T:8.3f}  当前 {cur_cost:10.2f}  最优 {best_cost:10.2f}")
    return best, best_cost


def local_polish(sol, rounds=3, nb=None, ev=None):
    """退火后再做一轮“只接受改进”的爬山，进一步修正"""
    nb = nb or neighbor
    ev = ev or evaluate
    best, best_cost = sol, ev(sol)
    for _ in range(rounds):
        improved = False
        for _ in range(6000):
            cand = nb(best)
            if cand is None:
                continue
            c = ev(cand)
            if c < best_cost - 1e-9:
                best, best_cost, improved = cand, c, True
        if not improved:
            break
    return best, best_cost


# ---------------------------------------------------------------
# 6. 初始解：用问题一的单点最优组批（每个服务区单独成批）
# ---------------------------------------------------------------
def initial_from_q1():
    from q1 import optimize_area
    sol = []
    for s in AREAS:
        plan, info = optimize_area(s, TYPES["A"]["rho"], "NET")
        ptr = {k: 0 for k in info}
        for p in sorted(plan, key=lambda p: (-p["combo"][0], -p["mass"])):
            ids = []
            for c, k in zip(p["combo"], p["kinds"]):
                ids += info[k]["ids"][ptr[k]: ptr[k] + c]
                ptr[k] += c
            sol.append(new_trip(p["g"], [[s, ids]]))
    return sol


def check_plan(plan, relay=None):
    """资源可行性检验：返回检验项列表 [(检验内容, 是否通过, 说明)]"""
    checks = []
    # (1) 每箱恰好送一次
    cnt = {}
    for p in plan:
        for i in p["ev"]["ids"]:
            cnt[i] = cnt.get(i, 0) + 1
    ok = set(cnt) == set(BOXES) and all(v == 1 for v in cnt.values())
    checks.append(("80个货箱全部交付且每箱只安排一次", ok, f"已交付 {len(cnt)} 箱"))
    # (2) 载重、体积、能量
    ok = all(p["ev"]["mass"] <= TYPES[p["ev"]["g"]]["Q"] + 1e-9 and
             p["ev"]["vol"] <= TYPES[p["ev"]["g"]]["V"] + 1e-9 and
             p["ev"]["soc"] >= TYPES[p["ev"]["g"]]["rho"] - 1e-9 for p in plan)
    checks.append(("载质量、装载体积、返航SOC≥20%", ok, f"最低返航SOC {min(p['ev']['soc'] for p in plan) * 100:.2f}%"))
    # (3) 无人机、电池时间不重叠；电池只给本机型用
    ok = True
    for key in ("uav", "batt"):
        groups = {}
        for p in plan:
            groups.setdefault(p[key], []).append(p)
        for k, ps in groups.items():
            ps.sort(key=lambda p: p["start"])
            for a, b in zip(ps, ps[1:]):
                busy_end = a["end"] if key == "uav" else a["charge_end"]
                if b["start"] < busy_end - 1e-6:
                    ok = False
            if key == "batt" and any(p["ev"]["g"] != k[0] for p in ps):
                ok = False
            if key == "uav" and any(p["ev"]["g"] != dict(DRONES)[k] for p in ps):
                ok = False
    checks.append(("无人机任务不重叠、电池占用+充电不重叠、机型不混用", ok,
                   f"使用无人机 {len({p['uav'] for p in plan})} 架，电池 {len({p['batt'] for p in plan})} 组"))
    # (4) 硬时限
    viol = [(i, p["start"] + off) for p in plan for i, off in p["ev"]["deliver"].items()
            if BOXES[i]["hard"] is not None and p["start"] + off > BOXES[i]["hard"] + 1e-6]
    checks.append(("医疗物资期望时间、首批截止时间", len(viol) == 0, f"违反 {len(viol)} 箱"))
    return checks
