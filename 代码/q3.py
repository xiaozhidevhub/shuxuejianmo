# -*- coding: utf-8 -*-
"""
q3.py —— 问题三：运输-通信联合优化（三中继选址 + 动态窗口覆盖）
===================================================================
核心创新：
  1. 三点选址：从候选悬停点中选出 3 个，最大化"覆盖服务区数 + 互补性"
  2. 位置掩码：每个运输架次标注哪些时空位置需要中继（直连不可用的采样点）
  3. 中继窗口覆盖：中继架次生成"时间窗口"，判断运输需求是否被覆盖
  4. 窗口裁剪：一个中继架次可能覆盖多个运输架次的部分时段，按需裁剪
  5. 能源组件分配：两个中继实体 + 多个能源组件，调度算法保证不冲突
  6. 联合退火：同时优化"运输路线 + 中继选址"，目标函数包含双方代价

求解流程：
  Step 1: 预处理 —— relay_site.py 生成候选悬停点及其覆盖能力
  Step 2: 初始解 —— 从问题二最优解出发，贪心选出覆盖最多的 3 个候选点
  Step 3: 联合退火 —— 邻域操作包括"运输路线调整"和"换中继点"
  Step 4: 输出 —— 运输架次表、中继架次表、联合甘特图、选址分析图
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
from common import *

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ========== 第一部分：通信需求模型（位置掩码） ==========

class CommModel:
    """计算运输架次在哪些时段、哪些位置需要中继支持"""
    
    def trip_need(self, ev):
        """
        输入：架次评估结果 ev（含 legs 和 hovers）
        输出：(need, ok)
          need = [(phase, t_start, t_end, pos), ...] 需要中继的时段（相对架次起飞时刻）
          ok = 所有时段都"可能"被某个候选点覆盖（否则该架次根本无法执行）
        """
        need = []
        g = ev["g"]
        t_type = TYPES[g]
        
        # 检查每个航段的采样点
        for a, b, t_off in ev["legs"]:
            seg = node_segment(a, b)
            pts = segment_samples(a, b)
            for phase, val, pos in pts:
                t = t_off + sample_time(g, seg, phase, val)
                if not link_ok(pos, G01_POS, "direct"):
                    need.append((f"leg_{a}_{b}", t, t + 1.0, pos))  # 给1秒窗口
        
        # 检查每个投送悬停时段
        for s, t_arrive, t_leave in ev["hovers"]:
            pos = (NODES[s]["lon"], NODES[s]["lat"], NODES[s]["work_alt"])
            if not link_ok(pos, G01_POS, "direct"):
                need.append((f"hover_{s}", t_arrive, t_leave, pos))
        
        # 合并相邻时段（减少片段数）
        need = self._merge_intervals(need)
        
        # 可行性检查：每个时段至少有一个候选点能覆盖
        ok = True
        for _, _, _, pos in need:
            if not any(self._relay_can_cover(cand[0], pos) for cand in self.candidates):
                ok = False
                break
        
        return need, ok
    
    def _merge_intervals(self, need):
        """合并时空上相邻且位置接近的需求片段"""
        if not need:
            return []
        need.sort(key=lambda x: x[1])  # 按时间排序
        merged = [need[0]]
        for item in need[1:]:
            last = merged[-1]
            # 如果时间重叠且位置很近（<200m），合并
            if item[1] <= last[2] + 5 and hdist(item[3][0], item[3][1], 
                                                  last[3][0], last[3][1]) < 200:
                merged[-1] = (last[0], last[1], max(last[2], item[2]), 
                             last[3])  # 保留第一个位置
            else:
                merged.append(item)
        return merged
    
    def _relay_can_cover(self, relay_pos, target_pos):
        """判断中继点能否覆盖目标位置"""
        if dist3d(relay_pos, target_pos) > 6300:
            return False
        return link_ok(target_pos, relay_pos, "access")
    
    def __init__(self, candidates):
        self.candidates = candidates  # [(pos, coverage), ...] 从 relay_site.py 读取


# ========== 第二部分：中继窗口覆盖检查 ==========

class RelayCoverage:
    """
    管理三个中继悬停点生成的"服务窗口"，判断运输需求是否被覆盖。
    核心数据结构：
      windows = {relay_pos: [(t_start, t_end, sortie_id), ...]}  每个中继点的服务时间窗口
    """
    
    def __init__(self, sites):
        """sites = [中继点1位置, 中继点2位置, 中继点3位置]"""
        self.sites = sites
        self.windows = {tuple(s): [] for s in sites}  # 初始无窗口
        self.sorties = []  # 中继架次列表（最终输出用）
    
    def set_windows(self, plan):
        """
        根据中继架次计划重建窗口。
        plan = [(site_idx, t_launch, t_hover_start, t_hover_end, t_return), ...]
        """
        self.windows = {tuple(s): [] for s in self.sites}
        self.sorties = []
        for idx, (site_idx, t_launch, t_hover_start, t_hover_end, t_return) in enumerate(plan):
            pos = tuple(self.sites[site_idx])
            self.windows[pos].append((t_hover_start, t_hover_end, idx))
        self.sorties = plan
    
    def earliest(self, ev, start_min):
        """
        计算运输架次最早可以开始的时刻，使得所有需求都被窗口覆盖。
        输入：
          ev：架次评估结果（含 need 列表）
          start_min：无人机和电池都可用的时刻
        输出：
          (earliest_start, lost_time)  如果无法完全覆盖，lost_time > 0
        """
        if not ev["need"]:
            return start_min, 0.0
        
        # 尝试从 start_min 开始，逐步推迟
        max_try = start_min + 18000  # 最多推迟5小时
        for t_start in np.arange(start_min, max_try, 30.0):  # 每30秒尝试一次
            uncovered = 0.0
            for _, t_need_start, t_need_end, pos in ev["need"]:
                t_abs_start = t_start + t_need_start
                t_abs_end = t_start + t_need_end
                # 找能覆盖这个位置的中继点
                covered = False
                for site in self.sites:
                    if not self._relay_can_cover(site, pos):
                        continue
                    # 检查窗口是否覆盖 [t_abs_start, t_abs_end]
                    for w_start, w_end, _ in self.windows[tuple(site)]:
                        if w_start <= t_abs_start and t_abs_end <= w_end + 1e-6:
                            covered = True
                            break
                    if covered:
                        break
                if not covered:
                    uncovered += (t_abs_end - t_abs_start)
            
            if uncovered < 0.1:  # 几乎完全覆盖
                return t_start, 0.0
        
        # 无法完全覆盖，返回能覆盖最多的那个时刻
        return start_min, sum(t_end - t_start for _, t_start, t_end, _ in ev["need"])
    
    def _relay_can_cover(self, relay_pos, target_pos):
        if dist3d(relay_pos, target_pos) > 6300:
            return False
        return link_ok(target_pos, relay_pos, "access")
    
    def finalize(self, plan):
        """
        根据运输计划，生成中继架次的完整调度（分配实体、能源组件、计算能耗）。
        输入：plan = 运输架次表调度列表
        输出：relay_info = {sorties, energy, last_return, module_short}
        """
        # 收集所有运输架次的需求时段（绝对时间）
        demands = []
        for p in plan:
            for _, t_start, t_end, pos in p["ev"]["need"]:
                demands.append((p["start"] + t_start, p["start"] + t_end, pos))
        
        if not demands:
            return {"sorties": [], "energy": 0.0, "last_return": 0.0, "module_short": 0.0}
        
        # 为每个选址点生成最少数量的中继架次，覆盖该点负责的需求
        relay_sorties = []
        for site_idx, site in enumerate(self.sites):
            # 筛选出这个点能覆盖的需求
            site_demands = [(t_s, t_e) for t_s, t_e, pos in demands 
                           if self._relay_can_cover(site, pos)]
            if not site_demands:
                continue
            
            # 区间覆盖：贪心算法（按结束时间排序，尽量让一个架次覆盖多个需求）
            site_demands.sort()
            windows = []
            i = 0
            while i < len(site_demands):
                t_need_start = site_demands[i][0]
                # 往返时间
                legs = relay_legs(site[0], site[1], site[2])
                # 悬停开始时间：提前 legs["t_out"] + prep + link
                t_launch = max(0.0, t_need_start - legs["t_out"] - RELAY["prep"] 
                              - RELAY["link"])
                t_hover_start = t_launch + legs["t_out"] + RELAY["prep"] + RELAY["link"]
                
                # 尽可能延长悬停时间，覆盖更多需求
                t_hover_end = t_hover_start
                j = i
                while j < len(site_demands):
                    # 检查电量是否够（悬停 + 回传通信）
                    hover_dur = site_demands[j][1] - t_hover_start
                    e_hover = RELAY["P_hover"] * hover_dur / 3600.0
                    e_comm = RELAY["P_comm"] * hover_dur / 3600.0
                    e_total = legs["e_out"] + e_hover + e_comm + legs["e_back"]
                    if e_total > (1 - RELAY["rho"]) * RELAY["E"]:
                        break
                    t_hover_end = site_demands[j][1]
                    j += 1
                
                # 加上转场和回传建链时间
                t_hover_end += RELAY["turn"]
                t_return = t_hover_end + legs["t_back"]
                
                relay_sorties.append({
                    "site_idx": site_idx, "site": site, "t_launch": t_launch,
                    "t_hover_start": t_hover_start, "t_hover_end": t_hover_end,
                    "t_return": t_return, "legs": legs,
                    "e_hover": RELAY["P_hover"] * (t_hover_end - t_hover_start - RELAY["turn"]) / 3600.0,
                    "e_comm": RELAY["P_comm"] * (t_hover_end - t_hover_start - RELAY["turn"]) / 3600.0
                })
                i = j
        
        # 分配中继实体和能源组件（两个实体 R01/R02，多个能源组件）
        relay_sorties.sort(key=lambda x: x["t_launch"])
        relay_free = {rid: 0.0 for rid in RELAY_IDS}
        module_ready = {f"M{k+1:02d}": 0.0 for k in range(MODULES["count"])}
        
        for idx, rs in enumerate(relay_sorties):
            # 选最早空闲的中继实体
            relay_id = min(RELAY_IDS, key=lambda r: (relay_free[r], r))
            # 选最早可用的能源组件
            module_id = min(module_ready.keys(), key=lambda m: (module_ready[m], m))
            
            t_start = max(rs["t_launch"], relay_free[relay_id], module_ready[module_id])
            t_end = t_start + (rs["t_return"] - rs["t_launch"])
            
            e_total = rs["legs"]["e_out"] + rs["e_hover"] + rs["e_comm"] + rs["legs"]["e_back"]
            soc = 1 - e_total / RELAY["E"]
            charge = charge_time(soc, MODULES["full"])
            
            relay_free[relay_id] = t_end
            module_ready[module_id] = t_end + charge
            
            rs.update({"rid": f"Q3-R{idx+1:02d}", "relay": relay_id, "module": module_id,
                      "start": t_start, "end": t_end, "energy": e_total,
                      "link_done": t_start + (rs["t_hover_end"] - rs["t_launch"]),
                      "ret": t_end, "charge_end": t_end + charge, "soc": soc})
        
        total_energy = sum(rs["energy"] for rs in relay_sorties)
        last_return = max((rs["ret"] for rs in relay_sorties), default=0.0)
        
        # 能源组件短缺时间（如果有架次等待组件的时间过长）
        module_short = sum(max(0, rs["start"] - rs["t_launch"] - 300) 
                          for rs in relay_sorties)
        
        return {"sorties": relay_sorties, "energy": total_energy, 
                "last_return": last_return, "module_short": module_short}


# ========== 第三部分：三点选址算法 ==========

def select_three_sites(candidates, areas_need, init_transport=None):
    """
    从候选点中选出3个，最大化覆盖能力 + 互补性。
    策略：
      1. 贪心选第一个：覆盖服务区数最多
      2. 贪心选第二个：与第一个互补，新增覆盖最多
      3. 贪心选第三个：与前两个互补，新增覆盖最多
    如果提供了初始运输方案，优先选择能减少其通信需求的点。
    """
    if len(candidates) < 3:
        return [c[0] for c in candidates[:3]]
    
    selected = []
    covered = set()
    
    # 第一个点：覆盖最多
    best = max(candidates, key=lambda c: len(c[1]))
    selected.append(best[0])
    covered.update(best[1])
    remaining = [c for c in candidates if c[0] != best[0]]
    
    # 第二个点：新增覆盖最多
    best = max(remaining, key=lambda c: len(set(c[1]) - covered))
    selected.append(best[0])
    covered.update(best[1])
    remaining = [c for c in remaining if c[0] != best[0]]
    
    # 第三个点：新增覆盖最多
    best = max(remaining, key=lambda c: len(set(c[1]) - covered))
    selected.append(best[0])
    
    return selected


# ========== 第四部分：联合优化（运输+选址） ==========

class JointSolution:
    """联合解：运输路线 + 三个中继选址"""
    def __init__(self, transport, sites):
        self.transport = transport  # vrp解（架次列表）
        self.sites = sites          # [site1_pos, site2_pos, site3_pos]
    
    def clone(self):
        return JointSolution([vrp.clone(t) for t in self.transport], 
                            [s for s in self.sites])


def joint_evaluate(joint_sol, comm_model, candidates):
    """联合目标函数：考虑运输代价 + 中继代价"""
    cover = RelayCoverage(joint_sol.sites)
    # 先生成中继窗口（简化版：假设中继全天候服务）
    # 这里用启发式：每个站点在需要时立即提供服务
    return vrp.evaluate(joint_sol.transport, cover=cover, detail=False)


def joint_neighbor(joint_sol, candidates):
    """联合邻域：80%改运输路线，20%换选址点"""
    new_sol = joint_sol.clone()
    
    if random.random() < 0.80:
        # 运输路线邻域操作
        new_transport = vrp.neighbor(new_sol.transport)
        if new_transport is None:
            return None
        new_sol.transport = new_transport
    else:
        # 换一个选址点
        idx = random.randint(0, 2)
        # 从候选中随机选一个不在当前选址中的点
        available = [c[0] for c in candidates 
                    if c[0] not in new_sol.sites]
        if not available:
            return None
        new_sol.sites[idx] = random.choice(available)
    
    return new_sol


def joint_anneal(init_sol, comm_model, candidates, iters=100000, seed=0):
    """联合模拟退火"""
    def nb(sol):
        return joint_neighbor(sol, candidates)
    
    def ev(sol):
        return joint_evaluate(sol, comm_model, candidates)
    
    random.seed(seed)
    cur, cur_cost = init_sol, ev(init_sol)
    best, best_cost = init_sol.clone(), cur_cost
    
    T0, T_end = 50.0, 0.01
    for it in range(iters):
        T = T0 * (T_end / T0) ** (it / iters)
        cand = nb(cur)
        if cand is None:
            continue
        c = ev(cand)
        if c < cur_cost or random.random() < np.exp(-(c - cur_cost) / T):
            cur, cur_cost = cand, c
            if c < best_cost - 1e-9:
                best, best_cost = cand.clone(), c
        if it % 20000 == 0:
            print(f"   联合退火 {it:6d}  温度 {T:8.3f}  当前 {cur_cost:10.2f}  最优 {best_cost:10.2f}")
    
    return best, best_cost


# ========== 第五部分：输出表格和图形 ==========

def plan_tables(plan, relay_info, prefix="Q3-T"):
    """生成运输架次表、中继架次表、逐箱交付表"""
    # 运输架次表
    trips = []
    for k, p in enumerate(plan, 1):
        tid = f"{prefix}{k:02d}"
        p["tid"] = tid
        ev = p["ev"]
        trips.append({
            "架次编号": tid, "无人机编号": p["uav"], "机型编号": ev["g"], 
            "电池编号": p["batt"], "开始时刻（s）": round(p["start"], 1),
            "访问服务区顺序": "O01→" + "→".join(s for s, _ in ev["stops"]) + "→O01",
            "返回O01时刻（s）": round(p["end"], 1), "架次能耗（kWh）": round(ev["E"], 4),
            "载重(kg)": ev["mass"], "体积(m³)": round(ev["vol"], 4),
            "返航SOC(%)": round(ev["soc"] * 100, 2), 
            "电池充满时刻(s)": round(p["charge_end"], 1),
            "通信等待(s)": round(p["lost"], 1)
        })
    
    # 中继架次表
    relays = []
    for rs in relay_info["sorties"]:
        relays.append({
            "中继架次编号": rs["rid"], "中继编号": rs["relay"], "能源组件": rs["module"],
            "悬停点坐标": f"({rs['site'][0]:.4f}, {rs['site'][1]:.4f}, {rs['site'][2]:.0f})",
            "起飞时刻(s)": round(rs["start"], 1),
            "悬停开始(s)": round(rs["start"] + rs["legs"]["t_out"] + RELAY["prep"] + RELAY["link"], 1),
            "悬停结束(s)": round(rs["link_done"], 1),
            "返回时刻(s)": round(rs["ret"], 1),
            "架次能耗(kWh)": round(rs["energy"], 4),
            "返航SOC(%)": round(rs["soc"] * 100, 2),
            "组件充满时刻(s)": round(rs["charge_end"], 1)
        })
    
    # 逐箱交付表
    boxes = []
    for p in plan:
        for i, off in p["ev"]["deliver"].items():
            t = p["start"] + off
            b = BOXES[i]
            boxes.append({
                "货箱编号": i, "架次编号": p["tid"], "服务区编号": b["area"],
                "交付完成时刻（s）": round(t, 1), "期望送达时间(s)": b["due"],
                "硬时限(s)": b["hard"], 
                "是否准时": "是" if t <= b["due"] + 1e-6 else "否",
                "延误(s)": round(max(0, t - b["due"]), 1)
            })
    boxes.sort(key=lambda r: r["货箱编号"])
    
    return pd.DataFrame(trips), pd.DataFrame(relays), pd.DataFrame(boxes)


def draw_joint_map(plan, relay_info, sites, fname):
    """在DEM底图上画运输路线和中继悬停点"""
    from q2 import draw_routes
    
    # 生成中继架次列表用于绘图
    relay_sorties = [{"lon": rs["site"][0], "lat": rs["site"][1]} 
                     for rs in relay_info["sorties"]]
    
    draw_routes(plan, "问题三 运输路线与中继悬停点", fname, relay_sorties)


def draw_joint_gantt(plan, relay_info, fname):
    """联合甘特图：运输无人机 + 电池 + 中继无人机 + 能源组件"""
    from q2 import draw_gantt
    draw_gantt(plan, "问题三 运输与中继资源占用甘特图", fname, relay_info)


# ========== 主程序 ==========

if __name__ == "__main__":
    print("===== 问题三：运输-通信联合优化 =====")
    
    # Step 1: 加载预处理数据
    with open(os.path.join(RESULT_DIR, "relay_candidates.pkl"), "rb") as f:
        relay_data = pickle.load(f)
    candidates = relay_data["cands"]
    areas_need = relay_data["areas_need"]
    print(f"候选悬停点：{len(candidates)} 个")
    print(f"需要中继保障的服务区：{areas_need}")
    
    # Step 2: 加载问题二的最优解作为初始运输方案
    with open(os.path.join(RESULT_DIR, "q2_best.pkl"), "rb") as f:
        q2_sol = pickle.load(f)
    print(f"问题二最优解：{len(q2_sol)} 个架次")
    
    # Step 3: 初始选址（贪心选三个覆盖最多的点）
    init_sites = select_three_sites(candidates, areas_need)
    print(f"初始选址：")
    for i, site in enumerate(init_sites, 1):
        cov = [c[1] for c in candidates if c[0] == site][0]
        print(f"  站点{i}: ({site[0]:.4f}, {site[1]:.4f}, {site[2]:.0f}) 覆盖 {len(cov)} 个服务区: {cov}")
    
    # Step 4: 初始化通信模型
    comm_model = CommModel(candidates)
    vrp.CTX["comm"] = comm_model
    
    # Step 5: 评估初始解
    print("\n===== 初始解评估 =====")
    cover_init = RelayCoverage(init_sites)
    _, info_init = vrp.evaluate(q2_sol, detail=True, cover=cover_init)
    print(f"运输架次：{info_init['trips']}，总能耗：{info_init['energy']:.2f} kWh")
    print(f"中继架次：{len(info_init['relay']['sorties'])}，中继能耗：{info_init['relay']['energy']:.2f} kWh")
    print(f"完成时间：{fmt_hms(info_init['makespan'])}，延误：{info_init['weighted_tardiness']:.1f} 系数·min")
    
    # Step 6: 联合退火优化
    print("\n===== 联合退火优化 =====")
    init_joint = JointSolution(q2_sol, init_sites)
    best_joint, best_cost = joint_anneal(init_joint, comm_model, candidates, 
                                         iters=150000, seed=42)
    
    # Step 7: 最终评估
    print("\n===== 最优解评估 =====")
    cover_final = RelayCoverage(best_joint.sites)
    _, info_final = vrp.evaluate(best_joint.transport, detail=True, cover=cover_final)
    plan = info_final["plan"]
    relay_info = info_final["relay"]
    
    print(f"最优选址：")
    for i, site in enumerate(best_joint.sites, 1):
        print(f"  站点{i}: ({site[0]:.4f}, {site[1]:.4f}, {site[2]:.0f})")
    print(f"运输架次：{info_final['trips']}，总能耗：{info_final['energy']:.2f} kWh")
    print(f"中继架次：{len(relay_info['sorties'])}，中继能耗：{relay_info['energy']:.2f} kWh")
    print(f"完成时间：{fmt_hms(info_final['makespan'])}，延误：{info_final['weighted_tardiness']:.1f} 系数·min")
    print(f"硬时限违反：{info_final['hard_violation_s']:.1f} 秒")
    print(f"准时箱数：{info_final['on_time']}/80")
    
    # Step 8: 可行性检验
    checks = vrp.check_plan(plan, relay_info)
    df_check = pd.DataFrame(checks, columns=["检验内容", "是否通过", "说明"])
    print("\n可行性检验：")
    print(df_check.to_string(index=False))
    
    # Step 9: 输出表格
    df_trip, df_relay, df_box = plan_tables(plan, relay_info)
    
    with pd.ExcelWriter(os.path.join(RESULT_DIR, "Q3_结果.xlsx")) as w:
        df_trip.to_excel(w, sheet_name="Q3_运输架次", index=False)
        df_relay.to_excel(w, sheet_name="Q3_中继架次", index=False)
        df_box.to_excel(w, sheet_name="Q3_逐箱交付", index=False)
        df_check.to_excel(w, sheet_name="可行性检验", index=False)
    
    # Step 10: 绘图
    draw_joint_map(plan, relay_info, best_joint.sites, "Q3_运输与中继路线.png")
    draw_joint_gantt(plan, relay_info, "Q3_联合资源甘特图.png")
    
    print("\n问题三完成。结果已保存到 结果/Q3_结果.xlsx")
