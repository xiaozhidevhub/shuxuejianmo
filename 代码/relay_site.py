# -*- coding: utf-8 -*-
"""
relay_site.py —— 中继悬停点候选分析（问题三的预处理）
===================================================
对每个候选悬停点 P（DEM 范围内，离地 300 m，且 P 与 G01 回传可用），
判断它能否“完整保障”服务区 s 的单点往返：O01→s 航段、s 点投送悬停、s→O01 航段中
所有直连不可用的采样点，P 都能提供接入链路。
结果：cover[P] = 能完整保障的服务区集合。保存到 结果/relay_candidates.pkl
"""
import pickle
import time
import numpy as np
from common import *


def direct_ok(pos):
    return link_ok(pos, G01_POS, "direct")


def relay_covers(relay_pos, pos):
    if dist3d(relay_pos, pos) > 6300:
        return False
    return link_ok(pos, relay_pos, "access")


if __name__ == "__main__":
    t0 = time.time()
    # 1. 每个服务区单点往返中“直连不可用”的点，按离服务区由近到远排序（近处最难，先检查可提前淘汰）
    need_pts = {}
    for s in AREAS:
        pts = [pos for _, _, pos in segment_samples("O01", s) + segment_samples(s, "O01")]
        pts.append((NODES[s]["lon"], NODES[s]["lat"], NODES[s]["work_alt"]))
        bad = [p for p in pts if not direct_ok(p)]
        bad.sort(key=lambda p: hdist(p[0], p[1], NODES[s]["lon"], NODES[s]["lat"]))
        need_pts[s] = bad
        print(f"{s}: 往返+投送 直连不可用点 {len(bad)}")
    areas_need = [s for s in AREAS if need_pts[s]]

    # 2. 候选网格（约 330 m 间距），逐个检查
    lons = [NODES[k]["lon"] for k in NODES]
    lats = [NODES[k]["lat"] for k in NODES]
    cands = []
    for lo in np.arange(min(lons) - 0.02, max(lons) + 0.02, 0.003):
        for la in np.arange(min(lats) - 0.02, max(lats) + 0.02, 0.003):
            ground = float(dem_at(lo, la))
            p = (float(lo), float(la), ground + RELAY["max_agl"])
            if not link_ok(p, G01_POS, "back"):
                continue
            cov = []
            for s in areas_need:
                if all(relay_covers(p, q) for q in need_pts[s]):   # all() 遇到第一个 False 就停
                    cov.append(s)
            if cov:
                cands.append((p, cov))
    print(f"有效候选点 {len(cands)} 个，用时 {time.time() - t0:.0f}s")
    cands.sort(key=lambda x: -len(x[1]))
    for p, cov in cands[:15]:
        print(f"  {p[0]:.4f},{p[1]:.4f},{p[2]:.0f}  覆盖 {len(cov)}: {cov}")
    # 每个服务区能被多少个候选点保障
    for s in areas_need:
        print(s, "可保障候选点数", sum(1 for _, cov in cands if s in cov))
    with open(os.path.join(RESULT_DIR, "relay_candidates.pkl"), "wb") as f:
        pickle.dump({"cands": cands, "areas_need": areas_need}, f)
