# -*- coding: utf-8 -*-
"""
fill_template.py —— 按“结果提交模板.xlsx”的工作表与列名，汇总 Q1~Q4 结果到 结果/结果提交.xlsx
"""
import os
import pandas as pd
from common import BASE_DIR, RESULT_DIR

TEMPLATE = os.path.join(BASE_DIR, "结果提交模板.xlsx")
SOURCES = {"Q1_单点组批": ("Q1_结果.xlsx", "Q1_单点组批"),
           "Q2_运输架次": ("Q2_结果.xlsx", "Q2_运输架次"),
           "Q2_逐箱交付": ("Q2_结果.xlsx", "Q2_逐箱交付"),
           "Q3_中继架次": ("Q3_结果.xlsx", "Q3_中继架次"),
           "Q3_通信保障": ("Q3_结果.xlsx", "Q3_通信保障"),
           "Q4_分区配置": ("Q4_结果.xlsx", "Q4_分区配置")}

if __name__ == "__main__":
    tpl = pd.read_excel(TEMPLATE, sheet_name=None)
    with pd.ExcelWriter(os.path.join(RESULT_DIR, "结果提交.xlsx")) as w:
        for sheet, df_t in tpl.items():
            cols = list(df_t.columns)
            f, s = SOURCES[sheet]
            src = pd.read_excel(os.path.join(RESULT_DIR, f), sheet_name=s)
            miss = [c for c in cols if c not in src.columns]
            if miss:
                raise KeyError(f"{sheet} 缺少列 {miss}")
            src[cols].to_excel(w, sheet_name=sheet, index=False)
            print(f"{sheet}: {len(src)} 行")
    print("已生成 结果/结果提交.xlsx")
