# -*- coding: utf-8 -*-
"""
提取公式.py
作用：Word 里的数学公式是“公式对象（OMML）”，普通读取读不到。
      这里直接解析 docx 内部的 XML，把每段的普通文字和公式文字一起输出。
"""
import zipfile                       # docx 本质是一个 zip 压缩包
import re                            # 正则表达式，用来处理 XML 文本

路径 = "山区洪涝灾害下无人机运输与通信协同优化.docx"

# 打开 docx 压缩包，读取正文 XML
with zipfile.ZipFile(路径) as 压缩包:
    xml = 压缩包.read("word/document.xml").decode("utf-8")

# 按段落 <w:p ...> ... </w:p> 切分
段落列表 = re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S)

with open("公式提取结果.txt", "w", encoding="utf-8") as 输出:
    for 段 in 段落列表:
        # 把公式区域 <m:oMath> 用【】标记出来，方便区分
        段 = re.sub(r"<m:oMath[ >]", "【公式:<m:x ", 段)
        段 = 段.replace("</m:oMath>", "】")
        # 常见公式结构的标记：分数、上下标，加上符号便于阅读
        段 = 段.replace("<m:den>", "/(").replace("</m:den>", ")")
        段 = 段.replace("<m:sub>", "_{").replace("</m:sub>", "}")
        段 = 段.replace("<m:sup>", "^{").replace("</m:sup>", "}")
        # 提取所有文本节点 <w:t> 和 <m:t>，以及我们插入的标记
        片段 = re.findall(r"【公式:|】|/\(|\)|_\{|\^\{|\}|<[wm]:t[^>]*>([^<]*)</[wm]:t>", 段)
        # findall 带分组时只返回分组内容，重新用 finditer 拼接更稳妥
        文字 = ""
        for m in re.finditer(r"【公式:|】|/\(|\)|_\{|\^\{|\}|<[wm]:t[^>]*>([^<]*)</[wm]:t>", 段):
            文字 += m.group(1) if m.group(1) is not None else m.group(0)
        if "【公式" in 文字:
            输出.write(文字 + "\n\n")
print("完成")
