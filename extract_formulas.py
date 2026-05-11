# -*- coding: utf-8 -*-
"""
从《伤寒论》背诵条文中提取所有方剂（处方）。

方剂识别模式：
  - XX汤/散/丸主之（主治句型）
  - 宜/与/可与/当以/当服 + XX汤/散/丸（建议句型）
  - 属XX汤（归类句型）

输出：
  1. 控制台打印按章节分组的方剂及其条文
  2. formulas.json — 结构化数据
  3. formulas_summary.md — 方剂汇总表
"""

import re
import json
import sys
from collections import defaultdict
from pathlib import Path

# 强制 stdout 使用 utf-8，避免 ✓ 等字符的编码问题
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read_file(filepath: str) -> str:
    """读取文件，自动检测编码（UTF-8 或 GBK 系列）。"""
    for encoding in ["utf-8", "gbk", "gb2312", "gb18030"]:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise RuntimeError(f"无法解码文件: {filepath}")


def parse_articles(raw_text: str) -> list[dict]:
    """
    将原始文本解析为条文列表。

    文件格式（无表头/无行号前缀，纯文本）：
        太阳病脉证并治（上）          ← 章节标题
        1、太阳之为病，脉浮...        ← 条文
        2、太阳病，发热...            ← 条文
        ...
        太阳病脉证并治（中）          ← 下一章节

    每条: {section, article_number, content}
    """
    articles = []
    current_section = ""

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # 条文: 以 "数字、" 或 "数字．" 开头
        article_match = re.match(r"(\d+)[、．](.+)", line)
        if article_match:
            num = int(article_match.group(1))
            content = article_match.group(2).strip()
            articles.append({
                "section": current_section,
                "article_number": num,
                "content": content,
            })
        else:
            # 否则是章节标题
            current_section = line

    return articles


# ─── 方剂提取正则 ─────────────────────────────────────────────
# 方剂名: 中文 + 结尾（汤/散/丸/煎）
# 使用非贪婪匹配来到达最近的结尾词

FORMULA_CHAR = r"[一-鿿\w]"
FORMULA_END = r"(?:汤|散|丸|煎)"

# 模式1: "XX汤/散/丸主之" — 主治句型
RE_ZHUZHI = re.compile(rf"({FORMULA_CHAR}+?{FORMULA_END})主之")

# 模式2: "宜XX汤/散/丸", "与XX汤/散/丸", "可与XX汤",
#         "当以XX汤", "当服XX汤", "先宜服XX汤"
RE_YIYU = re.compile(
    r"(?:先宜服|先宜|宜|与|可与|当以|当服|以|服|用)"
    rf"({FORMULA_CHAR}+?{FORMULA_END})"
)

# 模式3: "属XX汤/散/丸"
RE_SHU = re.compile(rf"属({FORMULA_CHAR}+?{FORMULA_END})")


def extract_formulas(articles: list[dict]) -> list[dict]:
    """
    从条文中提取所有方剂出现记录。
    返回: [{formula, section, article_number, content, match_type}, ...]
    """
    records = []

    for art in articles:
        text = art["content"]
        seen_in_this_article = set()

        def add_match(formula_name: str, match_type: str):
            name = formula_name.strip("，。；、").strip()
            if name not in seen_in_this_article:
                seen_in_this_article.add(name)
                records.append({
                    "formula": name,
                    "section": art["section"],
                    "article_number": art["article_number"],
                    "content": text,
                    "match_type": match_type,
                })

        # 模式1: 主治句型（最可靠）
        for m in RE_ZHUZHI.finditer(text):
            add_match(m.group(1), "主治")

        # 模式2: 建议句型
        for m in RE_YIYU.finditer(text):
            add_match(m.group(1), "宜/与/可")

        # 模式3: 归类句型
        for m in RE_SHU.finditer(text):
            add_match(m.group(1), "属")

    return records


def normalize_name(name: str) -> str:
    """去除方剂名末尾可能的干扰字符。"""
    return name.rstrip("，。；、，")


def build_summary(records: list[dict]) -> dict[str, list[dict]]:
    """将记录按方剂名聚合。"""
    groups = defaultdict(list)
    for r in records:
        groups[normalize_name(r["formula"])].append(r)
    return dict(groups)


def print_summary_by_section(
    articles: list[dict],
    records: list[dict],
) -> None:
    """按章节打印方剂汇总。"""
    art_to_formulas = defaultdict(set)
    for r in records:
        art_to_formulas[(r["section"], r["article_number"])].add(
            normalize_name(r["formula"])
        )

    current_section = ""
    for art in articles:
        if art["section"] != current_section:
            current_section = art["section"]
            print(f"\n{'='*60}")
            print(f"  {current_section}")
            print(f"{'='*60}")

        formulas = art_to_formulas.get((art["section"], art["article_number"]))
        if formulas:
            print(f"\n  第{art['article_number']}条: {art['content']}")
            print(f"    -> 方剂: {', '.join(sorted(formulas))}")


def print_formula_index(summary: dict[str, list[dict]]) -> None:
    """打印方剂索引（按名称排序）。"""
    print(f"\n{'='*60}")
    print(f"  方剂索引（共 {len(summary)} 首）")
    print(f"{'='*60}")

    for i, (name, recs) in enumerate(sorted(summary.items()), 1):
        locations = [f"第{r['article_number']}条" for r in recs]
        sections = sorted(set(r["section"] for r in recs))
        print(f"\n  {i}. {name}")
        print(f"     出现 {len(recs)} 次 | 章节: {' / '.join(sections)}")
        print(f"     条文位置: {', '.join(locations)}")
        snippet = recs[0]["content"]
        if len(snippet) > 60:
            snippet = snippet[:60] + "..."
        print(f"     主治: {snippet}")


def export_json(summary: dict[str, list[dict]], filepath: str) -> None:
    """导出为 JSON 文件。"""
    output = {}
    for name, recs in sorted(summary.items()):
        output[name] = {
            "count": len(recs),
            "articles": [
                {
                    "section": r["section"],
                    "article_number": r["article_number"],
                    "content": r["content"],
                    "match_type": r["match_type"],
                }
                for r in recs
            ],
        }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] JSON 已导出到: {filepath}")


def export_markdown(
    summary: dict[str, list[dict]],
    filepath: str,
) -> None:
    """导出为 Markdown 汇总文件。"""
    lines = [
        "# 《伤寒论》方剂汇总",
        "",
        f"共提取 **{len(summary)}** 首方剂。",
        "",
        "| 序号 | 方剂名 | 出现次数 | 所在章节 | 相关条文 |",
        "|------|--------|----------|----------|----------|",
    ]

    for i, (name, recs) in enumerate(sorted(summary.items()), 1):
        sections = " / ".join(sorted(set(r["section"] for r in recs)))
        article_nums = [f"第{r['article_number']}条" for r in recs]
        lines.append(
            f"| {i} | {name} | {len(recs)} | {sections} | "
            f"{', '.join(article_nums)} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 各章节方剂分布")
    lines.append("")

    by_section = defaultdict(set)
    for name, recs in summary.items():
        for r in recs:
            by_section[r["section"]].add(name)

    for section, formulas in sorted(by_section.items()):
        lines.append(f"### {section}")
        lines.append("")
        for f in sorted(formulas):
            lines.append(f"- {f}")
        lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Markdown 汇总已导出到: {filepath}")


def main():
    base_dir = Path(__file__).parent
    input_file = base_dir / "伤寒论背诵条文-宋.txt"

    print("=" * 60)
    print("  《伤寒论》方剂提取工具")
    print("=" * 60)

    # 1. 读取 & 解析
    print(f"\n读取文件: {input_file}")
    raw = read_file(str(input_file))
    articles = parse_articles(raw)
    print(f"解析到 {len(articles)} 条条文")

    # 2. 提取方剂
    records = extract_formulas(articles)
    print(f"提取到 {len(records)} 条方剂记录")

    # 3. 按方剂名聚合
    summary = build_summary(records)
    print(f"去重后共 {len(summary)} 首方剂")

    # 4. 按章节输出
    print_summary_by_section(articles, records)

    # 5. 方剂索引
    print_formula_index(summary)

    # 6. 导出文件
    export_json(summary, str(base_dir / "formulas.json"))
    export_markdown(summary, str(base_dir / "formulas_summary.md"))


if __name__ == "__main__":
    main()
