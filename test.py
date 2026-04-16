# -*- coding: utf-8 -*-
import spacy
from spacy import displacy
from collections import Counter

# 加载中文模型（需要先安装：python -m spacy download zh_core_web_md）
try:
    nlp = spacy.load("zh_core_web_md")
    print("成功加载中文模型 zh_core_web_md")
    use_full_model = True
except OSError:
    # 如果无法加载中文模型，使用英文空白模型
    print("无法加载中文模型 zh_core_web_md，使用英文空白模型")
    print("建议：在Python 3.9.25环境中运行，或执行 'python -m spacy download zh_core_web_md' 安装模型")
    nlp = spacy.blank("en")
    # 添加sentencizer组件用于句子分割
    nlp.add_pipe('sentencizer')
    use_full_model = False

# 读取文本或使用默认文本
try:
    with open("ancient_chinese.txt", "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    # 如果文件不存在，使用默认文本
    text = "晋太元中，武陵人捕鱼为业。缘溪行，忘路之远近。忽逢桃花林，夹岸数百步，中无杂树，芳草鲜美，落英缤纷。渔人甚异之，复前行，欲穷其林。林尽水源，便得一山，山有小口，仿佛若有光。便舍船，从口入。初极狭，才通人。复行数十步，豁然开朗。土地平旷，屋舍俨然，有良田美池桑竹之属。阡陌交通，鸡犬相闻。其中往来种作，男女衣着，悉如外人。黄发垂髫，并怡然自乐。"

# 处理文本
doc = nlp(text)

# 1. 基本统计
print("=" * 50)
print("文本基本信息:")
print(f"总字符数: {len(text)}")
print(f"总词数: {len(list(doc))}")
try:
    print(f"总句子数: {len(list(doc.sents))}")
except ValueError:
    print("总句子数: 无法计算（需要sentencizer组件）")

# 2. 分词展示（前50个词）
print("\n" + "=" * 50)
print("分词示例（前50个词）:")
for i, token in enumerate(doc):
    if i >= 50:
        break
    print(f"{token.text:<4}", end=" " if (i+1) % 10 != 0 else "\n")
print("\n")

# 3. 词性统计
print("=" * 50)
print("词性标注统计:")
pos_counts = Counter()
for token in doc:
    pos_counts[token.pos_] += 1

for pos, count in pos_counts.most_common(10):
    print(f"{pos:>10}: {count:>4}次")

# 4. 词性详情示例
print("\n" + "=" * 50)
print("详细词性标注示例（选取一段）:")

# 选取《桃花源记》段落进行详细分析
taohua_text = "晋太元中，武陵人捕鱼为业。缘溪行，忘路之远近。忽逢桃花林，夹岸数百步，中无杂树，芳草鲜美，落英缤纷。"
taohua_doc = nlp(taohua_text)

print("\n原文:")
print(taohua_text)
print("\n词性分析:")
for token in taohua_doc:
    try:
        print(f"{token.text:<4} -> {token.pos_:<6} ({spacy.explain(token.pos_) if use_full_model else 'N/A'})")
    except Exception:
        print(f"{token.text:<4} -> {token.pos_:<6} (N/A)")

# 5. 命名实体识别
print("\n" + "=" * 50)
print("命名实体识别:")
try:
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))

    entity_counts = Counter(entities)
    print("识别到的实体:")
    if entity_counts:
        for (text, label), count in entity_counts.most_common(15):
            print(f"{text:>8} -> {label:>10} ({count}次)")
    else:
        print("未识别到实体（需要完整模型）")
except Exception as e:
    print(f"无法进行命名实体识别: {e}")

# 6. 依存句法分析示例
print("\n" + "=" * 50)
print("依存句法分析示例:")
try:
    sample_sent = list(doc.sents)[0]  # 取第一句
    print(f"例句: {sample_sent}")
    for token in sample_sent:
        try:
            print(f"{token.text:<4} <-{token.dep_:>6}- {token.head.text:<4} ({token.head.pos_})")
        except Exception:
            print(f"{token.text:<4} <-{token.dep_:>6}- {token.head.text:<4} (N/A)")
except (IndexError, ValueError) as e:
    print(f"无法获取例句: {e}")

# 7. 可视化（可选，需要运行在支持的环境中）
# displacy.serve(doc, style="dep", options={"compact": True})

