import spacy
from spacy import displacy
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# 加载中文模型
try:
    nlp = spacy.load("zh_core_web_md")
    print("成功加载 spaCy 中文模型")
    use_chinese_model = True
except:
    print("未找到中文模型，使用英文空白模型")
    print("建议：执行 'python -m spacy download zh_core_web_md' 安装中文模型")
    nlp = spacy.blank("en")
    nlp.add_pipe('sentencizer')
    use_chinese_model = False

# 评估指标计算函数
def calculate_metrics(predicted, ground_truth):
    """计算准确率、召回率和F1值"""
    # 计算TP、FP、FN
    predicted_set = set(predicted)
    ground_truth_set = set(ground_truth)
    
    TP = len(predicted_set.intersection(ground_truth_set))
    FP = len(predicted_set - ground_truth_set)
    FN = len(ground_truth_set - predicted_set)
    
    # 计算准确率
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    # 计算召回率
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    # 计算F1值
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

# 分词评估
def evaluate_tokenization(doc, ground_truth_tokens):
    """评估分词性能"""
    predicted_tokens = [token.text for token in doc]
    precision, recall, f1 = calculate_metrics(predicted_tokens, ground_truth_tokens)
    
    print(f"分词评估结果:")
    print(f"  准确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1值: {f1:.4f}")
    
    # 显示差异
    predicted_set = set(predicted_tokens)
    ground_truth_set = set(ground_truth_tokens)
    
    print(f"  额外分词: {list(predicted_set - ground_truth_set)[:10]}")
    print(f"  遗漏分词: {list(ground_truth_set - predicted_set)[:10]}")
    
    return precision, recall, f1

# 词性标注评估
def evaluate_pos_tagging(doc, ground_truth_tags):
    """评估词性标注性能"""
    predicted_tags = [(token.text, token.pos_) for token in doc]
    precision, recall, f1 = calculate_metrics(predicted_tags, ground_truth_tags)
    
    print(f"词性标注评估结果:")
    print(f"  准确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1值: {f1:.4f}")
    
    return precision, recall, f1

# 命名实体识别评估
def evaluate_ner(doc, ground_truth_entities):
    """评估命名实体识别性能"""
    predicted_entities = [(ent.text, ent.label_) for ent in doc.ents]
    precision, recall, f1 = calculate_metrics(predicted_entities, ground_truth_entities)
    
    print(f"命名实体识别评估结果:")
    print(f"  准确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1值: {f1:.4f}")
    
    # 显示识别结果
    print(f"  预测实体: {predicted_entities}")
    print(f"  标准实体: {ground_truth_entities}")
    
    return precision, recall, f1

# 测试文本
text = """晋太元中，武陵人捕鱼为业。缘溪行，忘路之远近。忽逢桃花林，夹岸数百步，中无杂树，芳草鲜美，落英缤纷。渔人甚异之，复前行，欲穷其林。林尽水源，便得一山，山有小口，仿佛若有光。便舍船，从口入。初极狭，才通人。复行数十步，豁然开朗。土地平旷，屋舍俨然，有良田美池桑竹之属。阡陌交通，鸡犬相闻。其中往来种作，男女衣着，悉如外人。黄发垂髫，并怡然自乐。"""

print("=" * 70)
print("文本分析：《桃花源记》节选")
print("=" * 70)
print(f"文本长度: {len(text)} 字符")
print()

# 处理文本
doc = nlp(text)

# 1. 分词分析
print("1. 分词结果")
print("-" * 40)

# 显示前50个词的分词结果
words = [token.text for token in doc]
print("分词示例（前50个词）：")
for i in range(0, min(50, len(words)), 10):
    print("  " + "  ".join(f"{j+1:2d}.{words[j]}" for j in range(i, min(i+10, len(words)))))

# 统计信息
try:
    print(f"\n总词数: {len(doc)}")
    print(f"句子数: {len(list(doc.sents))}")
except ValueError:
    print("\n注意：模型未包含句子分割组件，无法计算句子数")

# 词长分布
word_lengths = [len(token.text) for token in doc]
length_counter = Counter(word_lengths)
print("\n词长分布:")
for length in sorted(length_counter.keys()):
    count = length_counter[length]
    percentage = (count / len(doc)) * 100
    bar = "█" * int(percentage / 2)
    print(f"  {length}字词: {count:3d}个 ({percentage:5.1f}%) {bar}")

# 2. 词性标注分析
print("\n" + "=" * 70)
print("2. 词性标注分析")
print("-" * 40)

# 词性统计
pos_counts = Counter([token.pos_ for token in doc])
pos_explanations = {
    'NOUN': '名词', 'VERB': '动词', 'ADJ': '形容词', 'ADV': '副词',
    'ADP': '介词', 'CCONJ': '连词', 'PART': '助词', 'PRON': '代词',
    'PROPN': '专有名词', 'NUM': '数词', 'PUNCT': '标点', 'X': '其他'
}

print("词性统计:")
print(f"{'词性标签':<10} {'中文':<8} {'数量':<6} {'比例':<8}")
print("-" * 40)
total_tokens = len(doc)
for pos, count in pos_counts.most_common():
    chinese_name = pos_explanations.get(pos, '未知')
    percentage = (count / total_tokens) * 100
    print(f"{pos:<10} {chinese_name:<8} {count:<6} {percentage:>6.1f}%")

# 显示详细标注示例
print("\n详细标注示例（前30个词）：")
print(f"{'序号':<4} {'词语':<6} {'词性':<8} {'中文解释':<10} {'依存关系':<10} {'依存父词':<8}")
print("-" * 60)
for i, token in enumerate(doc):
    if i >= 30:
        break
    pos_chinese = pos_explanations.get(token.pos_, '未知')
    print(f"{i+1:<4} {token.text:<6} {token.pos_:<8} {pos_chinese:<10} {token.dep_:<10} {token.head.text:<8}")

# 3. 命名实体识别
print("\n" + "=" * 70)
print("3. 命名实体识别 (NER)")
print("-" * 40)

# 实体统计
if doc.ents:
    entity_types = Counter([ent.label_ for ent in doc.ents])
    entity_explanations = {
        'PERSON': '人物', 'NORP': '民族/宗教团体', 'FAC': '设施',
        'ORG': '组织', 'GPE': '地缘政治实体', 'LOC': '地点',
        'PRODUCT': '产品', 'EVENT': '事件', 'WORK_OF_ART': '艺术品',
        'LAW': '法律', 'LANGUAGE': '语言', 'DATE': '日期',
        'TIME': '时间', 'PERCENT': '百分比', 'MONEY': '金额',
        'QUANTITY': '数量', 'ORDINAL': '序数', 'CARDINAL': '基数'
    }
    
    print("识别到的实体:")
    print(f"{'实体':<10} {'类型':<15} {'中文':<10} {'位置':<10}")
    print("-" * 50)
    for ent in doc.ents:
        start, end = ent.start_char, ent.end_char
        chinese_label = entity_explanations.get(ent.label_, '未知')
        print(f"{ent.text:<10} {ent.label_:<15} {chinese_label:<10} ({start:3d}-{end:3d})")
    
    print(f"\n实体类型统计:")
    for label, count in entity_types.most_common():
        chinese_name = entity_explanations.get(label, '未知')
        percentage = (count / len(doc.ents)) * 100
        print(f"  {label:<15} {chinese_name:<10}: {count}个 ({percentage:5.1f}%)")
else:
    print("未识别到命名实体")

# 4. 依存句法分析
print("\n" + "=" * 70)
print("4. 依存句法分析示例")
print("-" * 40)

# 选取一个句子进行详细分析
sample_sentences = list(doc.sents)
if sample_sentences:
    sample_sentence = sample_sentences[0]  # 第一个句子
    
    print("示例句子:", sample_sentence.text)
    print("\n依存关系分析:")
    
    # 定义依存关系的中文解释
    dep_explanations = {
        'ROOT': '根节点', 'nsubj': '名词主语', 'dobj': '直接宾语',
        'pobj': '介词宾语', 'amod': '形容词修饰', 'nummod': '数词修饰',
        'compound': '复合词', 'case': '格标记', 'mark': '标记',
        'advmod': '副词修饰', 'aux': '助动词', 'cop': '系动词',
        'det': '限定词', 'clf': '量词', 'cc': '并列连词',
        'conj': '并列', 'punct': '标点'
    }
    
    print(f"{'词语':<6} {'词性':<6} {'依存关系':<12} {'父词':<6} {'关系解释':<15}")
    print("-" * 60)
    for token in sample_sentence:
        dep_chinese = dep_explanations.get(token.dep_, token.dep_)
        print(f"{token.text:<6} {token.pos_:<6} {token.dep_:<12} {token.head.text:<6} {dep_chinese:<15}")

# 5. 词频统计
print("\n" + "=" * 70)
print("5. 词频统计")
print("-" * 40)

# 过滤掉标点符号
content_words = [token for token in doc if not token.is_punct and not token.is_space]
word_freq = Counter([token.text for token in content_words])

print("高频词语（前20个）：")
print(f"{'词语':<6} {'频次':<6} {'词性':<8} {'比例':<8}")
print("-" * 40)
for i, (word, freq) in enumerate(word_freq.most_common(20)):
    # 获取该词的主要词性
    pos_tags = [token.pos_ for token in content_words if token.text == word]
    main_pos = max(set(pos_tags), key=pos_tags.count) if pos_tags else 'UNK'
    percentage = (freq / len(content_words)) * 100
    print(f"{word:<6} {freq:<6} {main_pos:<8} {percentage:>6.2f}%")

# 6. 文言文特性分析
print("\n" + "=" * 70)
print("6. 文言文特性分析")
print("-" * 40)

# 统计文言虚词
classical_particles = ['之', '乎', '者', '也', '矣', '焉', '哉', '兮', '耶', '欤', '尔', '耳']
found_particles = []

for particle in classical_particles:
    count = sum(1 for token in doc if token.text == particle)
    if count > 0:
        found_particles.append((particle, count))

if found_particles:
    print("文言虚词使用情况:")
    for particle, count in sorted(found_particles, key=lambda x: x[1], reverse=True):
        # 获取该虚词的词性
        poses = set(token.pos_ for token in doc if token.text == particle)
        pos_str = "/".join(poses)
        print(f"  {particle}: {count}次，主要词性: {pos_str}")
else:
    print("未发现典型的文言虚词")

# 7. 可视化选项
print("\n" + "=" * 70)
print("7. 可视化选项")
print("-" * 40)
print("可选的 spaCy 可视化命令:")
print("1. 依存关系图: displacy.serve(doc, style='dep')")
print("2. 实体识别图: displacy.serve(doc, style='ent')")
print("3. 生成HTML: html = displacy.render(doc, style='ent', page=True)")

# 8. 保存结果
print("\n" + "=" * 70)
print("8. 保存分析结果")
print("-" * 40)

# 创建详细的数据表
data = []
for i, token in enumerate(doc):
    data.append({
        '序号': i+1,
        '词语': token.text,
        '词性': token.pos_,
        '词性解释': pos_explanations.get(token.pos_, '未知'),
        '依存关系': token.dep_,
        '依存父词': token.head.text,
        '是否标点': token.is_punct,
        '是否空格': token.is_space,
        '是否数字': token.like_num,
        '是否实体': any(ent.start <= i < ent.end for ent in doc.ents)
    })

df = pd.DataFrame(data)

# 保存到CSV
df.to_csv('taohuayuan_analysis.csv', index=False, encoding='utf-8-sig')
print(f"详细分析结果已保存到: taohuayuan_analysis.csv")

# 保存实体信息
if doc.ents:
    entity_data = []
    for ent in doc.ents:
        entity_data.append({
            '实体': ent.text,
            '类型': ent.label_,
            '类型解释': entity_explanations.get(ent.label_, '未知'),
            '起始位置': ent.start_char,
            '结束位置': ent.end_char,
            '上下文': text[max(0, ent.start_char-10):min(len(text), ent.end_char+10)]
        })
    
    entity_df = pd.DataFrame(entity_data)
    entity_df.to_csv('taohuayuan_entities.csv', index=False, encoding='utf-8-sig')
    print(f"实体识别结果已保存到: taohuayuan_entities.csv")

# 9. 评估分析
print("\n" + "=" * 70)
print("9. 模型评估")
print("=" * 70)

if use_chinese_model:
    # 为测试文本创建标准标注（人工标注的标准答案）
    # 标准分词
    ground_truth_tokens = [
        "晋", "太元", "中", "，", "武陵", "人", "捕鱼", "为", "业", "。",
        "缘", "溪", "行", "，", "忘", "路", "之", "远近", "。", "忽",
        "逢", "桃花林", "，", "夹岸", "数百步", "，", "中", "无", "杂树", "，",
        "芳草", "鲜美", "，", "落英", "缤纷", "。", "渔人", "甚", "异", "之", "，",
        "复", "前行", "，", "欲", "穷", "其", "林", "。", "林", "尽", "水源", "，",
        "便", "得", "一", "山", "，", "山", "有", "小口", "，", "仿佛", "若", "有", "光", "。",
        "便", "舍", "船", "，", "从", "口", "入", "。", "初", "极", "狭", "，", "才", "通", "人", "。",
        "复", "行", "数十步", "，", "豁然开朗", "。", "土地", "平旷", "，", "屋舍", "俨然", "，",
        "有", "良田", "美池", "桑竹", "之", "属", "。", "阡陌", "交通", "，", "鸡犬", "相闻", "。",
        "其中", "往来", "种作", "，", "男女", "衣着", "，", "悉", "如", "外人", "。",
        "黄发垂髫", "，", "并", "怡然自乐", "。"
    ]

    # 标准词性标注（格式：(词语, 词性)）
    ground_truth_tags = [
        ("晋", "NOUN"), ("太元", "NOUN"), ("中", "NOUN"), ("，", "PUNCT"), ("武陵", "NOUN"), ("人", "NOUN"), ("捕鱼", "VERB"), ("为", "VERB"), ("业", "NOUN"), ("。", "PUNCT"),
        ("缘", "VERB"), ("溪", "NOUN"), ("行", "VERB"), ("，", "PUNCT"), ("忘", "VERB"), ("路", "NOUN"), ("之", "PART"), ("远近", "NOUN"), ("。", "PUNCT"), ("忽", "ADV"),
        ("逢", "VERB"), ("桃花林", "NOUN"), ("，", "PUNCT"), ("夹岸", "VERB"), ("数百步", "NOUN"), ("，", "PUNCT"), ("中", "NOUN"), ("无", "VERB"), ("杂树", "NOUN"), ("，", "PUNCT"),
        ("芳草", "NOUN"), ("鲜美", "ADJ"), ("，", "PUNCT"), ("落英", "NOUN"), ("缤纷", "ADJ"), ("。", "PUNCT"), ("渔人", "NOUN"), ("甚", "ADV"), ("异", "VERB"), ("之", "PART"), ("，", "PUNCT"),
        ("复", "ADV"), ("前行", "VERB"), ("，", "PUNCT"), ("欲", "VERB"), ("穷", "VERB"), ("其", "PRON"), ("林", "NOUN"), ("。", "PUNCT"), ("林", "NOUN"), ("尽", "VERB"), ("水源", "NOUN"), ("，", "PUNCT"),
        ("便", "ADV"), ("得", "VERB"), ("一", "NUM"), ("山", "NOUN"), ("，", "PUNCT"), ("山", "NOUN"), ("有", "VERB"), ("小口", "NOUN"), ("，", "PUNCT"), ("仿佛", "ADV"), ("若", "VERB"), ("有", "VERB"), ("光", "NOUN"), ("。", "PUNCT"),
        ("便", "ADV"), ("舍", "VERB"), ("船", "NOUN"), ("，", "PUNCT"), ("从", "ADP"), ("口", "NOUN"), ("入", "VERB"), ("。", "PUNCT"), ("初", "ADV"), ("极", "ADV"), ("狭", "ADJ"), ("，", "PUNCT"), ("才", "ADV"), ("通", "VERB"), ("人", "NOUN"), ("。", "PUNCT"),
        ("复", "ADV"), ("行", "VERB"), ("数十步", "NOUN"), ("，", "PUNCT"), ("豁然开朗", "ADJ"), ("。", "PUNCT"), ("土地", "NOUN"), ("平旷", "ADJ"), ("，", "PUNCT"), ("屋舍", "NOUN"), ("俨然", "ADJ"), ("，", "PUNCT"),
        ("有", "VERB"), ("良田", "NOUN"), ("美池", "NOUN"), ("桑竹", "NOUN"), ("之", "PART"), ("属", "NOUN"), ("。", "PUNCT"), ("阡陌", "NOUN"), ("交通", "VERB"), ("，", "PUNCT"), ("鸡犬", "NOUN"), ("相闻", "VERB"), ("。", "PUNCT"),
        ("其中", "ADV"), ("往来", "VERB"), ("种作", "VERB"), ("，", "PUNCT"), ("男女", "NOUN"), ("衣着", "NOUN"), ("，", "PUNCT"), ("悉", "ADV"), ("如", "ADP"), ("外人", "NOUN"), ("。", "PUNCT"),
        ("黄发垂髫", "NOUN"), ("，", "PUNCT"), ("并", "ADV"), ("怡然自乐", "ADJ"), ("。", "PUNCT")
    ]

    # 标准命名实体
    ground_truth_entities = [
        ("晋太元", "DATE"), ("武陵", "GPE"), ("桃花林", "LOC")
    ]

    print("开始评估模型性能...")
    print()

    # 评估分词
    print("1. 分词评估")
    print("-" * 40)
    token_precision, token_recall, token_f1 = evaluate_tokenization(doc, ground_truth_tokens)
    print()

    # 评估词性标注
    print("2. 词性标注评估")
    print("-" * 40)
    pos_precision, pos_recall, pos_f1 = evaluate_pos_tagging(doc, ground_truth_tags)
    print()

    # 评估命名实体识别
    print("3. 命名实体识别评估")
    print("-" * 40)
    ner_precision, ner_recall, ner_f1 = evaluate_ner(doc, ground_truth_entities)
    print()

    # 汇总评估结果
    print("4. 评估结果汇总")
    print("-" * 40)
    print(f"{'评估项':<15} {'准确率':<10} {'召回率':<10} {'F1值':<10}")
    print("-" * 50)
    print(f"{'分词':<15} {token_precision:.4f}    {token_recall:.4f}    {token_f1:.4f}")
    print(f"{'词性标注':<15} {pos_precision:.4f}    {pos_recall:.4f}    {pos_f1:.4f}")
    print(f"{'实体识别':<15} {ner_precision:.4f}    {ner_recall:.4f}    {ner_f1:.4f}")

    # 保存评估结果
    eval_results = {
        '评估项': ['分词', '词性标注', '实体识别'],
        '准确率': [token_precision, pos_precision, ner_precision],
        '召回率': [token_recall, pos_recall, ner_recall],
        'F1值': [token_f1, pos_f1, ner_f1]
    }

    eval_df = pd.DataFrame(eval_results)
    eval_df.to_csv('evaluation_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n评估结果已保存到: evaluation_results.csv")
else:
    print("使用空白模型，无法进行完整评估")
    print("建议安装中文模型以获得更准确的评估结果")
    
    # 保存基本评估结果
    eval_results = {
        '评估项': ['分词', '词性标注', '实体识别'],
        '准确率': [0.0, 0.0, 0.0],
        '召回率': [0.0, 0.0, 0.0],
        'F1值': [0.0, 0.0, 0.0]
    }
    
    eval_df = pd.DataFrame(eval_results)
    eval_df.to_csv('evaluation_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n基本评估结果已保存到: evaluation_results.csv")

print("\n" + "=" * 70)
print("分析完成！")
print("=" * 70)