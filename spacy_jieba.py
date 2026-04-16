# coding=utf-8
import spacy
from spacy.tokens import Doc
import jieba
import re
from collections import Counter

# 1. 加载 spaCy 模型（禁用不需要的组件）
try:
    nlp = spacy.load("zh_core_web_md", disable=["parser", "ner", "attribute_ruler", "lemmatizer"])
    print("成功加载中文模型 zh_core_web_md")
    use_full_model = True
except OSError:
    print("无法加载中文模型 zh_core_web_md，使用英文空白模型")
    print("建议：执行 'python -m spacy download zh_core_web_md' 安装模型")
    nlp = spacy.blank("en")
    nlp.add_pipe('sentencizer')
    use_full_model = False

# 2. 定义自定义分词器
class JiebaTokenizer:
    def __init__(self, vocab, enable_pos=False):
        self.vocab = vocab
        self.enable_pos = enable_pos
        # 为 jieba 添加一些文言文专有词汇
        self._add_ancient_words()
    
    def _add_ancient_words(self):
        """添加文言文特有词汇到 jieba 词典"""
        ancient_words = [
            "庄周", "胡蝶", "栩栩然", "蘧蘧然", "自喻", 
            "俄然", "物化", "辩斗", "车盖", "盘盂",
            "沧沧凉凉", "探汤", "太元", "武陵", "缘溪",
            "桃花林", "落英缤纷", "豁然开朗", "屋舍俨然",
            "阡陌", "黄发垂髫", "滕文公", "太史公", "仲尼",
            "祗回", "六艺", "孙子", "校之以计"
        ]
        for word in ancient_words:
            jieba.add_word(word, freq=1000, tag='nz')
    
    def __call__(self, text):
        # 使用 jieba 精确模式分词
        words = list(jieba.cut(text, cut_all=False))
        
        # 清理空格和空字符
        words = [w.strip() for w in words if w.strip()]
        
        # 创建 spaCy 的 Doc 对象
        return Doc(self.vocab, words=words)

# 3. 替换 spaCy 的分词器
nlp.tokenizer = JiebaTokenizer(nlp.vocab)

# 4. 读取古文文本
ancient_text = """昔者庄周梦为胡蝶，栩栩然胡蝶也，自喻适志与！不知周也。俄然觉，则蘧蘧然周也。不知周之梦为胡蝶与，胡蝶之梦为周与？周与胡蝶，则必有分矣。此之谓物化。

孔子东游，见两小儿辩斗，问其故。一儿曰："我以日始出时去人近，而日中时远也。"一儿以日初出远，而日中时近也。一儿曰："日初出大如车盖，及日中则如盘盂，此不为远者小而近者大乎？"一儿曰："日初出沧沧凉凉，及其日中如探汤，此不为近者热而远者凉乎？"孔子不能决也。两小儿笑曰："孰为汝多知乎？"

晋太元中，武陵人捕鱼为业。缘溪行，忘路之远近。忽逢桃花林，夹岸数百步，中无杂树，芳草鲜美，落英缤纷。渔人甚异之，复前行，欲穷其林。林尽水源，便得一山，山有小口，仿佛若有光。便舍船，从口入。初极狭，才通人。复行数十步，豁然开朗。土地平旷，屋舍俨然，有良田、美池、桑竹之属。阡陌交通，鸡犬相闻。其中往来种作，男女衣着，悉如外人。黄发垂髫，并怡然自乐。

孟子曰："天时不如地利，地利不如人和。三里之城，七里之郭，环而攻之而不胜。夫环而攻之，必有得天时者矣，然而不胜者，是天时不如地利也。城非不高也，池非不深也，兵革非不坚利也，米粟非不多也，委而去之，是地利不如人和也。故曰：域民不以封疆之界，固国不以山溪之险，威天下不以兵革之利。得道者多助，失道者寡助。寡助之至，亲戚畔之。多助之至，天下顺之。以天下之所顺，攻亲戚之所畔，故君子有不战，战必胜矣。"

滕文公问曰："滕，小国也，间于齐楚。事齐乎？事楚乎？"孟子对曰："是谋非吾所能及也。无已，则有一焉：凿斯池也，筑斯城也，与民守之，效死而民弗去，则是可为也。"

太史公曰：诗有之："高山仰止，景行行止。"虽不能至，然心乡往之。余读孔氏书，想见其为人。适鲁，观仲尼庙堂车服礼器，诸生以时习礼其家，余祗回留之不能去云。天下君王至于贤人众矣，当时则荣，没则已焉。孔子布衣，传十余世，学者宗之。自天子王侯，中国言六艺者折中于夫子，可谓至圣矣！

孙子曰：兵者，国之大事，死生之地，存亡之道，不可不察也。故经之以五事，校之以计，而索其情：一曰道，二曰天，三曰地，四曰将，五曰法。道者，令民与上同意也，故可以与之死，可以与之生，而不畏危。天者，阴阳、寒暑、时制也。地者，远近、险易、广狭、死生也。将者，智、信、仁、勇、严也。法者，曲制、官道、主用也。凡此五者，将莫不闻，知之者胜，不知者不胜。故校之以计，而索其情，曰：主孰有道？将孰有能？天地孰得？法令孰行？兵众孰强？士卒孰练？赏罚孰明？吾以此知胜负矣。"""

# 5. 文本预处理函数
def preprocess_text(text):
    """清理文本，去除多余空格和标点问题"""
    # 统一引号
    text = text.replace('"', '')
    text = text.replace("'", "")
    
    # 替换连续空格
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# 6. 处理文本
print("=" * 60)
print("开始处理古文文本")
print("=" * 60)

# 预处理
cleaned_text = preprocess_text(ancient_text)
doc = nlp(cleaned_text)

# 7. 分析结果展示
print(f"文本总长度: {len(cleaned_text)} 字符")
print(f"分词后词数: {len(doc)}")
try:
    print(f"句子数量: {len(list(doc.sents))}")
except ValueError:
    print("句子数量: 无法计算（需要sentencizer组件）")

# 7.1 分词结果示例
print("\n" + "=" * 60)
print("分词结果示例（前100个词）:")
words = [token.text for token in doc]
for i in range(0, min(100, len(words)), 10):
    print(" ".join(words[i:i+10]))
print("...")

# 7.2 词性标注分析
print("\n" + "=" * 60)
print("词性标注统计:")
pos_counter = Counter()
for token in doc:
    pos_counter[token.pos_] += 1

print(f"{'词性':<10} {'数量':<8} {'百分比':<8} 说明")
print("-" * 50)
total = len(doc)
for pos, count in pos_counter.most_common(15):
    percentage = (count / total) * 100
    pos_explain = spacy.explain(pos) or ""
    print(f"{pos:<10} {count:<8} {percentage:>6.2f}%  {pos_explain}")

# 7.3 详细词性标注示例
print("\n" + "=" * 60)
print("详细词性标注示例（选取《桃花源记》片段）:")

# 提取《桃花源记》片段
taohua_start = cleaned_text.find("晋太元中")
taohua_end = cleaned_text.find("并怡然自乐。") + len("并怡然自乐。")
taohua_text = cleaned_text[taohua_start:taohua_end]

taohua_doc = nlp(taohua_text)
print(f"\n原文:\n{taohua_text[:100]}...")
print(f"\n词性分析（前30个词）:")
for i, token in enumerate(taohua_doc):
    if i >= 30:
        break
    pos_explain = spacy.explain(token.pos_)
    print(f"{token.text:<4} -> {token.pos_:<6} ({pos_explain})")

# 7.4 文言虚词分析
print("\n" + "=" * 60)
print("文言虚词使用统计:")
function_words = ["之", "乎", "者", "也", "矣", "焉", "哉", "曰", "于", "以", "而", "所", "其", "何", "则", "故", "是", "此"]

func_counter = Counter()
for token in doc:
    if token.text in function_words:
        func_counter[token.text] += 1

print(f"{'虚词':<4} {'次数':<6} {'占比':<8} 主要词性")
print("-" * 40)
for word in function_words:
    if word in func_counter:
        count = func_counter[word]
        percentage = (count / len(doc)) * 100
        
        # 获取该词的主要词性
        poses = []
        for token in doc:
            if token.text == word and token.pos_ not in poses:
                poses.append(token.pos_)
        
        pos_str = "/".join(poses[:2]) if poses else "N/A"
        print(f"{word:<4} {count:<6} {percentage:>6.2f}%  {pos_str}")

# 7.5 对比分析：jieba vs spaCy 原生分词
print("\n" + "=" * 60)
print("分词对比：jieba vs spaCy原生分词")

# 重新加载原生spaCy
try:
    nlp_original = spacy.load("zh_core_web_md", disable=["parser", "ner", "attribute_ruler", "lemmatizer"])
except OSError:
    nlp_original = spacy.blank("en")
sample_text = "庄周梦为胡蝶，栩栩然胡蝶也"

# jieba分词
jieba_words = list(jieba.cut(sample_text))
print(f"\n示例文本: {sample_text}")
print(f"jieba分词: {jieba_words}")

# spaCy原生分词
doc_original = nlp_original(sample_text)
spacy_words = [token.text for token in doc_original]
print(f"spaCy分词: {spacy_words}")

# 找出差异
if jieba_words != spacy_words:
    print("\n分词差异:")
    for i, (j_word, s_word) in enumerate(zip(jieba_words, spacy_words)):
        if j_word != s_word:
            print(f"  位置 {i}: jieba='{j_word}' vs spaCy='{s_word}'")

# 8. 性能测试
print("\n" + "=" * 60)
print("性能测试:")

import time

test_text = cleaned_text[:500]  # 前500字符

# 测试 jieba
start = time.time()
jieba_words = list(jieba.cut(test_text))
jieba_time = time.time() - start

# 测试 spaCy
start = time.time()
doc_test = nlp(test_text)
spacy_time = time.time() - start

# 测试原生 spaCy
start = time.time()
doc_original = nlp_original(test_text)
spacy_original_time = time.time() - start

print(f"文本长度: {len(test_text)} 字符")
print(f"jieba 分词时间: {jieba_time:.4f} 秒")
print(f"jieba+spaCy 处理时间: {spacy_time:.4f} 秒")
print(f"spaCy原生 处理时间: {spacy_original_time:.4f} 秒")

# 9. 保存分词结果
print("\n" + "=" * 60)
print("保存分词结果到文件...")

try:
    with open("ancient_text_segmented.txt", "w", encoding="utf-8") as f:
        try:
            for i, sent in enumerate(doc.sents):
                f.write(f"句子 {i+1}:\n")
                words = [token.text for token in sent]
                pos_tags = [token.pos_ for token in sent]
                
                # 每10个词一行
                for j in range(0, len(words), 10):
                    segment = words[j:j+10]
                    pos_segment = pos_tags[j:j+10]
                    f.write("  ".join([f"{w}({p})" for w, p in zip(segment, pos_segment)]) + "\n")
                f.write("\n")
        except ValueError:
            # 如果无法分割句子，保存整个文本
            f.write("全文分词结果:\n")
            words = [token.text for token in doc]
            pos_tags = [token.pos_ for token in doc]
            for j in range(0, len(words), 10):
                segment = words[j:j+10]
                pos_segment = pos_tags[j:j+10]
                f.write("  ".join([f"{w}({p})" for w, p in zip(segment, pos_segment)]) + "\n")
    print("处理完成！结果已保存到 ancient_text_segmented.txt")
except Exception as e:
    print(f"保存文件时出错: {e}")