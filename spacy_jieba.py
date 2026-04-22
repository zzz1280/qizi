# coding=utf-8
import spacy
from spacy.tokens import Doc
import jieba
import jieba.posseg as pseg
import jieba.analyse
import re
import pandas as pd
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
    def __init__(self, vocab, enable_pos=True):
        self.vocab = vocab
        self.enable_pos = enable_pos
        # 为 jieba 添加文言文专有词汇
        self._add_ancient_words()
        # 初始化词性映射
        self._init_pos_mapping()
    
    def _add_ancient_words(self):
        """添加文言文特有词汇到 jieba 词典"""
        # 人物名称
        people = [
            ("庄周", 10000, "nz"), ("胡蝶", 10000, "nz"), ("孔子", 10000, "nz"), 
            ("孟子", 10000, "nz"), ("滕文公", 10000, "nz"), ("太史公", 10000, "nz"), 
            ("仲尼", 10000, "nz"), ("孙子", 10000, "nz"), ("渔人", 5000, "n"), 
            ("武陵人", 5000, "n"), ("两小儿", 5000, "n"), ("孔氏", 5000, "nz"),
            ("夫子", 5000, "n"), ("君子", 5000, "n"), ("贤人", 5000, "n"),
            ("君王", 5000, "n"), ("天子", 5000, "n"), ("王侯", 5000, "n"),
            ("学者", 5000, "n"), ("诸生", 5000, "n")
        ]
        
        # 地名和场所
        places = [
            ("武陵", 10000, "ns"), ("桃花林", 8000, "ns"), ("鲁", 5000, "ns"),
            ("齐", 5000, "ns"), ("楚", 5000, "ns"), ("滕", 5000, "ns"),
            ("中国", 5000, "ns"), ("天下", 5000, "ns"), ("庙堂", 5000, "n"),
            ("山溪", 5000, "n"), ("封疆", 5000, "n")
        ]
        
        # 时间词
        time_words = [
            ("太元", 8000, "t"), ("日始出", 5000, "t"), ("日中", 5000, "t"),
            ("日初出", 5000, "t"), ("当时", 5000, "t"), ("时制", 5000, "n")
        ]
        
        # 成语和固定短语
        idioms = [
            ("栩栩然", 10000, "l"), ("蘧蘧然", 10000, "l"), ("自喻适志", 8000, "l"),
            ("俄然觉", 8000, "l"), ("物化", 8000, "n"), ("落英缤纷", 10000, "l"),
            ("豁然开朗", 10000, "l"), ("屋舍俨然", 8000, "l"), ("黄发垂髫", 10000, "l"),
            ("怡然自乐", 8000, "l"), ("天时不如地利", 8000, "l"), ("地利不如人和", 8000, "l"),
            ("得道者多助", 8000, "l"), ("失道者寡助", 8000, "l"), ("高山仰止", 8000, "l"),
            ("景行行止", 8000, "l"), ("沧沧凉凉", 8000, "l"), ("不可不察", 5000, "l"),
            ("委而去之", 5000, "l"), ("环而攻之", 5000, "l")
        ]
        
        # 常用动词
        verbs = [
            ("梦为", 5000, "v"), ("辩斗", 5000, "v"), ("东游", 5000, "v"),
            ("捕鱼", 5000, "v"), ("缘溪", 5000, "v"), ("忘路", 5000, "v"),
            ("忽逢", 5000, "v"), ("夹岸", 5000, "v"), ("舍船", 5000, "v"),
            ("种作", 5000, "v"), ("探汤", 5000, "v"), ("习礼", 5000, "v"),
            ("折中", 5000, "v"), ("校之", 5000, "v"), ("索其", 5000, "v"),
            ("域民", 5000, "v"), ("固国", 5000, "v"), ("威天下", 5000, "v"),
            ("凿斯", 5000, "v"), ("筑斯", 5000, "v"), ("效死", 5000, "v"),
            ("祗回", 5000, "v"), ("想见", 5000, "v"), ("适鲁", 5000, "v")
        ]
        
        # 常用名词
        nouns = [
            ("车盖", 8000, "n"), ("盘盂", 8000, "n"), ("良田", 8000, "n"),
            ("美池", 8000, "n"), ("桑竹", 8000, "n"), ("阡陌", 8000, "n"),
            ("交通", 5000, "n"), ("鸡犬", 5000, "n"), ("芳草", 5000, "n"),
            ("杂树", 5000, "n"), ("车服", 5000, "n"), ("礼器", 5000, "n"),
            ("兵革", 5000, "n"), ("米粟", 5000, "n"), ("兵众", 5000, "n"),
            ("士卒", 5000, "n"), ("曲制", 5000, "n"), ("官道", 5000, "n"),
            ("主用", 5000, "n"), ("阴阳", 5000, "n"), ("寒暑", 5000, "n"),
            ("赏罚", 5000, "n"), ("胜负", 5000, "n"), ("存亡", 5000, "n"),
            ("死生", 5000, "n"), ("远近", 5000, "n"), ("险易", 5000, "n"),
            ("广狭", 5000, "n"), ("封疆", 5000, "n"), ("山溪", 5000, "n"),
            ("六艺", 5000, "n"), ("布衣", 5000, "n"), ("庙堂", 5000, "n")
        ]
        
        # 形容词
        adjectives = [
            ("鲜美", 8000, "a"), ("平旷", 8000, "a"), ("极狭", 5000, "a"),
            ("坚利", 5000, "a"), ("多知", 5000, "a")
        ]
        
        # 整合所有词汇
        all_words = people + places + time_words + idioms + verbs + nouns + adjectives
        
        # 添加到jieba词典
        for word, freq, tag in all_words:
            jieba.add_word(word, freq=freq, tag=tag)
    
    def _init_pos_mapping(self):
        """初始化词性映射，用于更准确地标注文言文中的词汇"""
        # jieba词性到spaCy词性的映射
        self.pos_mapping = {
            # 名词类
            'n': 'NOUN',      # 名词
            'nr': 'PROPN',    # 人名
            'ns': 'PROPN',    # 地名
            'nt': 'PROPN',    # 机构团体
            'nz': 'PROPN',    # 其他专名
            'nl': 'NOUN',     # 名词性惯用语
            'ng': 'NOUN',     # 名词性语素
            
            # 动词类
            'v': 'VERB',      # 动词
            'vd': 'VERB',     # 副动词
            'vn': 'VERB',     # 名动词
            'vf': 'VERB',     # 趋向动词
            'vx': 'VERB',     # 形式动词
            'vi': 'VERB',     # 不及物动词
            'vl': 'VERB',     # 动词性惯用语
            'vg': 'VERB',     # 动词性语素
            
            # 形容词类
            'a': 'ADJ',       # 形容词
            'ad': 'ADJ',      # 副形词
            'an': 'ADJ',      # 名形词
            'ag': 'ADJ',      # 形容词性语素
            'al': 'ADJ',      # 形容词性惯用语
            
            # 副词
            'd': 'ADV',       # 副词
            
            # 介词
            'p': 'ADP',       # 介词
            
            # 连词
            'c': 'CCONJ',     # 连词
            
            # 代词
            'r': 'PRON',      # 代词
            
            # 助词
            'u': 'PART',      # 助词
            'uzhe': 'PART',   # 着
            'ule': 'PART',    # 了、喽
            'uguo': 'PART',   # 过
            
            # 叹词
            'e': 'INTJ',      # 叹词
            
            # 数词
            'm': 'NUM',       # 数词
            
            # 量词
            'q': 'NUM',       # 量词
            
            # 标点符号
            'x': 'PUNCT',     # 标点符号
            'w': 'PUNCT',     # 标点
        }
        
        # 文言文专用词性覆盖表（优先级最高）
        self.ancient_pos_override = {
            # 虚词（助词）
            '之': 'PART', '乎': 'PART', '者': 'PART', '也': 'PART', 
            '矣': 'PART', '焉': 'PART', '哉': 'PART', '耳': 'PART', 
            '耶': 'PART', '邪': 'PART', '与': 'PART',
            
            # 动词
            '曰': 'VERB', '为': 'VERB', '行': 'VERB', '去': 'VERB',
            '来': 'VERB', '至': 'VERB', '见': 'VERB', '闻': 'VERB',
            '知': 'VERB', '觉': 'VERB', '梦': 'VERB', '游': 'VERB',
            '问': 'VERB', '对': 'VERB', '曰': 'VERB', '云': 'VERB',
            '无': 'VERB', '有': 'VERB', '得': 'VERB', '舍': 'VERB',
            '入': 'VERB', '出': 'VERB', '攻': 'VERB', '守': 'VERB',
            '战': 'VERB', '胜': 'VERB', '败': 'VERB', '死': 'VERB',
            '生': 'VERB', '荣': 'VERB', '没': 'VERB', '传': 'VERB',
            '宗': 'VERB', '读': 'VERB', '观': 'VERB', '留': 'VERB',
            '去': 'VERB', '筑': 'VERB', '凿': 'VERB', '效': 'VERB',
            '令': 'VERB', '意': 'VERB', '畏': 'VERB', '危': 'VERB',
            '索': 'VERB', '校': 'VERB', '经': 'VERB', '察': 'VERB',
            
            # 名词
            '道': 'NOUN', '天': 'NOUN', '地': 'NOUN', '将': 'NOUN',
            '法': 'NOUN', '民': 'NOUN', '国': 'NOUN', '兵': 'NOUN',
            '城': 'NOUN', '池': 'NOUN', '路': 'NOUN', '林': 'NOUN',
            '山': 'NOUN', '水': 'NOUN', '口': 'NOUN', '船': 'NOUN',
            '人': 'NOUN', '树': 'NOUN', '步': 'NOUN', '业': 'NOUN',
            '光': 'NOUN', '狭': 'ADJ',
            
            # 代词
            '其': 'PRON', '此': 'PRON', '是': 'PRON', '斯': 'PRON',
            '夫': 'PRON', '彼': 'PRON', '吾': 'PRON', '我': 'PRON',
            '汝': 'PRON', '尔': 'PRON', '孰': 'PRON',
            
            # 连词
            '而': 'CCONJ', '则': 'CCONJ', '故': 'CCONJ', '然': 'CCONJ',
            '虽': 'CCONJ', '或': 'CCONJ', '且': 'CCONJ', '乃': 'CCONJ',
            
            # 介词
            '于': 'ADP', '以': 'ADP', '自': 'ADP', '从': 'ADP',
            '向': 'ADP', '在': 'ADP', '因': 'ADP',
            
            # 副词
            '甚': 'ADV', '忽': 'ADV', '复': 'ADV', '欲': 'ADV',
            '便': 'ADV', '初': 'ADV', '才': 'ADV', '并': 'ADV',
            '不': 'ADV', '非': 'ADV', '未': 'ADV', '莫': 'ADV',
            '必': 'ADV', '固': 'ADV', '亦': 'ADV', '皆': 'ADV',
            '俱': 'ADV', '咸': 'ADV', '悉': 'ADV', '尽': 'ADV',
            '颇': 'ADV', '殊': 'ADV', '尤': 'ADV', '益': 'ADV',
            '愈': 'ADV', '弥': 'ADV', '既': 'ADV', '已': 'ADV',
            '尝': 'ADV', '曾': 'ADV', '适': 'ADV', '方': 'ADV',
            '正': 'ADV', '当': 'ADV', '将': 'ADV', '且': 'ADV',
            '即': 'ADV', '则': 'ADV', '乃': 'ADV', '遂': 'ADV',
            '因': 'ADV', '仍': 'ADV', '俄': 'ADV', '旋': 'ADV',
            '寻': 'ADV', '寻而': 'ADV', '已而': 'ADV', '未几': 'ADV',
            '少': 'ADV', '顷': 'ADV', '刻': 'ADV', '忽': 'ADV',
            '遽': 'ADV', '立': 'ADV', '即': 'ADV', '辄': 'ADV',
            '每': 'ADV', '常': 'ADV', '素': 'ADV', '雅': 'ADV',
            '本': 'ADV', '原': 'ADV', '盖': 'ADV', '殆': 'ADV',
            '庶': 'ADV', '幸': 'ADV', '请': 'ADV', '愿': 'ADV',
            '惟': 'ADV', '唯': 'ADV', '但': 'ADV', '仅': 'ADV',
            '止': 'ADV', '特': 'ADV', '直': 'ADV', '徒': 'ADV',
            '空': 'ADV', '漫': 'ADV', '枉': 'ADV', '虚': 'ADV',
            '反': 'ADV', '翻': 'ADV', '却': 'ADV', '倒': 'ADV',
            '转': 'ADV', '更': 'ADV', '益': 'ADV', '弥': 'ADV',
            '愈': 'ADV', '尤': 'ADV', '最': 'ADV', '极': 'ADV',
            '甚': 'ADV', '太': 'ADV', '过': 'ADV', '颇': 'ADV',
            '殊': 'ADV', '良': 'ADV', '信': 'ADV', '实': 'ADV',
            '诚': 'ADV', '确': 'ADV', '真': 'ADV', '果': 'ADV',
            '信': 'ADV', '必': 'ADV', '断': 'ADV', '决': 'ADV',
            '定': 'ADV', '准': 'ADV', '宜': 'ADV', '当': 'ADV',
            '应': 'ADV', '合': 'ADV', '该': 'ADV', '须': 'ADV',
            '需': 'ADV', '要': 'ADV', '必': 'ADV', '定': 'ADV',
        }
    
    def __call__(self, text):
        try:
            # 使用 jieba 精确模式分词，启用 HMM 模型
            words = list(jieba.cut(text, cut_all=False, HMM=True))
            
            # 清理空格和空字符
            words = [w.strip() for w in words if w.strip()]
            
            # 为每个词分配词性
            if self.enable_pos:
                # 使用jieba的词性标注
                pos_tags = []
                word_flags = list(pseg.cut(text))
                
                for word in words:
                    try:
                        # 查找对应的词性
                        found_pos = None
                        for flag in word_flags:
                            if flag.word == word:
                                found_pos = self._map_pos(flag.flag, word)
                                break
                        
                        if found_pos is None:
                            found_pos = self._infer_pos(word)
                        
                        pos_tags.append(found_pos)
                    except Exception:
                        pos_tags.append("NOUN")
                
                # 创建带有词性标注的 spaCy Doc 对象
                return Doc(self.vocab, words=words, pos=pos_tags)
            else:
                # 创建基本的 spaCy Doc 对象
                return Doc(self.vocab, words=words)
        except Exception as e:
            # 分词失败时返回原始文本作为单个词
            print(f"分词过程中出错: {e}")
            return Doc(self.vocab, words=[text])
    
    def _map_pos(self, jieba_pos, word=None):
        """将jieba词性映射到spaCy词性"""
        # 首先检查文言文专用词性覆盖表（优先级最高）
        if word and word in self.ancient_pos_override:
            return self.ancient_pos_override[word]
        
        if jieba_pos in self.pos_mapping:
            return self.pos_mapping[jieba_pos]
        # 默认返回名词
        return "NOUN"
    
    def _infer_pos(self, word):
        """根据词汇特征推断词性"""
        # 标点符号
        if len(word) == 1 and word in "，。！？；：""''（）《》【】":
            return "PUNCT"
        
        # 数字
        if word.isdigit():
            return "NUM"
        
        # 文言文虚词
        if word in ["之", "乎", "者", "也", "矣", "焉", "哉", "耳", "耶", "邪"]:
            return "PART"
        
        # 常见连词
        if word in ["而", "则", "故", "然", "虽", "或", "且", "既", "因"]:
            return "CCONJ"
        
        # 常见代词
        if word in ["其", "我", "汝", "尔", "吾", "余", "予", "彼", "此", "是", "斯", "夫"]:
            return "PRON"
        
        # 常见介词
        if word in ["于", "以", "为", "与", "从", "缘", "向", "在", "到", "自"]:
            return "ADP"
        
        # 常见副词
        if word in ["甚", "忽", "仿佛", "若", "初", "才", "复", "欲", "便", "并", "不", "未", "非", "必", "固"]:
            return "ADV"
        
        # 长度大于等于3的可能是专有名词
        if len(word) >= 3:
            return "PROPN"
        
        # 默认名词
        return "NOUN"

# 3. 替换 spaCy 的分词器
nlp.tokenizer = JiebaTokenizer(nlp.vocab)

# 4. 配置参数
class Config:
    """配置参数"""
    # 分词参数
    CUT_ALL = False  # 是否使用全模式分词
    USE_HMM = True  # 是否使用HMM模型
    
    # 关键词提取参数
    TOP_K_KEYWORDS = 20  # 提取关键词数量
    KEYWORD_POS = ('n', 'vn', 'v')  # 关键词词性过滤
    
    # 摘要参数
    SUMMARY_SENTENCES = 3  # 摘要句子数量
    
    # 性能参数
    ENABLE_PROGRESS = True  # 是否显示进度

config = Config()

# 5. 读取古文文本
ancient_text = """昔者庄周梦为胡蝶，栩栩然胡蝶也，自喻适志与！不知周也。俄然觉，则蘧蘧然周也。不知周之梦为胡蝶与，胡蝶之梦为周与？周与胡蝶，则必有分矣。此之谓物化。

孔子东游，见两小儿辩斗，问其故。一儿曰："我以日始出时去人近，而日中时远也。"一儿以日初出远，而日中时近也。一儿曰："日初出大如车盖，及日中则如盘盂，此不为远者小而近者大乎？"一儿曰："日初出沧沧凉凉，及其日中如探汤，此不为近者热而远者凉乎？"孔子不能决也。两小儿笑曰："孰为汝多知乎？"

晋太元中，武陵人捕鱼为业。缘溪行，忘路之远近。忽逢桃花林，夹岸数百步，中无杂树，芳草鲜美，落英缤纷。渔人甚异之，复前行，欲穷其林。林尽水源，便得一山，山有小口，仿佛若有光。便舍船，从口入。初极狭，才通人。复行数十步，豁然开朗。土地平旷，屋舍俨然，有良田、美池、桑竹之属。阡陌交通，鸡犬相闻。其中往来种作，男女衣着，悉如外人。黄发垂髫，并怡然自乐。

孟子曰："天时不如地利，地利不如人和。三里之城，七里之郭，环而攻之而不胜。夫环而攻之，必有得天时者矣，然而不胜者，是天时不如地利也。城非不高也，池非不深也，兵革非不坚利也，米粟非不多也，委而去之，是地利不如人和也。故曰：域民不以封疆之界，固国不以山溪之险，威天下不以兵革之利。得道者多助，失道者寡助。寡助之至，亲戚畔之。多助之至，天下顺之。以天下之所顺，攻亲戚之所畔，故君子有不战，战必胜矣。"

滕文公问曰："滕，小国也，间于齐楚。事齐乎？事楚乎？"孟子对曰："是谋非吾所能及也。无已，则有一焉：凿斯池也，筑斯城也，与民守之，效死而民弗去，则是可为也。"

太史公曰：诗有之："高山仰止，景行行止。"虽不能至，然心乡往之。余读孔氏书，想见其为人。适鲁，观仲尼庙堂车服礼器，诸生以时习礼其家，余祗回留之不能去云。天下君王至于贤人众矣，当时则荣，没则已焉。孔子布衣，传十余世，学者宗之。自天子王侯，中国言六艺者折中于夫子，可谓至圣矣！

孙子曰：兵者，国之大事，死生之地，存亡之道，不可不察也。故经之以五事，校之以计，而索其情：一曰道，二曰天，三曰地，四曰将，五曰法。道者，令民与上同意也，故可以与之死，可以与之生，而不畏危。天者，阴阳、寒暑、时制也。地者，远近、险易、广狭、死生也。将者，智、信、仁、勇、严也。法者，曲制、官道、主用也。凡此五者，将莫不闻，知之者胜，不知者不胜。故校之以计，而索其情，曰：主孰有道？将孰有能？天地孰得？法令孰行？兵众孰强？士卒孰练？赏罚孰明？吾以此知胜负矣。"""

# 6. 文本预处理函数
def preprocess_text(text):
    """清理文本，去除多余空格和标点问题"""
    # 统一引号
    text = text.replace('"', '')
    text = text.replace("'", "")
    
    # 替换连续空格
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# 7. 处理文本
print("=" * 60)
print("开始处理古文文本")
print("=" * 60)

# 预处理
cleaned_text = preprocess_text(ancient_text)
doc = nlp(cleaned_text)

# 8. 分析结果展示
print(f"文本总长度: {len(cleaned_text)} 字符")
print(f"分词后词数: {len(doc)}")
try:
    print(f"句子数量: {len(list(doc.sents))}")
except ValueError:
    print("句子数量: 无法计算（需要sentencizer组件）")

# 8.1 分词结果示例
print("\n" + "=" * 60)
print("分词结果示例（前100个词）:")
words = [token.text for token in doc]
for i in range(0, min(100, len(words)), 10):
    print(" ".join(words[i:i+10]))
print("...")

# 8.2 词性标注分析
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

# 8.3 详细词性标注示例
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

# 8.4 文言虚词分析
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

# 8.5 对比分析：jieba vs spaCy 原生分词
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

# 9. 性能测试
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

# 10. 保存分词结果
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

# 11. 关键词提取
print("\n" + "=" * 60)
print("关键词提取:")

if config.ENABLE_PROGRESS:
    print("正在提取TF-IDF关键词...")

# 使用TF-IDF算法提取关键词
tfidf_keywords = []
try:
    tfidf_keywords = jieba.analyse.extract_tags(
        cleaned_text, 
        topK=config.TOP_K_KEYWORDS, 
        withWeight=True, 
        allowPOS=config.KEYWORD_POS
    )
    
    print("\nTF-IDF 关键词:")
    for keyword, weight in tfidf_keywords[:10]:
        print(f"{keyword:<10} {weight:.4f}")
except Exception as e:
    print(f"TF-IDF关键词提取失败: {e}")

if config.ENABLE_PROGRESS:
    print("\n正在提取TextRank关键词...")

# 使用TextRank算法提取关键词
textrank_keywords = []
try:
    textrank_keywords = jieba.analyse.textrank(
        cleaned_text, 
        topK=config.TOP_K_KEYWORDS, 
        withWeight=True, 
        allowPOS=config.KEYWORD_POS
    )
    
    print("\nTextRank 关键词:")
    for keyword, weight in textrank_keywords[:10]:
        print(f"{keyword:<10} {weight:.4f}")
except Exception as e:
    print(f"TextRank关键词提取失败: {e}")

# 12. 文本摘要
print("\n" + "=" * 60)
print("文本摘要:")

if config.ENABLE_PROGRESS:
    print("正在生成文本摘要...")

def generate_summary(text, num_sentences=config.SUMMARY_SENTENCES):
    """基于句子重要性生成文本摘要"""
    try:
        # 分割句子
        sentences = re.split('[。！？；]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return "无内容可摘要"
        
        # 计算每个句子的重要性得分
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            try:
                # 提取句子中的关键词
                keywords = jieba.analyse.extract_tags(sentence, topK=5, allowPOS=config.KEYWORD_POS)
                # 计算得分（关键词数量 + 句子长度权重）
                score = len(keywords) + len(sentence) / 20.0
                sentence_scores[i] = score
            except Exception:
                # 计算失败时使用默认得分
                sentence_scores[i] = len(sentence) / 20.0
        
        # 排序句子并选择前N个
        sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        top_sentences = [sentences[i] for i, _ in sorted_sentences[:num_sentences]]
        
        # 按照原顺序排列
        top_sentences_indices = [i for i, _ in sorted_sentences[:num_sentences]]
        top_sentences_indices.sort()
        summary = "。".join([sentences[i] for i in top_sentences_indices]) + "。"
        
        return summary
    except Exception as e:
        print(f"生成摘要时出错: {e}")
        return "摘要生成失败"

summary = generate_summary(cleaned_text)
print(f"摘要:\n{summary}")

# 13. 模型评估
print("\n" + "=" * 70)
print("9. 模型评估")
print("=" * 70)

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

# 为《桃花源记》片段创建标准标注（人工标注的标准答案）
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
token_precision, token_recall, token_f1 = evaluate_tokenization(taohua_doc, ground_truth_tokens)
print()

# 评估词性标注
print("2. 词性标注评估")
print("-" * 40)
pos_precision, pos_recall, pos_f1 = evaluate_pos_tagging(taohua_doc, ground_truth_tags)
print()

# 评估命名实体识别
print("3. 命名实体识别评估")
print("-" * 40)
ner_precision, ner_recall, ner_f1 = evaluate_ner(taohua_doc, ground_truth_entities)
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
import pandas as pd

eval_results = {
    '评估项': ['分词', '词性标注', '实体识别'],
    '准确率': [token_precision, pos_precision, ner_precision],
    '召回率': [token_recall, pos_recall, ner_recall],
    'F1值': [token_f1, pos_f1, ner_f1]
}

eval_df = pd.DataFrame(eval_results)
eval_df.to_csv('evaluation_results.csv', index=False, encoding='utf-8-sig')
print(f"\n评估结果已保存到: evaluation_results.csv")

# 14. 保存增强分析结果
print("\n" + "=" * 60)
print("保存增强分析结果...")

try:
    with open("ancient_text_analysis.txt", "w", encoding="utf-8") as f:
        f.write("# 文言文分析报告\n\n")
        
        # 基本信息
        f.write("## 基本信息\n")
        f.write(f"文本总长度: {len(cleaned_text)} 字符\n")
        f.write(f"分词后词数: {len(doc)}\n")
        try:
            f.write(f"句子数量: {len(list(doc.sents))}\n\n")
        except ValueError:
            f.write("句子数量: 无法计算（需要sentencizer组件）\n\n")
        
        # 关键词
        f.write("## 关键词提取\n")
        f.write("### TF-IDF 关键词\n")
        for keyword, weight in tfidf_keywords[:15]:
            f.write(f"{keyword}: {weight:.4f}\n")
        
        f.write("\n### TextRank 关键词\n")
        for keyword, weight in textrank_keywords[:15]:
            f.write(f"{keyword}: {weight:.4f}\n")
        
        # 摘要
        f.write("\n## 文本摘要\n")
        f.write(f"{summary}\n\n")
        
        # 词性统计
        f.write("## 词性统计\n")
        f.write(f"{'词性':<10} {'数量':<8} {'百分比':<8}\n")
        f.write("-" * 30 + "\n")
        total = len(doc)
        for pos, count in pos_counter.most_common(15):
            percentage = (count / total) * 100
            f.write(f"{pos:<10} {count:<8} {percentage:>6.2f}%\n")
    
    print("增强分析结果已保存到 ancient_text_analysis.txt")
except Exception as e:
    print(f"保存增强分析结果时出错: {e}")

# 15. 生成分词和词性标注CSV文档
print("\n" + "=" * 60)
print("生成分词和词性标注CSV文档...")

try:
    # 词性解释映射
    pos_explanations = {
        'NOUN': '名词', 'VERB': '动词', 'ADJ': '形容词', 'ADV': '副词',
        'ADP': '介词', 'CCONJ': '连词', 'PART': '助词', 'PRON': '代词',
        'PROPN': '专有名词', 'NUM': '数词', 'PUNCT': '标点', 'X': '其他'
    }
    
    # 创建详细的数据表
    data = []
    for i, token in enumerate(doc):
        data.append({
            '序号': i+1,
            '词语': token.text,
            '词性': token.pos_,
            '词性解释': pos_explanations.get(token.pos_, '未知'),
            '是否标点': token.is_punct,
            '是否空格': token.is_space,
            '是否数字': token.like_num,
            '词长': len(token.text)
        })
    
    df = pd.DataFrame(data)
    
    # 保存到CSV
    df.to_csv('segmentation_results.csv', index=False, encoding='utf-8-sig')
    print(f"分词和词性标注结果已保存到: segmentation_results.csv")
    print(f"共保存 {len(df)} 个分词结果")
except Exception as e:
    print(f"保存CSV文件时出错: {e}")

# 16. 最终报告
print("\n" + "=" * 60)
print("分析完成！")
print("=" * 60)
print("生成的文件：")
print("1. ancient_text_segmented.txt - 分词和词性标注结果")
print("2. ancient_text_analysis.txt - 增强分析报告")
print("3. segmentation_results.csv - 分词和词性标注详细数据表")
print("4. evaluation_results.csv - 模型评估结果")
print("\n请查看这些文件以获取详细的分析结果。")
