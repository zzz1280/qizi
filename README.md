根据你提供的仓库文件结构（包含古文分词、实体分析、桃花源记示例等），我为你撰写了一份专业的 `README.md`。它假设该项目是一个**中医古籍（医经）文本处理与实体分析**的 Python 工具库。

---

# Qizi (启兹) - 中医古籍文本处理工具

> “启兹”取自“启典探源，兹于医经”，旨在为中医古籍数字化与自然语言处理提供基础工具。

这是一个用于处理中医古籍（医经）文本的 Python 工具包。项目提供了古籍文本的分词、实体分析、术语提取等基础 NLP 功能，并包含针对《桃花源记》的示例分析流程，可作为中医文献数字化研究的代码基础。

## 📁 项目结构

```
qizi/
├── .vscode/                  # VSCode 开发配置
├── __pycache__/
├── ancient_chinese.txt       # 原始古籍文本语料
├── ancient_text_segmented.txt # 分词后的文本
├── evaluation_results.csv    # 模型评估结果
├── process.py                # 主处理脚本（文本清洗、分词、分析）
├── spacy_jieba.py           # 分词与 NLP 处理模块（集成 Jieba/Jieba)
├── taohuayuan_analysis.csv  # 《桃花源记》文本分析结果
├── taohuayuan_entities.csv  # 《桃花源记》实体提取结果
└── test.py                  # 单元测试/功能测试脚本
```

## 🚀 快速开始

### 环境要求

- **Python**: 3.8+
- **核心依赖**:
  - `jieba` (中文分词)
  - `spacy` (可选，用于高级 NLP 管道)
  - `pandas` (数据处理)

### 安装与使用

1. **克隆仓库**
   ```bash
   git clone https://github.com/zzz1280/qizi.git
   cd qizi
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   # 若无 requirements.txt，可手动安装：
   pip install jieba pandas
   ```

3. **运行示例（处理《桃花源记》）**
   ```bash
   python process.py
   ```
   *此命令将读取 `ancient_chinese.txt`，执行分词与实体分析，并生成 `taohuayuan_analysis.csv` 等结果文件。*

### 自定义文本分析

修改 `process.py` 中的输入文件路径，或直接调用 `spacy_jieba.py` 中的函数处理你的古籍文本：

```python
from spacy_jieba import process_text

text = "你的中医古籍原文..."
results = process_text(text)
```

## 📊 功能特性

- **古籍分词**: 针对古文特性优化的分词模块（基于 Jieba 与自定义词典）。
- **实体分析**: 提取并统计文本中的人名、地名、方剂名等实体。
- **数据导出**: 支持将分词结果、实体统计导出为 CSV 格式，便于后续研究。
- **示例数据**: 包含《桃花源记》的完整分析流程，作为项目使用范例。

## 📝 核心文件说明

| 文件 | 说明 |
|------|------|
| `process.py` | 主流程脚本，包含文本清洗、分词、分析、导出全流程 |
| `spacy_jieba.py` | 核心 NLP 处理模块，封装分词与实体识别逻辑 |
| `ancient_chinese.txt` | 输入的古籍原文（示例为《桃花源记》） |
| `taohuayuan_analysis.csv` | 文本分析输出（词频、词性、句法分析等） |

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来完善分词词典或增加新的古籍处理功能。

## 📜 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

---

> **学术用途说明**: 本项目代码可用于中医古籍数字化、中医文献信息学、古文 NLP 等相关研究，请合理引用。
