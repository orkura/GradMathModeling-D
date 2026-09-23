# LaTeX 论文协作说明

正式论文入口为 `MathModel.tex`。每个小节单独存放在一个 `.tex` 文件中，使用 `\input` 组装全文。
文件拆分不改变标题层级：一级标题使用 `\section`，二级标题使用 `\subsection`，三级标题使用 `\subsubsection`。

当前版本是写作骨架，正文中的“待撰写”和参考文献占位均需在定稿前替换。
各文件中的写作要点是注释，不会出现在 PDF 中。

公共配置包含当前模板的封面条件闭合补丁及书签间距处理，主文件在封面期间关闭页码锚点以避免重复。
升级模板时需重新核对这些兼容设置。
题目和队员信息沿用 `example.tex`，该示例继续保留供排版参考。

## 文件职责与负责人

| 文件或目录 | 内容 | 主要负责人 |
| --- | --- | --- |
| `MathModel.tex` | 封面信息、全文顺序、参考文献入口 | 整合人员，待认领 |
| `preamble.tex` | 宏包、公共命令、编号和目录设置 | 整合人员，待认领 |
| 各级 `index.tex` | 章节标题和子文件引用顺序 | 整合人员，待认领 |
| `sections/00-abstract.tex` | 摘要与关键词 | 待认领 |
| `sections/01-introduction/background.tex` | 1.1 问题背景 | 待认领 |
| `sections/01-introduction/restatement.tex` | 1.2 问题重述 | 待认领 |
| `sections/02-analysis.tex` | 二、总体分析 | 待认领 |
| `sections/03-assumptions.tex` | 三、模型假设 | 待认领 |
| `sections/04-notation.tex` | 四、符号说明 | 待认领 |
| `sections/05-models/problem1/` | 5.1 单点往返运输能力与货箱组批 | 各小节待认领 |
| `sections/05-models/problem2/` | 5.2 异构无人机多点多架次运输调度 | 各小节待认领 |
| `sections/05-models/problem3/` | 5.3 通信约束下运输与中继联合调度 | 各小节待认领 |
| `sections/05-models/problem4/` | 5.4 救援任务分区与资源配置 | 各小节待认领 |
| `sections/06-validation/error.tex` | 6.1 误差分析及可行性检查汇总 | 待认领 |
| `sections/06-validation/sensitivity.tex` | 6.2 灵敏度分析 | 待认领 |
| `sections/07-evaluation/strengths.tex` | 7.1 模型的优点 | 待认领 |
| `sections/07-evaluation/limitations.tex` | 7.2 模型的缺点 | 待认领 |
| `sections/08-improvements/improvements.tex` | 8.1 模型的改进 | 待认领 |
| `sections/08-improvements/extensions.tex` | 8.2 模型的推广 | 待认领 |
| `reference.bib` | 参考文献条目 | 整合人员，待认领 |
| `figures/` | 论文图片 | 随对应小节分工 |

每个问题目录均包含以下五个文件：

```text
problem1/                 # problem2、problem3、problem4 结构相同
├─ index.tex              # 本问题标题与子文件引用
├─ analysis.tex           # 具体分析
├─ preparation.tex        # 模型准备
├─ formulation.tex        # 模型建立
└─ solution.tex           # 模型求解
```

每个正文文件指定一名主要负责人，可在该文件顶部的“负责人”注释中填写姓名。
一名成员可以负责多个文件，同一问题也可以按小节分工。
结构稳定后，日常写作通常只修改负责的正文文件及其图片；共享文件由整合人员统一调整。

## 编译

从仓库根目录进入论文目录，使用 XeLaTeX：

```powershell
cd Latex
latexmk -xelatex -outdir=build MathModel.tex
```

生成的 PDF 为 `build/MathModel.pdf`，中间文件也保存在 `build/`。
该目录已经被仓库忽略，不应把个人编译产物加入 Git。
正式提交稿如需另行保存，应按仓库协作规范与对应源文件一同提交。

各小节首行的 `% !TEX root = ...` 指向统一入口，支持该指令的编辑器可以从小节启动全文编译。
命令行仍应在 `Latex/` 中编译 `MathModel.tex`。
小节文件不是独立文档，不添加 `\documentclass`、`\begin{document}` 或 `\end{document}`。

## 写作与交叉引用

- `\input` 路径以 `Latex/` 为基准，即使命令写在深层小节中，也应使用 `sections/...`。
- 保留每个小节已有的标题与标签；不要把所有小节都改成 `\section`，也不要手写章节序号。
- 一级标题显示为“一、”，二级和三级显示为“1.1”和“1.1.1”；目录展示到三级。
- 公式、图和表按一级章节使用阿拉伯数字编号。
- 标签使用问题和含义作为前缀，例如 `eq:q1:energy`、`fig:q2:schedule`、`tab:q3:resources`。
  同一问题有多人撰写时，可进一步使用 `eq:q1:formulation:energy`。
- 通过 `\ref{...}` 引用其他文件中的章节、公式和图表，不手写编号。
- 图片建议放入 `figures/problem1/` 至 `figures/problem4/` 等对应目录。
  例如文件 `figures/problem1/routes.pdf` 可用 `\includegraphics{problem1/routes}` 引用。
  模板已配置 `figures/` 为图片搜索路径。
- 每句或每个逻辑单元单独一行，段落之间留空行，便于查看 Git 差异。
- 每个文件中的环境应自行配对闭合，不把一个表格、公式或列表跨文件拆开。
- 新增宏包和公共命令统一放入 `preamble.tex`，由整合人员处理依赖及重名。

## 各问题的数据交接

总体分析说明完整技术路线；各问的具体分析说明本问的决策对象、难点及方法选择。
公共假设和符号统一维护，题目规定与自行简化应明确区分。

问题一的基础物理计算可以供后续问题引用，但其“单服务区直接往返”限制不适用于问题二的多点运输。
问题二引入实体无人机、共享电池和时序；问题三继续加入中继与连续通信约束。
后续问题应明确复用哪些计算、增加哪些决策及约束。

问题四必须读取并固定问题三的货箱组批、服务区访问顺序、运输与中继任务安排及通信保障关系。
同一运输架次涉及的服务区必须属于同一任务组，各组执行期间资源不得跨组调配。
问题三交付时应明确数据文件、版本、字段、单位和生成方式；方案更新后需同步检查问题四结果。

第六章除了误差和灵敏度，还应汇总或引用货箱交付、时限、资源占用及连续通信的可行性检查。
没有实测真值时，不把内部一致性检查表述为实测精度。

## 参考文献

现有 `reference.bib` 是模板示例文献。骨架默认显示参考文献占位，不把示例作为本题依据。
开始引用真实文献时，由整合人员完成以下步骤：

1. 在 `reference.bib` 中整理经过核实的实际文献，移除不再需要的示例条目。
2. 将 `preamble.tex` 中的 `\paperbibliographyfalse` 改为 `\paperbibliographytrue`。
3. 正文使用 `\cite{文献键}`，并按上述命令重新编译；latexmk 会按需运行 BibTeX。

启用时应至少有一条有效引用，避免无引用导致 BibTeX 报错。
模板会自动把参考文献加入目录，无需在正文重复添加参考文献标题。

## Git 协作

分支命名、提交和合并遵循仓库根目录的 [CONTRIBUTING.md](../CONTRIBUTING.md)。
个人草稿与正式成果的流转遵循 [README.md](../README.md)。

合并前完整编译全文，检查缺图、缺字体、未解析引用、重复标签和编号。
确认只提交源文件、文献与所需图片，避免多人同时更新同一个编译 PDF。
