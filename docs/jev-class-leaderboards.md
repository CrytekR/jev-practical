# Jev 类模型评测榜单：公信力与最强开源模型

记录日期：2026-10-26（Asia/Shanghai）。数字来自当日打开的榜单页面、论文和聚合文件，不是模型厂商新闻稿的复述。本报告没有重跑任何模型。

## 结论

**研究上最值得当作「最强开源 Jev 类模型」的是 [`baselabs/mammoth-1`](https://huggingface.co/baselabs/mammoth-1)（27B，基座 Qwen3.5-27B）。** Hugging Face 模型卡的许可证字段是 MIT，仓库不门控。

这个结论钉在两个公开比较上，而且两个比较说的是同一件事的不同侧面：

| 比较 | 口径 | mammoth-1 | 紧挨着它的模型 |
| --- | --- | --- | --- |
| JevBench，方法 v1.7，页面抓取于 2026-10-26 | Capability = 智力与校准的平均。去掉「不超过 Jev 两倍成本」这道资格线之后，按页面排序 | **85.2**（智力 76.2，校准 94.3，约 $0.10 / 千次决策，成本约为 Jev 的 3.2 倍，因此被官方 Jev 类资格线排除） | KnowLine 31B-Gen1 页面也印成 **85.2**（智力 77.4，校准 93.0）。智力更高的是 KnowLine，校准更高的是 mammoth-1。本次没有打开 KnowLine 的许可证 |
| Decision Index 0.4.1 公开分 | 37 个公开基准、110,201 次请求、109,759 次可计分；先做机会校正。聚合文件由本报告下载后自行排序 | **75.34，229 行中的第 1** | Jeffan-31b-1710 的公开分为 73.13。闭源参照 Jev 1.14 的公开分是 60.70；它的 Full 分 60.24，Full 名次第 41 |

同一份 Index 文件里，头条 Full 分不含公开分，公式是 `0.60 × 私有同技能 + 0.40 × 私有新领域`。Full 分第一是 **Fastino GLiDE 1.2**（71.52），mammoth-1 是第 3（68.44）。私有题只有维护者能跑，所以 Full 分不能单独封王。mammoth-1 的价值在于：公开分第一，Full 分仍留在前三，没有出现「公开分很高、私有题塌掉」的那种落差。

若把「Jev 类」收成 JevBench 自己的资格线（成本与调整后延迟都不超过 Jev v1.5 参照的 2 倍），资格线内的开放权重第一名换成 **H2O-Lightning-31B v1.0.7**，Capability **84.3**（智力 79.3，校准 89.4，约 $0.044 / 千次）。它的权重在 [`h2oai/h2o-lightning-31b`](https://huggingface.co/h2oai/h2o-lightning-31b)，许可证是 H2O Model Non-Commercial Licence v1.1：非商业、非生产、需要在 Hub 上接受条款。模型卡写的发布修订是 `v1.2.2`（checkpoint step 22276），并写明榜上的 v1.0.7 跑的是这同一套权重。它是「又快又便宜的开放权重第一」，不是「可商用开源第一」，也不是去掉价格帽之后的 Capability 第一。

没有第三份证据能把这些微调模型排进 Deußer、Sparrenberg 与 Sifa 的 37 个公开数据集。那篇评测里，开放侧更强的是未微调的 **Qwen3.8-27B** 下一 token 读出，而且它在 27/37 个数据集上仍低于闭源 Jev 1.13.0。

## 「开源」和「Jev 类」在本文里指什么

Jev 类指接口，不是某一家的权重：输入状态和封闭问题，返回 `choice`、`score` 或 `noul` 的概率，不生成自由文本。通用聊天模型若只用选项代号的下一 token 概率作答，算同一类读出，但是未特化的基座。

开源在本文里分成三档，不混用：

1. **权重可下载。** H2O、StartLux、GLiDE、mammoth-1 都属于这一档。
2. **研究可免费使用，商业或生产另有限制。** H2O 禁止商业和生产。GLiDE 使用 Fastino Community License：年收入超过 1000 万美元的实体若把模型用于生产中的商业用途，需要另签协议。基座 Qwen3.8-27B 在该许可证文本里写明为 Apache-2.0。
3. **Hub 许可证为 MIT 或 Apache-2.0，且不门控。** mammoth-1 的模型卡许可证字段是 MIT。Torchcast Decision 4B 当前 README 写权重为 Apache-2.0，但作者同时写「v1.3 尚未被 JevBench 测过」，而榜单行标成 v1.3。版本对不上，本文不把它写成已被榜单证明的 Apache 第一名。

## 公信力怎么判

一条榜只有同时说清下面几件事，才用来支持「谁最强」：

- 题目和金标从哪来。数据集自带的人工标签，高于两个大模型投票产生的标签。
- 外部能不能重跑。公开划分加上原始响应，高于密封题；密封题高于「只有维护者持有的私有题」。
- 协议是否钉死。模型版本、题面、选项顺序、未作答怎么计分，要能指到一次发布。
- 分数变了，是模型变了还是公式变了。数周内改过头条公式的榜，名次必须带版本。
- 谁出的钱、谁选的题。厂商用自己组装的题宣布自己第一，只作旁证。

按这个标准，本文使用的材料分成四档。

| 档 | 材料 | 用来做什么 |
| --- | --- | --- |
| 高 | Deußer 等，arXiv:2609.37647 | 判断闭源 Jev 相对未微调开放读出的位置。不覆盖 10 月的决策微调 |
| 中 | Benchmark Heaven 的 JevBench | 比较专用决策模型的智力、校准、速度和成本。密封题和单人维护压低了它能支持的结论强度 |
| 中，且只采用公开分 | Jev Decision Index 0.4.1 | 公开分可与 JevBench 对照。Full 分因私有题降到「不能单独封王」 |
| 不用于封王 | TypeSafe 官方 eval、厂商自测、游戏胜率榜、同名的另一套 JevBench | 标签、样本或测量对象与上面三档不是同一件事 |

## JevBench（Benchmark Heaven）

维护者是 Benchmark Heaven，站点写明独立于 TypeSafe，页脚自述为一人维护的爱好项目。评分代码在 [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench)（MIT）。规范页面：<https://benchmarkheaven.com/jev-models>，方法页写的发布是 v1.6.1，现场开放权重榜已经按方法 v1.7 展示。H2O 的模型页写数据版本为 JevBench v1.7.4，发布日 2026-10-25。v1.6.1 的结果校验和 `5cd8c1332226ea9c179883d2285c0d8b8cc593ae38f2645197364a7ddd9c83c1` 对不上现在这张榜，本文不拿它当作 v1.7 的字节级凭证。

一次发布含 1,500 道决策，其中 1,200 道密封、300 道公开。自托管系统答满 1,500 道。密封题和预测不提供下载，方法页写明完整复跑必须走官方评测。成本按冻结的参考价估算，不是账单。自托管延迟的排名用「测得中位数 × 2 + 150 ms」，这是假设，不是端到端测量。方法 v1.7 另加一条：单次决策的测量时间超过 10 秒，该题记错，校准记零。

头条 Capability 是智力与校准的算术平均，只在两道帽之内排名：每千次决策不超过 0.0646 美元，调整后中位延迟不超过 1.23 秒。站点把这道帽叫作 Jev 类资格线，参照是 Jev v1.5。Composite 是智力、校准、速度、成本的加权调和平均，任一轴低于 50 还有平方惩罚；开放权重榜把它标成次要的「official」名次。

2026-10-26 抓到的资格线内前几名：

| 资格线内名次 | 系统 | Capability | 智力 | 校准 | 约每千次成本 | official 名次 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | H2O-Lightning-31B v1.0.7 | 84.3 | 79.3 | 89.4 | $0.044 | 3 |
| 2 | basal-omni-1.0-ultra | 82.1 | 74.1 | 90.2 | $0.045 | 8 |
| 3 | DOPEST 1.0 31B | 82.0 | 73.9 | 90.2 | $0.049 | 27 |
| 4 | Arcbright Elf 1 31B | 81.8 | 71.9 | 91.6 | $0.047 | 12 |
| 5 | Quyet-1.0-Large | 81.7 | 73.4 | 90.0 | $0.045 | 9 |
| 13 | Torchcast Decision 4B（榜上标 v1.3） | 77.5 | 67.1 | 87.8 | $0.015 | **1** |
| 参照，不排名 | Jev 1.13.0 | 77.1 | 63.6 | 90.6 | $0.032 | — |
| 参照，不排名 | Jev 1.14.0 | 75.0 | 59.9 | 90.1 | $0.032 | — |

资格线外、按 Capability 排在页面最前的开放权重：

| 页面顺序 | 系统 | Capability | 智力 | 校准 | 被排除的原因 |
| --- | --- | --- | --- | --- | --- |
| 1 | baselabs mammoth-1 | 85.2 | 76.2 | 94.3 | 成本 3.2 倍于参照 |
| 2 | KnowLine 31B-Gen1 | 85.2 | 77.4 | 93.0 | 成本 2.9 倍于参照 |
| 3 | Fastino GLiDE 1.2 | 84.9 | 76.4 | 93.5 | 成本 6.4 倍于参照 |

H2O 的 84.3 低于这三行。它的第一名是资格线造出来的，不是 Capability 原值的全场第一。

这张榜还有三处必须连着名次一起读：

- **公式在一周内改过第一名。** 模型卡在 2026-10-22 读到的 board v1.7.47 上，Capability 第一仍是 Chon-1.0-12B（85.34），H2O-31B 是开放权重第二。10 秒规则落地之后，第三方记录写 Chon 掉到 official 第 30、Capability 70.1，H2O 升到 84.3。模型没变，计分变了。
- **同一行可以有四个名次。** Torchcast Decision 4B 在资格线内 Capability 第 13，Composite 的 official 名次是第 1。Gautam Parab 在 2026-10-09 已经用更早的发布演示过同一现象：H2O-Lightning-4B 的宣传第一名是次要的 Composite，Capability 头条当时是 Quyet-1.0-Large。
- **作者卡片与榜上版本标签不一致。** Torchcast 的 README 写 v1.3 还没上 JevBench，并给出 v1.2 在 v1.6.34 上的 Composite 73.03。现场榜把一行标成 v1.3，智力 67.1、校准 87.8，对不上那张 v1.2 表。

公信力因此停在「中」：协议、校验和历史和「独立于厂商」是真的；密封集、单人维护、付费快车道、延迟假设和连续改分，使它不能当成已经复现的科学排名。它仍然是唯一把大量决策微调放在同一套 1,500 题上、并同时给出校准和成本的公开榜。

## Decision Index 0.4.1

文件来自 Hugging Face Space [`multimodalart/jev-decision-index`](https://huggingface.co/spaces/multimodalart/jev-decision-index) 的 `data/index.json`，`generated_utc` 为 2026-10-22T20:10:35Z（Asia/Shanghai 为 2026-10-23 04:10），`suite.label` 为 Decision Index 0.4.1，`suite.edition` 为 `release-v4`，`corpus_sha256` 为 `b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5`。228 个模型行加单独存放的 Jev 行，共 229 行。硬件记录为 1× NVIDIA RTX PRO 6000。Jev 参照版本是 jev-1.14.0，文件里的评测版本文字是「Lab run, 2026-10-14」。

公开索引：0.2.1 面板的 37 个基准、五个领域；每个基准先做 `(score − chance) / (1 − chance)` 并截断到 0–1；未作答计错。文件记录 110,201 次请求、109,759 次可计分。文件自述公开套件仍会在模型旁边展示，带私有对应题的基准另外给出同能力的私有分。

头条 `balanced_skill` 的公式原文是：`0.60*private_same_skills + 0.40*private_new_domains`，两部分都用未微调的库存模型映到公开分的尺度上。`index_note` 写明头条 Full 分只来自私有测试。0.3 版审计（The Stochastic Parrot，2026-10-19，对应 2026-10-16 的 174 行快照）量到的权重还是公开 20%、同技能私有 50%、新领域 30%。0.4.1 把公开分移出了头条。算术可以复核，私有题的内容、题量和「与公开题测的是同一能力」都没有随文件公布。

本报告用该文件排出的 Full 分与公开分：

| Full 名次 | 系统 | Full | 公开分 | 公开名次 | 权重与许可证（本次核对到的） |
| --- | --- | --- | --- | --- | --- |
| 1 | Fastino GLiDE 1.2 | 71.52 | 71.99 | 5 | [`fastino/GLiDE`](https://huggingface.co/fastino/GLiDE)，27.8B，基座 Qwen3.8-27B。Fastino Community License，不门控。ECE 0.0204 |
| 2 | Jeffan-31b-1710 | 69.75 | 73.13 | 2 | [`tattrongvu/Jeffan-31b-1710`](https://huggingface.co/tattrongvu/Jeffan-31b-1710)，基座 Gemma-4-31B-it。许可证本次未打开 |
| 3 | baselabs mammoth-1 | 68.44 | **75.34** | **1** | [`baselabs/mammoth-1`](https://huggingface.co/baselabs/mammoth-1)，MIT，不门控。ECE 0.0408 |
| 6 | StartLux-Decision v1.1 27B | 67.92 | 72.88 | 3 | 权重 CC BY-NC 4.0。0.3.2 审计日它是 Full 与公开的双第一（69.12 / 72.88） |
| 16 | H2O Lightning 31B | 66.58 | 66.69 | — | 非商业许可证。ECE 0.0842 |
| 41 | Jev 1.14 | 60.24 | 60.70 | — | 闭源 API |

0.3.2 的第一名 StartLux，到 0.4.1 的 Full 分落到第 6。和 JevBench 一样，隔一周的「第一」不能脱离版本引用。

公开分第一与 Full 分第一分裂，是这张榜对本文最有用的结果：mammoth-1 在可描述的公开题上领先；GLiDE 在不可复跑的私有题上领先，同时公开分仍是第 5，两条分几乎相同（71.99 与 71.52）。GLiDE 不像是只背住了公开套件。它仍不能替代一次外部复跑。

## 学术评测：Deußer、Sparrenberg、Sifa

论文 *Evaluating and Benchmarking the System One Model Jev*，arXiv:2609.37647。评的是钉死的 `jev-1.13.0`，零样本，37 个公开数据集，每个数据集一张冻结题面，全量评测划分，346,009 次请求。开放对照只有两个：Qwen3.8-27B 与 Gemma-4-E4B，用同一请求上的选项代号下一 token 概率，再归一化。思考关闭，不生成文本。作者放出代码、评测工具和全部原始响应。

和榜单相关的结果：

- Jev 在 27 个数据集上高于 Qwen，9 个低于 Qwen，1 个持平。Jev 的领先里有 11 个数据集的 95% bootstrap 区间不重叠；Qwen 的 9 次领先，区间都与 Jev 重叠。
- Gemma-4-E4B 在全部 37 个数据集上低于 Jev，其中 31 个区间不重叠。
- Choice 概率：Jev 在 22 个数据集、279,925 条答案上的合并 ECE 为 0.028；Qwen 为 0.063；Gemma 为 0.208。
- 单条 Noul：Jev 的 ECE 为 0.052，Qwen 0.113，Gemma 0.123。多标签 Noul 上 Qwen 的 ECE 更好（0.119，Jev 0.168）。
- 量规式 Score 上，Qwen 在 9 项里与 Jev 持平或略高。这是开放读出真正咬住 Jev 的区域。

GLiDE 的许可证文本写明其基座 Qwen3.8-27B 为 Apache-2.0。把「最强开源」理解成「这篇论文里测过的、许可证宽松的开放读出」，答案是 Qwen3.8-27B，不是任何一个 10 月的决策微调。论文没有测 mammoth-1、GLiDE 或 H2O，不能倒推出它们在这 37 个数据集上的名次。

方向也和社区榜相反：在人工或数据集金标上，未微调的 Qwen 整体低于 Jev；在 JevBench 和 Decision Index 上，一批决策微调的开放权重高于 Jev 1.13/1.14。两套题不是同一个总体。社区榜上的领先，不能写成「已经在 Deußer 的 37 个数据集上超过 Jev」。

## 明确不用于封王的材料

**TypeSafe 的 evals.typesafe.ai。** 四个工作流、711 个案例的参照标签，是 GPT-6 Astra 与 Claude Fable 5.1 在高思考设置下的平均。Counterproof 的 Field Note 1 归档了其中 20 个案例的逐题文件：在双方都作答的 19 个案例里，两个参照模型对最终动作有分歧。厂商自己也写过，这种参照会偏向 OpenAI 与 Anthropic 的模型。它测量的是「和两个大模型的平均有多接近」，不是「和人工金标有多接近」。

**微软的 36 基准自测。** 2026-10-09 的文章把 Microsoft-Decision-1 的平均准确率写成 83.5%，Jev 82.3%，并声称比 H2O-Lightning-4B 快 2.5 倍。Gautam Parab 核对后：该模型当时不在 JevBench 数据文件里；1.2 分的差距没有区间；速度比使用的是榜单「测得时间 × 2 + 150 ms」之后的 210 ms，而 H2O-4B 的测得中位数是 29 ms。自选套件上的自测不进入本文的名次。

**OpenJev 作者自己的 10,000 题。** Hugging Face 上 `openjev/openjev` 的模型卡写：托管 Jev 85.4%，OpenJev 84.0%，同一基座微调前 80.4%，Nimble 9B 75.7%。其中 3,078 题在开发中用过。这是有用的自述差距，不是独立榜。

**名为 JevBench 的其他项目。** `jevbench.dev` 用游戏胜负排名，2026-09-25 的页面写「要超越的胜率 84%，第 1 名 TypeSafe Jev 1.13」，当时 4 个模型、24 场已核实的胜利。它测的是棋局结果，不是那 1,500 道封闭决策。Benchmark Heaven 的方法页另外声明：还有一个同名项目测的是蜕变一致性，与准确率/成本榜无关。Datawhale 食谱里的 242 题表是更早、更小的公开切片，置信区间很宽，不代表 v1.7 榜。

**Meng 的校准审计（2026-09-24 预印本）。** 它审计的是 Jev 1.13.0 在两个分类任务上的概率，并和开放分类器比较。题面措辞就能把差距挪动几个点。它支持「概率必须按任务重验」，不支持一张总排名。

## 名次为什么不能合成一个数

| 问题 | 答案 | 不能外推的原因 |
| --- | --- | --- |
| 资格线内、开放权重、JevBench Capability v1.7 | H2O-Lightning-31B，84.3 | 非商业许可证；1,200/1,500 题密封；10 秒规则刚把前一周的第一名换掉 |
| 不设 2 倍成本帽的 JevBench Capability | mammoth-1 与 KnowLine 31B-Gen1，页面均为 85.2 | 成本是参照价，不是发票；KnowLine 许可证未核对 |
| Decision Index 公开分 | mammoth-1，75.34 | 分数是维护者测量；本文复核了排序，没有复核每道题 |
| Decision Index Full 分 | Fastino GLiDE 1.2，71.52 | 头条不含公开分；私有题不可下载 |
| 可商用、且版本与榜单行对得上的 Apache 第一 | 没有闭合的证据 | Torchcast 4B 的卡与榜不一致 |
| 人工/数据集金标上测过的开放读出 | Qwen3.8-27B 强于 Gemma-4-E4B | 只测了两个未微调模型；多数数据集仍低于 Jev |

因此本文的工作结论是：**要一个能下载、许可证允许研究和商用、并在公开题上领先的 Jev 类模型，取 mammoth-1。** 要一个落在 JevBench 官方「Jev 类」价格和延迟帽里的开放权重，取 H2O-Lightning-31B，并接受非商业许可证。要一个外部能用原始响应重算、且基座许可证为 Apache-2.0 的读出，取 Qwen3.8-27B，同时记住它不是决策微调，也没有在那篇论文里胜过 Jev。

## 仍未验证的部分

- 本文没有下载密封题，也没有在本地加载 mammoth-1、GLiDE 或 H2O 的权重。
- JevBench v1.7 现场页没有像 v1.6.1 那样被本文保存成带 SHA-256 的结果文件。Capability 数字来自 2026-10-26 渲染出的榜单页，以及 H2O 模型页上的 v1.7.4 标记。
- KnowLine 31B-Gen1、Jeffan-31b-1710、basal-omni、DOPEST、Quyet 的许可证文件没有逐份打开。
- Decision Index 的公开分可以随套件版本再变。0.3.2 到 0.4.1 已经换过一次第一名。
- 社区榜高于 Jev、学术评测里 Jev 高于未微调 Qwen，这两句话同时为真。把其中一句挪到另一套题上，是本文明确拒绝的推断。

## 来源

1. Benchmark Heaven，JevBench 开放权重榜与方法页，<https://benchmarkheaven.com/jev-models>，<https://benchmarkheaven.com/jev-models/methodology>。现场榜于 2026-10-26 抓取；H2O 模型页写 v1.7.4，发布日 2026-10-25。
2. H2O-Lightning-31B 模型卡，<https://huggingface.co/h2oai/h2o-lightning-31b>。非商业许可证；修订 v1.2.2；作者对 board v1.7.47 的读数日期为 2026-10-22。
3. Jev Decision Index `data/index.json`，Space `multimodalart/jev-decision-index`，生成时间 2026-10-22T20:10:35Z，语料 SHA-256 见上文。
4. Fastino GLiDE 许可证，<https://huggingface.co/fastino/GLiDE/blob/main/LICENSE>。其中写明基座 Qwen3.8-27B 为 Apache-2.0。
5. baselabs/mammoth-1 的 Hub API：`license` 为 `mit`，`gated` 为 false。基座来自 Index 文件中的 `meta.base_model`：`Qwen/Qwen3.5-27B`。
6. Torchcast Decision 4B README，<https://huggingface.co/torchcast-ai/torchcast-decision-4b>。Apache-2.0 与「v1.3 尚未上榜」来自该文件。
7. Deußer, Sparrenberg, Sifa. *Evaluating and Benchmarking the System One Model Jev*. arXiv:2609.37647。
8. Gautam Parab. *Two Decision Models, One Benchmark, Two Different Number Ones*. 2026-10-09. <https://gautamparab.com/writing/two-decision-models-one-benchmark/>。
9. The Stochastic Parrot. *Jev Decision Index: Its Full Score Is 80 Percent Private Tests Run by Its Maintainers*. 2026-10-19，审计对象为 2026-10-16 的 0.3.2 快照。
10. Counterproof, Field Note 1，关于 evals.typesafe.ai 的参照标签。2026 年对该站点 20 个案例文件的计数。
11. StartLux-Decision 仓库说明：代码 Apache-2.0，权重 CC BY-NC 4.0。<https://github.com/StartLuxLabs/Startlux-Decision>。
12. `jevbench.dev`，2026-09-25 的游戏胜率页。与 Benchmark Heaven 的 JevBench 不是同一个基准。
