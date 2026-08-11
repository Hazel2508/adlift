# AdLift 从 Phase 1 到 Phase 2 的执行路线

## 1. 当前项目状态

### 已完成并可复用

- `src/prepare_data.py`
  - 将压缩 CSV 转换为 ZSTD Parquet。
  - 将 `f0–f11` 转换为 DOUBLE，将四个实验字段转换为 UTINYINT。
  - 验证总行数为 13,979,592。
- `notebook/02_experiment_design_and_balance.ipynb`
  - 已执行。
  - 对照组 2,096,937 行，处理组 11,882,655 行。
  - 完成 `f0–f11` 的 SMD 与方差比检查。
  - 所有绝对 SMD 小于 0.05，最大为 `f3` 的 0.048836。
  - 已生成 `reports/figures/covariate_balance_smd.png`。
- `notebook/03_duplicate_audit_and_data_partitioning.ipynb`
  - 已执行。
  - 记录了完全重复行和重复特征组，但没有删除记录。
  - 已生成基于 `f0–f11` 的确定性 60/20/20 train/validation/test 分区。
  - 分区规则错误数为 0，跨分区重复特征组为 0。
- 两个 Parquet 输出已存在：
  - 基础分析数据。
  - 带 `data_partition` 的机器学习数据。

### 需要先补齐的 Phase 1 缺口

`notebook/01_data_quality_audit.ipynb` 目前只有：

- 项目目标说明；
- 文件路径、导入和 DuckDB 连接初始化。

它没有执行输出，也还没有实际完成标题中声明的质量检查。因此不应在项目文档中将它标记为已完成。

### 已通过独立读取确认的基础事实

- 基础 Parquet 包含 13,979,592 行和 16 个字段。
- 特征实际为 `f0–f11`，不是 `f0–f12`。
- `treatment`、`exposure`、`visit`、`conversion` 的取值均为 0/1。
- 不存在 `treatment=0, exposure=1` 的记录。
- 不存在 `conversion=1, visit=0` 的记录。
- 对照组的 exposure rate 为 0；处理组的 exposure rate 约为 3.6037%。

## 2. 推荐执行顺序

## Step 0：确定 Notebook 边界

不要把所有分析继续堆进现有 Phase 1 Notebook。建议保持如下边界：

1. `01_data_quality_audit.ipynb`：数据完整性与逻辑质量。
2. `02_experiment_design_and_balance.ipynb`：随机分配与处理前均衡。
3. `03_duplicate_audit_and_data_partitioning.ipynb`：重复记录与机器学习分区。
4. `04_average_incrementality_ab_test.ipynb`：Phase 2 主分析。
5. `05_exposure_delivery_appendix.ipynb`：可选；如果 Phase 2 主 Notebook 不过长，也可将该附录放在 `04`。

## Step 1：先补齐 Notebook 01

补充以下单元格：

1. **Dataset shape and schema**
   - 验证 13,979,592 行。
   - 验证 16 个原始字段。
   - 明确特征为 `f0–f11`。
2. **Missing-value audit**
   - 每个字段输出 null 数和比例。
3. **Binary-domain audit**
   - 验证四个字段仅有 0/1。
4. **Logical consistency**
   - 检查 `treatment=0, exposure=1`。
   - 检查 `conversion=1, visit=0`。
5. **Basic allocation and rates**
   - 按 treatment 输出样本数、exposure rate、visit rate 和 conversion rate。
6. **Assertions**
   - 对总行数、字段集合、二元取值和逻辑关系设置可执行断言。
7. **Reader-facing conclusion**
   - 不与 Notebook 02 的 SMD 和 Notebook 03 的 duplicate audit 重复展开。
8. **Close connection**
   - 关闭 DuckDB 连接。

完成后从头到尾执行并保存输出。

## Step 2：对 Phase 1 做一次收尾检查

- 确认三个 Notebook 的标题、编号和文件名一致。
- 统一 kernel，当前 `02` 与 `03` 的 kernelspec 显示名不同。
- 确认所有 Notebook 都能从项目根目录或 `notebook/` 目录启动并正确找到数据。
- 将语音讨论中的 `f0–f12` 统一修正为数据中的 `f0–f11`。
- 确认 `02` 的 SMD 图文件仍存在。
- 确认 `03` 的临时文件不存在，最终 partitioned Parquet 存在。
- 保留重复行，不重新作出删除决策。

## Step 3：新建 Phase 2 主 Notebook

建议文件名：

`notebook/04_average_incrementality_ab_test.ipynb`

定位：一份可以从头到尾重跑、同时面向技术读者和业务读者的分析报告。

### 推荐 Notebook 结构

1. `# Phase 2 — A/B Testing and Average Incrementality`
2. `## TL;DR`
   - 最后再填入，不先写结论。
3. `## Context and Analysis Plan`
   - 粘贴精简版 Experiment Charter。
   - 明确主分析使用 treatment，而非 exposure。
4. `## Data and Input Validation`
   - 读取基础 Parquet，而不是只读 train partition。
   - Phase 2 是全样本平均效果分析，不需要 train/validation/test 分割。
   - 可以读取 partitioned Parquet，但必须使用全部分区且不按 `data_partition` 过滤。
5. `## Arm Counts and Outcome Rates`
6. `## Effect Sizes and Incremental Outcomes`
7. `## Analytical Confidence Intervals`
8. `## Bootstrap Confidence Intervals`
9. `## Bootstrap QA`
10. `## Hypothesis Tests and Holm Adjustment`
11. `## Required Visual Evidence`
12. `## Exposure Delivery Appendix`
13. `## Joint Business Interpretation`
14. `## Limitations`
15. `## Final Takeaways`

## Step 4：先完成一次 SQL 聚合，再转到 Python

不要将 1,398 万行数据全部读入 pandas。使用 DuckDB 一次聚合出每个 treatment 的：

- 总样本数；
- exposure 正例数与比例；
- visit 正例数与比例；
- conversion 正例数与比例。

聚合结果只有两行，之后的效果量、置信区间、检验和图表可在 Python 中完成。

建议立即断言：

- treatment 仅包含 0/1；
- 两组样本数均大于 0；
- 每个 outcome 在两组的正例数都大于 0；
- 结果率在 [0,1] 之间。

## Step 5：建立可复用的计算函数

不要对 visit 和 conversion 复制两套公式。建议创建一个函数，输入：

- outcome 名称；
- $N_1,N_0,y_1,y_0$。

输出一行结果：

- treatment rate；
- control rate；
- absolute lift；
- relative lift；
- treatment-arm incremental outcomes；
- treat-all incremental outcomes；
- absolute-lift analytical CI；
- relative-lift analytical CI。

两个 outcome 返回同一张 tidy result table，供后续检验和绘图使用。

## Step 6：完成 10,000 次高效 Bootstrap

- 固定随机种子，并在 Notebook 顶部写成参数。
- 不对千万行原数据反复抽样。
- 根据二分结局直接抽取每组正例数：

$$
Y_1^*\sim Binomial(N_1,\hat p_1),\qquad
Y_0^*\sim Binomial(N_0,\hat p_0)
$$

- 对 visit 和 conversion 分别产生 10,000 次 absolute lift 和 relative lift。
- 使用 2.5% 和 97.5% 分位数得到 percentile CI。
- 检查 control rate 为 0 的极端重抽样，即使在当前数据中几乎不可能发生。

## Step 7：实现 Bootstrap QA

### 稳定性检查

- 前 5,000 次和后 5,000 次分别计算区间。
- 对应端点差异不得超过完整 10,000 次区间宽度的 2%。
- 超过时，将重复次数提高到 50,000。

### 解析与 Bootstrap 一致性

对四个组合分别检查：

- visit absolute lift；
- visit relative lift；
- conversion absolute lift；
- conversion relative lift。

项目 QA 标准：

- 对应端点绝对差不超过解析区间宽度的 10%；
- 两个区间的宽度差不超过 10%。

不通过时依次检查 treatment 标签、分母、公式和抽样程序。这是工程 QA 标准，不是统计定理。

## Step 8：完成假设检验和 Holm 校正

对 visit 和 conversion 分别进行双侧两比例 z 检验：

$$
H_0:p_1-p_0=0,\qquad H_A:p_1-p_0\ne0
$$

输出：

- z statistic；
- raw p-value；
- 在两个 outcome 之间计算的 Holm-adjusted p-value。

可在文字中说明 $z^2$ 等价于对应 2×2 表的 Pearson chi-square statistic，但项目实现以 z 检验为准。

## Step 9：完成 Exposure 描述性附录

1. 计算 treatment 对 exposure rate 的差异。
2. 建立三组表：
   - control；
   - treatment-assigned but unexposed；
   - treatment-assigned and exposed。
3. 每组显示样本数、exposure rate、visit rate 和 conversion rate。
4. 在表格和结论附近明确写：
   - exposed-versus-unexposed 是观察性对比；
   - 它不是看到广告的因果效果。
5. 有时间再做可选的 Complier Effect/Wald ratio。

## Step 10：制作五张规定图

建议输出到 `reports/figures/phase2/`：

1. `arm_rates.png`
   - visit 和 conversion 的 treatment/control rate 与 95% CI。
2. `absolute_lift.png`
   - 百分点差与 CI，包含零参考线。
3. `relative_lift.png`
   - relative lift 与 Bootstrap CI，标注 control baseline。
4. `incremental_outcomes.png`
   - treatment-arm 规模和 treat-all 情景。
5. `experiment_delivery_funnel.png`
   - treatment assignment 到 exposure、visit、conversion 的描述性漏斗。

注意：漏斗不能暗示后续事件全部由 exposure 导致。

## Step 11：生成一张最终结果表

每个 outcome 一行，至少包含：

- control rate；
- treatment rate；
- absolute lift；
- analytical absolute CI；
- bootstrap absolute CI；
- relative lift；
- analytical relative CI；
- bootstrap relative CI；
- treatment-arm incremental outcomes；
- treat-all incremental outcomes；
- z statistic；
- raw p-value；
- Holm-adjusted p-value。

建议将机器可读版另存为：

`reports/tables/phase2_effect_summary.csv`

## Step 12：写最终业务结论

按顺序回答：

1. 广告是否提高 visit？
2. 广告是否提高 conversion？
3. 两个 outcome 的绝对和相对效果是多少？
4. 当前 treatment 规模下代表多少增量结果？
5. conversion 是否因稀少而更不精确？
6. visit lift 是否按相近比例转化为 conversion lift？
7. 因缺少 campaign、时间、成本和收入字段，还不能回答什么？

必须加入两个限定：

- 正的平均效果不代表每个用户都受益，它为 Phase 3 的异质性 uplift modeling 提供了理由。
- 数据已匿名化并子抽样，结论只适用于发布的实验样本，不是某个真实广告主的原始增量。

## Step 13：顶部 TL;DR 最后写

在所有计算和图表执行成功后，再在 Notebook 顶部写 4–6 条结论。每条都应包含已执行的具体数字，而不是预设结论。

## Step 14：从头到尾执行与交付

- 重启 kernel。
- Run All。
- 确认没有隐藏状态和越过顺序的单元格依赖。
- 确认所有结果表和五张图已生成。
- 确认 TL;DR 中的每个数字与实际输出一致。
- 确认 Notebook 中没有过大的原始数据表输出。
- 更新项目 README，说明如何运行 Phase 1 和 Phase 2。

## 3. 优先级与建议完成节奏

### P0：下次课前必须完成

1. 补齐并执行 Notebook 01。
2. 创建 Phase 2 主 Notebook。
3. 完成两个 outcome 的效果量和解析 CI。
4. 完成 10,000 次 Bootstrap 和 QA。
5. 完成 z 检验和 Holm 校正。
6. 完成 exposure 描述性附录。
7. 完成五张图、总表和业务结论。
8. 重启 kernel 后 Run All 验证。

### P1：强烈建议

1. 实现可复用效果计算函数。
2. 导出机器可读结果表。
3. 更新 README。
4. 统一 Notebook kernel 和语言风格。

### P2：加分项

1. Complier Effect / Wald IV ratio。
2. 将 Phase 1 SMD 图引入项目最终报告。
3. 把面试手册中的 30 秒和 2 分钟讲法转化为简历 bullet points。

## 4. 完成定义

Phase 2 只有在以下条件全部满足时才算完成：

- Notebook 能在清空状态下从头到尾执行。
- 两个 outcome 都有完整的效果量、两类 CI 和假设检验。
- Bootstrap 的稳定性和解析一致性检查已记录。
- Holm 校正已记录。
- Exposure 附录没有做错误的因果声称。
- 五张规定图和最终结果表已保存。
- 文字结论回答了 Phase 2 中规定的七个业务问题。
- 结论正确限定为发布的匿名子抽样实验样本。
