# ProxyFactor-v0 全 A 股本地代理因子集设计

## 1. 目标与边界

ProxyFactor-v0 用本地 Qlib 中国日频 K 线数据生成一套可审计、严格因果的代理因子集，用于在接入线上 25 万因子前验证 FactorPanel Encoder 的训练链路与表征学习能力。

数据范围固定为全 A 股、2008-01-01 至 2025-12-31、日频。股票是否有效以 Qlib `all.txt` 中的上市区间为准，不使用今日成分股回填历史。

本版只使用 OHLCV、VWAP 和 Amount 及其单股历史变换，不引入行业、市值、财务、新闻或当日全市场排名。横截面 Rank-Gaussian 和时序 robust-z 由模型输入层统一完成，不在数据集中重复生成。

## 2. 输入数据与时间语义

原始数据位于 `resources/data/qlib/cn_data`，日历实际覆盖 2000-01-04 至 2026-07-13，因此可以为 2008 年因子提供最长 120 个交易日的预热历史，也可以为 2025 年末生成 20 日未来收益标签。

价格字段遵循当前 Qlib 数据集的复权口径。所有因子公式只能使用当日和过去数据；`Ref` 的正数滞后只出现在因子公式中，负数 `Ref` 只允许出现在独立的收益标签公式中。

每个年度分区计算时额外读取前 120 个交易日作为预热区，写出前裁掉预热行。这保证年初滚动因子与一次性全时段计算一致。

## 3. 128 个代理因子

因子数量严格为 128：16 个单日基础因子，加上 14 个滚动因子族在 8 个窗口上的 112 个实例。

### 3.1 单日基础因子

| 名称 | Qlib 表达式 | 含义 |
|---|---|---|
| `pf_kmid` | `($close-$open)/($open+1e-12)` | 日内实体收益 |
| `pf_klen` | `($high-$low)/($open+1e-12)` | 当日振幅 |
| `pf_kmid2` | `($close-$open)/($high-$low+1e-12)` | 实体占振幅比例 |
| `pf_kup` | `($high-Greater($open,$close))/($open+1e-12)` | 上影线相对开盘 |
| `pf_kup2` | `($high-Greater($open,$close))/($high-$low+1e-12)` | 上影线占振幅比例 |
| `pf_klow` | `(Less($open,$close)-$low)/($open+1e-12)` | 下影线相对开盘 |
| `pf_klow2` | `(Less($open,$close)-$low)/($high-$low+1e-12)` | 下影线占振幅比例 |
| `pf_ksft` | `(2*$close-$high-$low)/($open+1e-12)` | 收盘偏移 |
| `pf_ksft2` | `(2*$close-$high-$low)/($high-$low+1e-12)` | 收盘在当日区间的归一化偏移 |
| `pf_open_close` | `$open/($close+1e-12)-1` | 开盘相对收盘 |
| `pf_high_close` | `$high/($close+1e-12)-1` | 最高价相对收盘 |
| `pf_low_close` | `$low/($close+1e-12)-1` | 最低价相对收盘 |
| `pf_vwap_close` | `$vwap/($close+1e-12)-1` | VWAP 相对收盘 |
| `pf_return_1` | `$close/(Ref($close,1)+1e-12)-1` | 1 日收益 |
| `pf_volume_change_1` | `$volume/(Ref($volume,1)+1e-12)-1` | 1 日成交量变化 |
| `pf_amount_change_1` | `$amount/(Ref($amount,1)+1e-12)-1` | 1 日成交额变化 |

### 3.2 滚动因子族

八个统一窗口是 `2, 3, 5, 10, 20, 30, 60, 120` 个交易日。下表每行产生 8 个因子，名称为 `pf_<family>_<window>`。

| 因子族 | Qlib 表达式模板 | 含义 |
|---|---|---|
| `roc` | `$close/(Ref($close,W)+1e-12)-1` | W 日动量 |
| `ma` | `$close/(Mean($close,W)+1e-12)-1` | 均线偏离 |
| `std` | `Std($close,W)/($close+1e-12)` | 价格尺度归一化波动 |
| `beta` | `Slope($close,W)/($close+1e-12)` | 归一化趋势斜率 |
| `rsqr` | `Rsquare($close,max(W,3))` | 趋势线性度；回归至少使用 3 个观测点 |
| `max` | `$close/(Max($high,W)+1e-12)-1` | 距离历史最高价 |
| `min` | `$close/(Min($low,W)+1e-12)-1` | 距离历史最低价 |
| `rsv` | `($close-Min($low,W))/(Max($high,W)-Min($low,W)+1e-12)` | 收盘在窗口高低区间的位置 |
| `corr` | `Corr($close,Log($volume+1),W)` | 价格与对数成交量相关 |
| `cord` | `Corr($close/(Ref($close,1)+1e-12)-1,Log($volume/(Ref($volume,1)+1e-12)+1),W)` | 收益与成交量变化相关 |
| `cntd` | `Mean($close>Ref($close,1),W)-Mean($close<Ref($close,1),W)` | 上涨与下跌天数差 |
| `sumd` | `(Sum(Greater($close-Ref($close,1),0),W)-Sum(Greater(Ref($close,1)-$close,0),W))/(Sum(Abs($close-Ref($close,1)),W)+1e-12)` | 上涨与下跌幅度差 |
| `vma` | `$volume/(Mean($volume,W)+1e-12)-1` | 成交量均线偏离 |
| `vstd` | `Std($volume,W)/(Mean($volume,W)+1e-12)` | 成交量变异程度 |

`pf_rsqr_2` 保留名义窗口 2 和既定名称，但实际表达式固定为 `Rsquare($close,3)`。原因是两点线性回归的 R² 恒为 1，无法作为有效训练特征；该最小观测数规则同时用于 Qlib 生成和独立 pandas 审计。

## 4. 标签、输出与存储

收益标签与因子物理分离，只包含 `ret_1d`、`ret_5d`、`ret_20d` 及各自的有效性 mask。标签按当日收盘到未来收盘的对数收益计算，不参与因子公式计算。

输出根目录是 `resources/data/proxy_factor_v0`：

```text
proxy_factor_v0/
├── factors/year=2008/part.parquet
├── ...
├── factors/year=2025/part.parquet
├── labels/year=2008/part.parquet
├── ...
├── labels/year=2025/part.parquet
├── manifest.json
└── quality_report.json
```

因子分区列为 `date, asset, pf_*`，共 130 列；日期和股票排序稳定，`(date, asset)` 在分区内唯一。数值使用 `float32`，非有限值统一转为 null，Parquet 使用 Zstandard 压缩。

`manifest.json` 记录数据源、Qlib 版本、时间范围、股票池、128 个名称与公式、窗口、标签口径、分区行数、文件大小和 SHA-256。`quality_report.json` 记录每个因子分年与全局的有效率、均值、标准差、极值和近常数比例。

## 5. 生成流程与恢复语义

生成器先从因子注册表构建公式，并在执行前断言名称唯一、数量为 128、因子公式不包含负 `Ref`。然后按年查询 Qlib `all` 股票池，计算预热数据，裁剪到目标年，完成数据清理、质量统计和原子写入。

每个分区先写入同目录临时文件，验证成功后通过 `os.replace` 发布。已完成分区只有在文件 SHA-256、公式集指纹和生成配置一致时才能跳过；任一项不一致就拒绝混用，要求显式重建该分区。

生成顺序为：先生成 2008 年小规模验收分区，通过公式、因果性、schema 和质量门槛后，再从 2008 至 2025 逐年扩展。最终成功条件不是小样本通过，而是 18 个年度的因子和标签分区全部存在并经完整验收。

## 6. 质量门槛

生成器必须通过以下检查：

- 因子注册表恰好包含 128 个唯一名称，无非因果公式。
- 各年分区 schema 一致，因子列顺序与 manifest 一致。
- 日期范围严格落在目标年，股票行落在对应上市区间。
- `(date, asset)` 无重复，排序稳定，数值中无正负无穷。
- 对测试原始 K 线的未来部分施加扰动，不得改变当前及过去因子值。
- 相邻年份边界因子值与一次性跨年计算结果一致。
- 任一因子全时段有效率低于 5% 或近常数比例高于 99% 时，完整生成任务失败，不发布最终 manifest。
- 随机抽取至少 3 个年份和 10 个因子，使用独立 pandas 计算与 Qlib 结果比对。
- 因子、标签、manifest 和质量报告的 SHA-256 全部可重复计算。

## 7. 与模型的契约

每个 `pf_*` 列代表一个独立因子样本族，不是模型输入 channel。训练时从一个因子列取出日期×股票矩阵，构造 `values[B,T,N]`。

全 A 股每日股票数可能超过 512，数据生成阶段保留全部有效股票；训练数据加载器再根据决策日和有效性 mask 采样最多 512 只股票。这使本地数据与未来线上全股票因子接口保持一致，同时不在底层数据中丢失股票。

## 8. 非目标

本任务不启动 Small 模型的完整训练，不实现行业或市值中性化，不将代理因子结果表述为可投资信号，也不用这套本地数据代替线上 25 万因子的最终预训练。
