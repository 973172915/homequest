# data 目录说明

- `sh_data.csv`：原始二手房挂牌数据（只读，不要直接改）。
- 本目录及子目录在沙箱里**不可写**（文件权限限制），所有生成物统一输出到项目根目录的 `artifacts/`：
  - `artifacts/reports/`：数据体检报告、模型评估报告
  - `artifacts/processed/`：清洗后的 parquet
  - `artifacts/models/`：模型文件
  - `artifacts/index/`：向量库
