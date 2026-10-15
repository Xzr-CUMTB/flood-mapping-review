"""Write numerical results from completed frozen evaluation. No plots or automatic paper claims."""
from pathlib import Path
from datetime import datetime,timezone
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'runs/core_s17_evaluation'
    completion=json.loads((out/'complete.json').read_text())
    if completion['status']!='complete':raise ValueError('Completed core evaluation required')
    maps=pd.read_csv(out/'map_metrics.csv');review=pd.read_csv(out/'review_budget_results.csv')
    intervals=pd.read_csv(out/'review_simultaneous_intervals.csv')
    points=pd.read_csv(out/'matched_operating_point_metrics.csv')
    lines=['# 封存核心测试：计算实验结果','',f'汇总时间：{datetime.now(timezone.utc).isoformat()}','',
        f"固定方案在 {completion['patches']} 个空间无重叠切片上完成评估，分为北京—保定与哈尔滨两个观测区域。",
        '', '模型与阈值在核心测试预测前固定；结果按发布的参考标签统计。主终点为 10% 有效面积复核预算后的剩余漏检，同时报告误报。复核采用选中单元内完美更正的模拟规则。',
        '', '## 地图性能','', '| 区域 | 模型 | IoU | 漏检 km² | 误报 km² | Brier | 高置信遗漏占参考淹没比例 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in maps.itertuples():
        lines.append(f'| {row.region} | {row.model} | {row.flood_iou:.4f} | {row.fn_km2:.2f} | {row.fp_km2:.2f} | {row.brier:.5f} | {row.confident_omission_01:.4f} |')
    lines+=['','## 主复核对照','',
        '下表为“前者减后者”的面积差；负数表示前者剩余错误更少。区间在每个区域内同时覆盖六个对照的漏检和误报共十二个估计，采用 10 km 配对空间分块。它们描述区域内空间组成不确定性。',
        '', '| 区域 | 前者 | 后者 | 终点 | 差值 km² | 同时 95% 区间 km² |','|---|---|---|---|---:|---:|']
    for row in intervals.itertuples():
        lines.append(f'| {row.region} | {row.first_model}/{row.first_policy} | {row.second_model}/{row.second_policy} | {row.endpoint} | {row.difference_first_minus_second:.2f} | [{row.simultaneous_lower95:.2f}, {row.simultaneous_upper95:.2f}] |')
    lines+=['','## 开发期确定的工作阈值','',
        '阈值在开发事件 OOF 数据中匹配融合模型的误报率或召回率，随后直接用于核心测试。下表保留迁移后的实际误报率与召回率。',
        '', '| 区域 | 开发期匹配目标 | 模型 | 阈值 | 测试误报率 | 测试召回率 | 漏检 km² | 误报 km² |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for row in points.itertuples():
        lines.append(f'| {row.region} | {row.scenario} | {row.model} | {row.threshold:.5f} | {row.false_positive_rate:.5f} | {row.recall:.4f} | {row.fn_km2:.2f} | {row.fp_km2:.2f} |')
    lines+=['','## 数据文件与证据范围','',
        '- `runs/core_s17_evaluation/review_budget_results.csv`：全部模型、排序策略、1/5/10/20% 预算、FN/FP 联合取舍。',
        '- `map_paired_spatial_intervals.csv`：地图指标与原始/校准后高置信遗漏的配对空间区间。',
        '- `router_value_unit_diagnostics.csv`：每个有效复核单元的预测价值、观测 Brier 差和门控摘要。',
        '- `configs/core_release_s17.json`：数据、配置、权重与评估代码 SHA256。',
        '', '主流程使用固定种子 17；三成员融合模型使用 17/29/43，用作集成不确定性对照。完整方法的多种子重复、独立标签复核和真实人工复核实验尚未包含在本批计算结果中。',
        '', '本阶段没有生成论文图。研究结论须同时结合各开发事件、两个核心区域、概率校准与误报代价；任何方法优势以完整配对结果为依据。']
    (ROOT/'reports/13_core_computational_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'reports/core_result_summary.json').write_text(json.dumps({'status':'complete','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'core_completion':completion,'map_rows':len(maps),'budget_rows':len(review),'simultaneous_rows':len(intervals),
        'report':'reports/13_core_computational_results.md','submission_ready_assessed':False},indent=2))
    print(json.dumps({'report':'reports/13_core_computational_results.md','map_rows':len(maps),'budget_rows':len(review)}))


if __name__=='__main__':main()
