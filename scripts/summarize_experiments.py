# -*- coding: utf-8 -*-
"""Build a factual, figure-free experiment report from completed local runs."""
from pathlib import Path
from datetime import datetime, timezone
import csv, json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'sar': 'SAR', 'fusion': '四波段光学＋SAR', 'rgb_fusion': 'RGB＋SAR', 'optical': '光学', 'probability_average': '专家概率平均'}


def main():
    rows = []
    for directory in sorted((ROOT / 'runs').glob('dev_*_seed*')):
        if not (directory / 'complete.json').exists():
            continue
        result = json.loads((directory / 'complete.json').read_text())
        config = json.loads((directory / 'config.json').read_text())
        if result.get('status') != 'complete':
            continue
        row = {'run_id': result['run_id'], 'modality': config['modality'], 'seed': config['seed'],
               'train_samples': config['train_samples'], 'validation_samples': config['val_samples'],
               'train_groups': config['train_groups'], 'validation_groups': config['val_groups'],
               'selected_epoch': result['selected_epoch'], 'elapsed_seconds': result['elapsed_seconds'],
               'peak_gpu_gb': result['peak_allocated_gpu_bytes'] / 1e9,
               'core_test_evaluated': result['core_test_evaluated'], **result['metrics']}
        rows.append(row)
    if not rows:
        print('No completed real-data development runs; no result report generated.')
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / 'reports/development_results.csv', index=False)
    lines = ['# 首轮真实数据开发实验', '',
             '更新时间：' + datetime.now(timezone.utc).isoformat(), '',
             '本报告由完成的运行记录自动汇总。验证集用于选择最低 NLL 检查点，因此这些结果用于开发筛选；2023 核心测试结果仍未读取。', '',
             '| 输入 | 训练/验证切片 | 选中 epoch | 洪水 IoU | 漏检率 | Brier | 高置信遗漏比例 | 峰值显存 GB |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        fmt = lambda value: 'NA' if value is None else f'{value:.4f}'
        lines.append(f"| {NAMES.get(row['modality'], row['modality'])} | {row['train_samples']}/{row['validation_samples']} | "
                     f"{row['selected_epoch']} | {fmt(row['flood_iou'])} | {fmt(row['fn_rate'])} | "
                     f"{fmt(row['brier'])} | {fmt(row['confident_omission_fraction'])} | {row['peak_gpu_gb']:.2f} |")
    lines += ['', '高置信遗漏定义为真值洪水中预测概率不超过 0.1 的测地面积比例；普通漏检阈值为 0.5。面积统计使用相同覆盖范围和无足迹重叠的切片子集。', '']
    mapping = {row['modality']: row for row in rows if row['seed'] == 17}
    if {'sar', 'fusion'} <= mapping.keys():
        s, f = mapping['sar'], mapping['fusion']
        di = f['flood_iou'] - s['flood_iou']
        dc = f['confident_omission_fraction'] - s['confident_omission_fraction']
        observation = '同时出现' if di > 0 and dc > 0 else '未同时出现'
        lines += ['## 预设假说的开发观察', '',
                  f'四波段融合相对 SAR：IoU 差值 {di:+.4f}，高置信遗漏比例差值 {dc:+.4f}。本轮{observation}“IoU 上升且高置信遗漏增加”的组合。该判断直接来自本轮数据，不作为预定结论。', '']
    calibration = ROOT / 'runs/calibration_development/metrics.csv'
    if calibration.exists():
        calibrated = pd.read_csv(calibration)
        overall = calibrated[calibrated.event_group.eq('ALL')]
        lines += ['## 概率刻度的跨事件检验', '',
                  '用另一个验证事件的抽样像元拟合正温度，再应用到当前事件；0.5 的分类边界保持不变。原检查点使用过这两个事件的验证 NLL，因此这里仍属于开发诊断。', '',
                  '| 模型 | 概率处理 | Brier | NLL | 高置信遗漏比例 |', '|---|---|---:|---:|---:|']
        for row in overall.to_dict('records'):
            label = '原始概率' if row['mode']=='raw' else '跨事件温度校准'
            lines.append(f"| {NAMES[row['model']]} | {label} | {row['brier']:.4f} | {row['nll']:.4f} | {row['confident_omission_fraction']:.4f} |")
        subset = overall[overall['mode'].eq('cross_event_temperature')].set_index('model')
        difference = subset.loc['fusion','confident_omission_fraction'] - subset.loc['sar','confident_omission_fraction']
        lines += ['', f'温度校准后的高置信遗漏差值（融合−SAR）为 {difference:+.4f}。概率刻度影响该指标的比较方向；后续以同预算残余遗漏、校准控制和事件配对结果联合判断方法。', '']
    screen = ROOT / 'runs/development_pair/paired_screening.json'
    table = ROOT / 'runs/development_pair/idealized_review_screen.csv'
    if screen.exists() and table.exists():
        screening = json.loads(screen.read_text())
        review = pd.read_csv(table)
        selected = review[review.budget_fraction.eq(.1)]
        lines += ['## 相同面积预算下的复核筛选', '',
                  '以下假设选中单元的全部错误被完美更正，表示理想纠错结果；真实人工耗时与判读准确率尚待测量。主要预算固定为有效面积的 10%。', '',
                  '| 模型 | 排序规则 | 实际面积预算 | 残余漏检 km² |', '|---|---|---:|---:|']
        for row in selected.to_dict('records'):
            lines.append(f"| {NAMES.get(row['model'],row['model'])} | {row['method']} | {row['actual_area_fraction']:.4f} | {row['residual_fn_km2']:.4f} |")
        lines += ['', f"融合后熵下降且 Brier 损失上升的洪水面积比例为 {screening['lower_entropy_higher_brier_flood_area_fraction']:.4f}。该统计描述配对像元变化，不能替代跨事件效果估计。", '']
    lines += ['## 下一阶段', '',
              '按灾害组生成专家 OOF 预测，训练并消融价值门控，再用完整系统 OOF 预测训练漏检风险头；补充光学单源、概率平均及简单风险排序对照。所有设计和阈值冻结后再评估 2023 核心测试。', '',
              '源记录位于 `runs/<run_id>/`：配置、数据清单、历史、检查点、逐切片指标和验证概率均保留。绘图保持暂停。', '']
    (ROOT / 'reports/11_development_results.md').write_text('\n'.join(lines), encoding='utf-8')
    print(frame[['run_id', 'flood_iou', 'fn_rate', 'confident_omission_fraction']].to_string(index=False))


if __name__ == '__main__':
    main()
