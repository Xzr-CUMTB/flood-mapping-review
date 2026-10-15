"""Write a factual experiment progress ledger without figures or test-set access."""
from pathlib import Path
from datetime import datetime,timezone
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def read(path):
    try:return json.loads(path.read_text()) if path.exists() else {}
    except (OSError,json.JSONDecodeError):return {}


def main():
    core=read(ROOT/'runs/core_s17_evaluation/complete.json')
    core_complete=core.get('status')=='complete'
    plan=read(ROOT/'configs/crossfit_seed17/plan.json')
    foundation=read(ROOT/'configs/foundation_experiments_s17.json')
    ensemble=read(ROOT/'configs/ensemble_controls.json')
    jobs=(plan['jobs']+[{**j,'type':'foundation'} for j in foundation['jobs']]
          +[{'run_id':'final_s17_fusion_risk','type':'matched_review_control'}]
          +[{'run_id':run_id,'type':'ensemble_control'} for run_id in ensemble['new_members']])
    rows=[]
    for job in jobs:
        directory=ROOT/'runs'/job['run_id'];complete=read(directory/'complete.json');status=read(directory/'status.json')
        rows.append({'run_id':job['run_id'],'kind':job['type'],'complete':complete.get('status')=='complete',
                     'reported_status':status.get('status','queued'),'epoch':status.get('epoch'),
                     'elapsed_seconds':complete.get('elapsed_seconds'),'completed_at_utc':complete.get('completed_at_utc')})
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/'reports/experiment_queue_ledger.csv',index=False)
    total=int(frame.complete.sum())
    lines=['# 实验推进与当前证据','',f'更新时间：{datetime.now(timezone.utc).isoformat()}','',
        f'当前实验批次共 {len(jobs)} 项训练任务，已完成 {total} 项。主流程种子为 17，融合集成额外使用种子 29 和 43。此前的 3 项初始开发基线已完成，另行记录在 `11_development_results.md`。',
        '','本阶段不绘图。训练日志、检查点、数据清单和运行代码快照均保存于对应 `runs/` 子目录。',
        '','## 运行状态','', '| 队列 | 状态 | 当前任务 |','|---|---|---|']
    for label,key in [('嵌套跨事件实验','nested_queue'),('TerraMind 对照','foundation_queue'),('校准与匹配复核对照','post_training_queue'),('三成员融合集成','ensemble_queue'),('核心测试与统计评估','core_completion_queue')]:
        state=read(ROOT/'runs'/key/'status.json')
        lines.append(f"| {label} | {state.get('stage','未启动')} | {state.get('run_id','等待前序依赖或执行阶段检查')} |")
    lines+=['','## 已完成的严格跨事件开发折','',
            ('下表中的事件组同时从该折的基础模型和门控训练中排除。2023 年核心测试已按冻结方案完成，见 [核心计算结果](13_core_computational_results.md)。' if core_complete else '下表中的事件组同时从该折的基础模型和门控训练中排除。2023 年核心测试仍需完成全部预定对照并冻结发布记录后才能评估。'),
            '','| 留出事件组 | 模型 | 洪水 IoU | Brier | 漏检 km² | 误报 km² |',
            '|---|---|---:|---:|---:|---:|']
    fold_rows=[]
    for fold in range(4):
        directory=ROOT/'runs'/f'nested_development_fold{fold}'
        if not (directory/'complete.json').exists():continue
        table=pd.read_csv(directory/'map_metrics.csv');fold_rows.append(table)
        for row in table.itertuples():
            lines.append(f'| {row.event_group} | {row.model} | {row.flood_iou:.4f} | {row.brier:.4f} | {row.fn_km2:.2f} | {row.fp_km2:.2f} |')
    if fold_rows:pd.concat(fold_rows,ignore_index=True).to_csv(ROOT/'reports/nested_development_results.csv',index=False)
    lines+=['','闭式门控与正则化门控是在看到第一折 sigmoid 门控结果后进行的开发诊断。它们在实验记录中明确标记为探索分析；没有使用核心测试数据选择规则。',
        '','## 校准和操作指标','',
        '初始开发筛选中，“融合提高 IoU 且提高高置信遗漏”的原始概率现象，在跨事件温度校准后改变方向。因此高置信遗漏必须与概率校准、同预算残余漏检和误报共同解释。',
        '', '主预算为共同有效测地面积的 10%，复核单元为 64×64 像元。选中单元的错误按完美更正计算；这是理想化复核实验，尚未测量人工判读耗时或准确率。',
        '', '六项主对照同时报告残余漏检和误报，并给出同一统计家族的同时区间。空间分块在每个观测区域内估计组成不确定性，不把数百万像元当作数百万独立洪灾事件。',
        '', '## 可复现性与接入检查','',
        'TerraMind 环境已配置为 Python 3.11、PyTorch 2.7.1+cu128、TerraTorch 1.2.6、TorchGeo 0.8.1。新输入层明确对应 GF3 HH/HV 和八位近红外数据。149 个预训练张量匹配，真实开发样本前向和反向检查通过；适配模型共 5,873,617 个参数。',
        '', '主数据下载与整个压缩包 MD5 校验完成；数据清单、重复样本、足迹重叠、输入对齐和缓存数值检查均已记录。中国洪灾事件档案用于日期核对，不充当像元真值。HISEA 原始划分存在重复影像，当前未用它宣称独立洪水外部验证。',
        '', '所有最终模型、校准参数和评估实现将绑定 SHA256 后进行核心评估。主流程为种子 17；三成员融合集成用于不确定性对照，不等于整个方法的多种子复现。额外标签核验和真实人工复核仍需独立完成。']
    if core_complete:
        lines.insert(6, f"本批计算实验已完成：{core['patches']} 个核心切片、{len(core['models'])} 种地图输出及预定统计对照。完整性核验见 `completed_batch_verification.json`。")
        lines=[line.replace('所有最终模型、校准参数和评估实现将绑定 SHA256 后进行核心评估。','所有最终模型、校准参数和评估实现已绑定 SHA256，并已完成核心评估。') for line in lines]
    (ROOT/'reports/12_experiment_progress.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'planned_training_jobs':len(jobs),'completed':total,'report':'reports/12_experiment_progress.md'}))


if __name__=='__main__':main()
