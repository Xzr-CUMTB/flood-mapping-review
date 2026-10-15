# 洪灾制图与有限预算复核：论文复现代码

论文：*Limited review reverses model rankings in satellite flood mapping*。
本候选分支补齐 6 张主图、10 张补充图的代码，以及数值、评估和模型状态核验入口。

数据包包含真实冻结结果、完整图表源数据、11 张 SI 表格、13 份最终模型状态和文件级 SHA-256。
数据文件单独归档，不放进 Git。Harvard Dataverse 按作者最新选择暂缓：尚未创建数据集、尚未上传，也没有 DOI。

复现分三级：CPU 重算论文数值并重绘图；CPU 从逐切片/复核单元统计量重建评价和区间；
使用原始 Sen2GF3Floods V2 与既定 Windows/CUDA 流程重新推理或训练。
具体命令见 [英文首页](README.md)、[完整流程](docs/REPRODUCING.md)、
[绘图说明](docs/FIGURE_REPRODUCTION.md) 和 [实际核验](docs/VALIDATION.md)。

原实验与冻结文件保持不变。公开表 S7 的两处末位舍入已校正，同时保存原表转录和差异报告；
原稿尚需同步，详见 [论文数值](docs/PAPER_RESULTS.md)。
代码采用 Apache-2.0，研究生成数据采用 CC BY 4.0；第三方内容保留原许可和署名。
PR 供作者审阅，未合并主分支；数据正式发布另行操作。
