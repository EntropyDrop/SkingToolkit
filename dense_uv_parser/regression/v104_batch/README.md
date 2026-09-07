# v104 全量 edited 图片推理

服务器项目：`/home/ds/llms/SkingToolkitDev`。

```bash
cd /home/ds/llms/SkingToolkitDev
/home/ds/miniconda3/envs/sking-v61-worker/bin/python dense_uv_parser/run_all_edited_v104.py
```

默认递归扫描 `/home/ds/llms/SKING_DDJ_Dataset/entropydrop_website_generations/` 下的 `*_edited.png`，固定使用已复核的 v104 第 5000 步 checkpoint 和配套 pipeline，启动两个后台 worker。输出是原目录下的 `*_result_v104.png`（64×64 RGBA）。不会把 source、旧结果或 edited_preview 当作输入。

每次新任务都会建立 `dense_uv_parser/runs/v104_all_edited_<UTC时间>/`，保存代码、模型、配置、输入清单和哈希快照。已有 v104 结果备份在该任务的 `previous_v104_results/`，然后才允许新结果原子替换；source 和其他版本结果不覆盖。之后修改工作区代码不会改变已启动任务。

常用命令：

```bash
# 只扫描，不生成结果
/home/ds/miniconda3/envs/sking-v61-worker/bin/python dense_uv_parser/run_all_edited_v104.py --dry-run

# 查看最近任务的目录和 PID
cat dense_uv_parser/runs/v104_all_edited_latest.json

# JOB 替换为启动时打印的绝对任务目录
/home/ds/miniconda3/envs/sking-v61-worker/bin/python dense_uv_parser/run_all_edited_v104.py --status JOB

# 中断或机器重启后，恢复同一个任务及其固定快照
/home/ds/miniconda3/envs/sking-v61-worker/bin/python dense_uv_parser/run_all_edited_v104.py --resume JOB
```

不带 `--resume` 是新建一次完整重跑，不是恢复最近任务。同一时间只允许一个 v104 全量任务；重复启动会提示已有任务运行，不会抢写输出。

`--workers 1` 至 `--workers 4` 只在创建任务时设置并发，默认 2。`--data-root PATH` 可改输入根目录；`--prepare-only` 只生成快照，稍后用 `--resume JOB` 启动。已准备任务恢复时以 `job.json` 中的原设置为准。

任务文件：

- `progress.json`：实时总数、完成数、失败数及 worker PID。
- `runner.log`、`logs/*.log`：调度和逐批推理日志。
- `results.jsonl`：逐图来源、输出哈希与失败记录，断点恢复时重新核对已完成结果。
- `artifacts/`：逐图完整 UV、渲染图和推理来源清单。
- `final_audit.json`：结束后的输入/旧结果保护、输出格式及来源审计。

批次失败后会把尚未完成的图片分别重试一次。恢复任务时跳过内容校验通过的结果，并能直接导出已经完成但尚未写回的推理产物。最后一条日志若因断电被截断，会保留前面完整记录后重试未完成项。

文件和来源审计不等于逐图语义验收；v104 的可见头部改善、头底面退步等已知局限见 `dense_uv_parser/V104_TRAINING.md`。

验证：`python dense_uv_parser/regression/v104_batch/test_batch.py`，覆盖递归同名输入隔离、其他版本不被选为输入、外部修改结果不被覆盖、空输出拒绝，以及完成产物和截断日志的断点恢复。
