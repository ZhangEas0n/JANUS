# JANUS

Universal adversarial perturbations for speaker verification (SV) and sensitive-intent recognition (SR).

JANUS 研究音频扰动对身份验证与语义安全边界的联合影响。本仓库包含通用扰动生成、FSC 联合评估、固定时间偏移评估、环境声拟合，以及 DistilHuBERT 物理音频评估代码。

这是持续开发中的研究代码。主实验采用 Hydra 配置入口；DistilHuBERT 实验使用独立的 argparse 脚本。两条实验路径的模型文件、任务头和阈值不能直接混用。文中的命令是按当前代码接口整理的运行示例，不代表已在新服务器完整复现。

## 1. 目录结构

| 路径 | 用途 |
| --- | --- |
| `btuap/` | JANUS 核心损失、扰动操作、SV/SR 工具、Mel/STFT/waveform 损失与 DistilHuBERT 任务头 |
| `src/` | 数据加载、预处理、模型、训练框架与评估工具 |
| `config/` | 实际使用的 Hydra 配置 |
| `scripts/data/` | FSC 标签、注册模板和联合评估协议生成 |
| `scripts/eval/` | 联合评估、Any-10、基线和阈值敏感性评估 |
| `scripts/utilities/` | 绘图、结果汇总及 checkpoint 工具 |
| `experiments/` | 异步注入、物理音频和 DistilHuBERT 专用脚本 |
| `preparation_scripts/` | 数据和预训练资源准备脚本 |
| `tests/` | 已有测试 |
| `docs/` | 路径、目录组织和物理评估说明 |

以下目录由使用者准备或运行时产生，不随 Git 仓库分发：

- `fluent_speech_commands_dataset/`：FSC 数据集。
- `model/`：本地主干模型和 SV checkpoint。
- `models/checkpoints/`：SR 及 DistilHuBERT 任务头。
- `data/`：环境声、注册清单和辅助索引。
- `perturbations/`：扰动张量、音频及生成报告。
- `recordings/`：物理录音。
- `results/`、`logs/`：实验输出及运行日志。

## 2. 环境与安装

### 获取代码

```bash
git clone https://github.com/ZhangEas0n/JANUS.git
cd JANUS
```

所有示例均从仓库根目录执行。建议使用独立环境，并安装音频系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y build-essential ffmpeg sox libsox-dev libsox-fmt-all libsndfile1
```

### 依赖版本说明

2026-10-20 核对旧服务器的 `btuap-env` 时，实际安装版本如下：

| 组件 | 版本 |
| --- | --- |
| Python | 3.8.10 |
| PyTorch | 1.11.0+cu113 |
| torchaudio | 0.11.0+cu113 |
| NumPy | 1.20.3 |
| pytorch-lightning | 1.4.5 |
| transformers | 4.8.2 |
| hydra-core | 1.1.2 |
| fairseq | 0.10.2 |

仓库中的 `environment.yml` 和 `pyproject.toml` 保留了早期依赖配置，其中 PyTorch/torchaudio 固定为 1.9.0/0.9.0，不能当作上述服务器环境的精确锁定文件。迁移已有实验时，优先从对应实验使用的环境导出依赖：

```bash
# 在旧服务器对应的虚拟环境中执行；将文件复制到新服务器。
python -m pip freeze > requirements-server.txt
python -m pip check
```

新服务器需要按 GPU 和驱动选择兼容的 PyTorch/CUDA 构建，并核对其余依赖。DistilHuBERT 脚本与旧 Hydra 代码共存，是否能使用同一个环境需单独验证；不要直接运行依赖整体升级来替代版本核对。

### Hydra 环境变量

在仓库根目录创建本地 `.env` 文件（不会上传到 Git）：

```dotenv
DATA_FOLDER=/absolute/path/to/JANUS
TEMP_FOLDER=/absolute/path/to/JANUS/tmp
LOG_FOLDER=/absolute/path/to/JANUS/logs
USE_COMET_ML=False
NUM_GPUS=1
```

将示例绝对路径替换为新服务器路径。无 GPU 检查时可设 `NUM_GPUS=0`。

```bash
mkdir -p tmp logs results perturbations
python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__); print(torch.cuda.is_available())"
python -c "import src.main, btuap.sr, btuap.sv"
python train_fsc_joint_uap_experiment.py --cfg job
```

最后一个命令用于检查 Hydra 配置组合，不启动训练。它不验证所有数据、权重及 GPU 运算是否可用。

## 3. 数据、模型和实验文件

克隆代码后，还需单独复制旧服务器上的数据和模型。建议保持以下相对位置：

```text
JANUS/
├── fluent_speech_commands_dataset/
│   ├── data/
│   │   ├── train_data.csv
│   │   ├── valid_data.csv
│   │   └── test_data.csv
│   └── wavs/
├── model/
│   ├── wav2vec2-base-960h/
│   ├── wav2vec_base.ckpt
│   └── distilhubert/
├── models/checkpoints/
│   ├── sr/intent_head_fsc.pt
│   └── distilhubert_fsc/
│       ├── distilhubert_fsc_heads.pt
│       └── thresholds.json
├── data/
│   └── environmental_sounds/
├── perturbations/
├── recordings/
└── results/
```

主干目录必须包含相应加载器需要的完整配置、权重，以及适用时的 tokenizer 文件。多个入口启用 Hugging Face 离线模式，DistilHuBERT 加载器使用 `local_files_only=True`，因此模型需要提前准备。

复现旧实验时，同时保留注册清单、数据划分、评估协议、阈值文件、扰动以及实际运行参数。路径变更后，CSV 内的绝对音频路径也需要核对。

## 4. FSC 数据准备

如果已经迁移了旧实验的数据和协议，优先使用原文件。新建协议时可执行：

```bash
python scripts/data/build_fsc_sr_labels.py \
  --fsc_root "$PWD/fluent_speech_commands_dataset"

python scripts/data/build_fsc_joint_protocol.py \
  --fsc_root "$PWD/fluent_speech_commands_dataset" \
  --enrollment_k 3 \
  --enrollment_seed 0
```

标签脚本的默认敏感意图集合以脚本定义为准；生成前应核对目标数据文件和已有备份。协议默认输出到 FSC 根目录下的 `data_processed/fsc_joint_protocol/`。随机目标配对协议与 Any-10 协议是不同评估方式，应分别记录。

## 5. 扰动生成入口

| 入口 | 用途 |
| --- | --- |
| `uap.py` | 原始联合风险扰动入口 |
| `uap_fsc_joint.py` | FSC 联合实验入口 |
| `train_fsc_joint_uap_experiment.py` | FSC 环境声拟合、载体与残差相关实验 |
| `experiments/train_distilhubert_janus_dh.py` | 基于已有初始扰动在 DistilHuBERT 上优化 |
| `experiments/train_distilhubert_fsc_heads.py` | DistilHuBERT 的 FSC 下游任务头训练 |

公共默认值位于 `btuap/config.py`。具体入口可能覆盖这些值；默认值不等于论文中每个实验的实际参数。

例如，环境声训练入口通过 `+fsc_joint_train.<参数>` 指定路径和设置：

```bash
python train_fsc_joint_uap_experiment.py \
  +fsc_joint_train.fsc_root="$PWD/fluent_speech_commands_dataset" \
  +fsc_joint_train.backbone="$PWD/model/wav2vec2-base-960h" \
  +fsc_joint_train.sv_ckpt="$PWD/model/wav2vec_base.ckpt" \
  +fsc_joint_train.intent_head_path="$PWD/models/checkpoints/sr/intent_head_fsc.pt" \
  +fsc_joint_train.env_sound_path="$PWD/data/environmental_sounds/bird.wav" \
  +fsc_joint_train.audio_length=3.0 \
  +fsc_joint_train.output_path="$PWD/perturbations/mel_mimic/bird/janus_example.pt" \
  +fsc_joint_train.train_log_path="$PWD/results/janus_example_train_log.json"
```

这是接口示例；鸟鸣音频由使用者提供，其他损失权重、载体开关及约束应按目标实验显式设置。使用新的输出文件名，以保留旧结果。

DistilHuBERT 优化脚本另需初始扰动和注册清单。查看完整参数：

```bash
python experiments/train_distilhubert_janus_dh.py --help
python experiments/train_distilhubert_fsc_heads.py --help
```

其中优化脚本固定从项目相对路径读取 DistilHuBERT 主干、任务头及阈值；注册清单支持 `enrollment_path`、`audio_path` 或 `path` 列，建议使用新服务器上的绝对音频路径。

## 6. 数字域联合评估

### 固定配对评估

```bash
python scripts/eval/evaluate_fsc_joint.py \
  +fsc_joint_eval.uap_path="$PWD/perturbations/original/janus.pt" \
  +fsc_joint_eval.output_dir="$PWD/results/joint/evaluation_run"
```

该入口需要第 4 节生成或迁移的协议，以及第 3 节的模型和任务头。其他路径可通过 `+fsc_joint_eval.<参数>` 覆盖。

### Any-10 评估

Any-10 从验证集建立固定注册用户集合，测试语音匹配任意一个注册用户即视为身份侧接受。

```bash
python scripts/eval/evaluate_fsc_any10.py \
  +fsc_any10_eval.uap_path="$PWD/perturbations/original/janus.pt" \
  +fsc_any10_eval.sv_threshold_path="$PWD/results/joint/evaluation_run/fsc_joint_thresholds.json" \
  +fsc_any10_eval.target_snr_db=10 \
  +fsc_any10_eval.async_offset_seconds=0 \
  +fsc_any10_eval.output_dir="$PWD/results/main/any10_10db_offset0"
```

模型和数据使用默认位置时可省略对应路径；否则使用 `fsc_root`、`backbone`、`sv_ckpt`、`intent_head_path` 覆盖。阈值文件缺失时部分入口会使用回退值，因此复现时应显式提供已保存的阈值文件或阈值参数，并核对日志。

固定偏移实验可分别设置 `async_offset_seconds=0, 0.25, 0.5, 1.0, 2.0`，每次使用独立输出目录。这是代码中的扰动时间平移评估，不能直接等同于真实设备延迟实验。参考 SNR 与 `epsilon` 在 Any-10 入口中互斥，不能同时指定。

## 7. DistilHuBERT 物理音频评估

该流程读取已有录音文件，不会在服务器上直接调用本地电脑麦克风。

准备录音清单 `metadata.csv`：

```csv
audio_path,clean_audio_path
/absolute/path/recorded_001.wav,/absolute/path/clean_001.wav
```

其中 `audio_path` 是待评估物理录音，`clean_audio_path` 是用于判定进入条件的基准音频。使用原始数字语音或干净物理录音作为基准会形成不同协议，报告时必须注明。

准备注册清单 `enrollment.csv`：

```csv
speaker_id,enrollment_path
speaker_01,/absolute/path/enroll_01.wav
speaker_01,/absolute/path/enroll_02.wav
speaker_02,/absolute/path/enroll_03.wav
```

同一身份的多条注册音频会合成为注册原型。执行：

```bash
mkdir -p results/physical/example
python experiments/eval_physical_audio_distilhubert.py \
  --project-root "$PWD" \
  --metadata /absolute/path/metadata.csv \
  --enrollment-list /absolute/path/enrollment.csv \
  --output "$PWD/results/physical/example/trials.csv" \
  --summary-output "$PWD/results/physical/example/summary.json" \
  --log "$PWD/results/physical/example/evaluation.log"
```

默认加载 `model/distilhubert/`、`models/checkpoints/distilhubert_fsc/distilhubert_fsc_heads.pt` 和同目录的 `thresholds.json`。也可通过 `--backbone`、`--heads-checkpoint`、`--thresholds` 覆盖。

输入音频由加载器转换为单声道 16 kHz。参考混合 SNR 和 RMS dBFS 描述数字信号，不等于房间实测声压级 dB SPL。不同 SNR 档次使用各自的录音清单及输出目录。

## 8. 指标与进入条件

语义分支使用敏感类概率与阈值比较，即 `p_sensitive >= sr_threshold`，并非仅判断最大概率类别。SV 使用评分与 `sv_threshold` 比较。

| 指标 | 含义 |
| --- | --- |
| ASR-SV | 身份侧攻击成功率 |
| ASR-SR | 语义侧攻击成功率 |
| ASR-Joint | 同一试验中身份侧和语义侧同时成功 |
| ASR-Union | 同一试验中至少一侧成功 |
| FAR / Any-10 FAR | 相应协议下的非目标身份接受率 |
| Sensitive Rate | 被判定为敏感意图的比例 |

分母必须结合入口核对：

- Any-10 的四项 ASR 使用共同进入集合：干净语音未被任意注册身份接受，且敏感概率低于阈值。
- DistilHuBERT 物理评估的 ASR-SV 使用 SV 进入集合，ASR-SR 使用 SR 进入集合；Joint 和 Union 使用两项条件均满足的共同集合。
- 物理评估同时输出无条件成功比例、条件 ASR 和各进入集合数量；没有合格样本时，条件 ASR 为 JSON `null`。

只有三项指标使用相同分母时，才可用 `ASR-Union = ASR-SV + ASR-SR - ASR-Joint`。不要直接将不同分母的结果相加。

阈值以具体实验保存的配置和 JSON 为准，不根据测试结果重新选择阈值来替代原协议。

## 9. 迁移到新服务器

1. 克隆代码并核对提交版本。
2. 从原实验环境导出依赖，记录 Python、GPU、驱动和 PyTorch 版本。
3. 单独复制数据、完整模型、任务头、扰动、注册清单及阈值文件。
4. 创建本地 `.env`，更新 CSV 和配置中的旧服务器绝对路径。
5. 完成导入和配置检查，再进行少量样本评估，核对模型、阈值和样本数量。
6. 确认后运行完整评估，使用新输出目录保存结果。

模型、数据和历史实验结果不包含在 GitHub 仓库中。第三方基线仓库也需按对应评估脚本的依赖另外准备。

## 10. 来源与许可

项目部分基础训练、数据处理和模型代码基于 Nik Vaessen 与 David A. van Leeuwen 的 *Fine-Tuning Wav2Vec2 for Speaker Recognition* 工作。原始版权信息和 MIT 许可保留在 [LICENCE](LICENCE) 中。

引用基础代码对应工作时，可使用：

```bibtex
@inproceedings{vaessen2022w2v2speaker,
  author    = {Nik Vaessen and David A. van Leeuwen},
  title     = {Fine-Tuning Wav2Vec2 for Speaker Recognition},
  booktitle = {ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {7967--7971},
  year      = {2022},
  doi       = {10.1109/ICASSP43922.2022.9746952}
}
```

JANUS 的正式论文引用信息将在确定后补充。
