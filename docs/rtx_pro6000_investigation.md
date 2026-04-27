# SoulX-LiveAct 在 RTX PRO 6000 / RTX 50 系列上的调研与适配记录

> 目标：持续追加，沉淀“可运行 + 可提速 + 可复现”的工程实践。  
> 最近更新：2026-04-27（FlashAttention/Torch 适配追加）

---

## 1. 背景与问题定义

当前仓库核心优化最初偏向 H100/H200（SM90）场景，但现在需要稳定运行在消费级/新架构 GPU（如 RTX 5090、RTX PRO 6000，compute capability 12.x）上。

已观察到的关键失败点：

- warmup 阶段注意力路径硬编码为 `AttnType.SAGE_FP8_SM90`
- 该 kernel 面向 Hopper（SM90）优化，不兼容 12.x 架构
- 导致 warmup 直接报错，中断推理流程

---

## 2. 代码现状审计（本仓库）

### 2.1 Attention 路径

- `model_liveact/model_memory_sp.py` 中：
  - 多卡/序列并行自注意力通过 `xFuserLongContextAttention` 执行
  - 原实现固定为 `AttnType.SAGE_FP8_SM90`
- `model_liveact/attention.py` 与 `wan/modules/attention.py` 中：
  - 提供 FlashAttention2/3 与 SDPA 路径
  - 但 warmup 失败主因来自前述 xFuser 路径硬编码，而非这些通用 fallback

### 2.2 依赖现状

仓库 `requirements.txt` 关键版本：

- `torch==2.8.0`
- `torchaudio==2.8.0`
- `torchvision==0.23.0`
- `xfuser==0.4.5`
- 另外 README 还要求额外安装：
  - `sageattention==2.2.0`
  - `vllm==0.11.0`

---

## 3. 外部资料调研结论（与本问题强相关）

### 3.1 SageAttention

根据 SageAttention 官方仓库说明：

- `sageattn_qk_int8_pv_fp8_cuda_sm90` 是 Hopper 专用路径
- 同时提供 `sageattn_qk_int8_pv_fp8_cuda`（非 SM90 专属）
- 项目更新记录明确提到对 RTX5090 的编译支持与性能结果
- 对 Blackwell / 新架构建议更高 CUDA 版本（文档提到 Blackwell / 2++ 场景建议 CUDA >= 12.8）

### 3.2 FlashAttention

根据 FlashAttention 官方仓库说明：

- FA3 标注为 Hopper 优化（H100/H800）
- FA4 开始覆盖 Hopper + Blackwell
- 这进一步印证：不能把 SM90 专用路径硬编码成“通用默认”

---

## 4. 本次修复方案（已落地）

文件：`model_liveact/model_memory_sp.py`

### 4.1 修复点

将固定 `AttnType.SAGE_FP8_SM90` 改为“按设备能力自动选择 + 运行时回退”：

- 当设备 `major == 9` 且有对应枚举时，优先 `SAGE_FP8_SM90`
- 否则按顺序尝试：
  - `SAGE_FP8`
  - `SAGE_AUTO`
  - `TORCH_FLASH`
  - `TORCH_EFFICIENT`
  - `TORCH_MATH`
- 若执行时报 SM90/compute capability/not supported 类错误，自动切换到非 SM90 fallback 并重试

### 4.2 预期收益

- 解决 warmup 阶段在 12.x GPU 上的硬失败
- 保留 SM90 上的最快路径
- 让 RTX 5090 / RTX PRO 6000 场景具备“先可用，再优化”的稳定基线

---

## 5. RTX PRO 6000 / RTX 50 系列推荐实践（当前建议）

> 以下建议以“稳定推理优先，性能逐步打开”为原则。

### 5.1 基线稳定配置

1. 先启用仓库已有低显存策略：
   - `--fp8_kv_cache`
   - `--block_offload`
   - `--t5_cpu`
2. attention 使用自动选择后的非 SM90 路径（本次已修复）
3. 若追求稳定优先，先用 BF16 全流程（除 KV cache）

### 5.2 加速策略分层

- 第一层（风险最低）：
  - Torch SDPA / Torch Flash fallback
  - FP8 KV cache
- 第二层（收益更高）：
  - `SAGE_FP8` / `SAGE_AUTO`
  - vLLM FP8 GEMM（仓库已有 `enable_fp8_gemm`）
- 第三层（版本依赖更强）：
  - 更高版本 attention kernel（含 Blackwell 定向优化）
  - 需与 CUDA、驱动、PyTorch 三者一起验证

### 5.3 版本对齐原则

- 不要只升级单个算子库；应做“驱动 + CUDA + PyTorch + attention 库”联合矩阵验证
- 若目标是 Blackwell/12.x 的最佳性能，优先选用文档明确支持该架构的 attention 实现，而非 SM90 专用实现

---

## 6. 验证与回归建议（持续追加）

### 6.1 必做验证

- 功能：warmup 是否通过（无 attention kernel 崩溃）
- 质量：首段视频是否出现明显退化（尤其开启 FP8 KV cache 时）
- 性能：记录至少以下指标
  - warmup 总时长
  - 稳态 FPS
  - 峰值显存

### 6.2 推荐实验矩阵

- 设备维度：
  - RTX PRO 6000（12.x）
  - RTX 5090（12.x）
- attention 维度：
  - `SAGE_FP8`
  - `SAGE_AUTO`
  - `TORCH_FLASH`
  - `TORCH_EFFICIENT`
- 精度维度：
  - BF16 + FP8 KV cache
  - BF16（无 FP8 KV）

---

## 7. 追加日志（Append-only）

### 2026-04-27

- 完成仓库 attention 路径定位，确认 warmup 失败根因是 `SAGE_FP8_SM90` 硬编码
- 增加架构自适配与 runtime fallback 逻辑，避免 12.x 设备走 SM90 专用 kernel
- 新增本调查文档，作为后续持续追加的统一入口

### 2026-04-27（FlashAttention/Torch 追加）

- 追加“最新版本观测”：
  - PyPI: `torch` 最新为 `2.11.0`
  - PyPI: `flash-attn` 最新为 `2.8.3`
- 结合本仓库 `xfuser==0.4.5` / `vllm==0.11.0` 兼容性，新增 `requirements_core_pro6000.txt`，固化当前项目建议核心栈：
  - `torch==2.8.0`
  - `torchaudio==2.8.0`
  - `torchvision==0.23.0`
  - `xfuser==0.4.5`
  - `vllm==0.11.0`
  - `flash-attn==2.8.3`
- 改造 `model_liveact/attention.py` 与 `wan/modules/attention.py`：
  - FA3 不再作为“只要可导入就默认启用”，改为仅在 SM90 设备优先
  - FA3/FA2 运行时失败会自动回退到 Torch SDPA
  - Torch SDPA 采用 backend 轮询（flash/efficient/cudnn/math）提升在新架构上的可用性与稳定性
