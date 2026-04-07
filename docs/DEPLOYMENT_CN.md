# 3D-MOOD Conda 部署与 Demo 验证（含踩坑）

本文记录在 Linux + Conda 下从零安装 3D-MOOD、运行 `scripts/demo.py` 的流程，以及常见网络与依赖问题。

## 环境要求（与官方 README 一致）

- Python **3.11+**
- PyTorch **2.4+**（需与 **NVIDIA 驱动** 匹配，见下文「驱动与 CUDA 版本」）
- NVIDIA GPU + 正确驱动（CPU 可跑 Demo，但 Swin-B 会非常慢）

## 推荐：Conda 环境

```bash
conda create -n 3dmood python=3.11 -y
conda activate 3dmood
```

环境名可自定；官方示例使用 `opendet3d`。

## 安装步骤

### 1. 安装 Vis4D

```bash
pip install vis4d==1.0.0
```

`vis4d` 会拉取较新的 PyTorch（例如带 **cu130** 的 2.11）。若你的机器 **驱动较旧**，可能出现 `torch.cuda.is_available()` 为 `False`（见下文处理方式）。

### 2. 安装 `vis4d_cuda_ops`（避免 `pip install git+...` 超时）

官方命令：

```bash
pip install git+https://github.com/SysCV/vis4d_cuda_ops.git --no-build-isolation --no-cache-dir
```

在国内或弱网环境下，`git+https` 克隆容易 **超时或中断**。可改用 **浅克隆 + 本地安装**：

```bash
export GIT_HTTP_LOW_SPEED_LIMIT=1000
export GIT_HTTP_LOW_SPEED_TIME=600

git clone --depth 1 https://github.com/SysCV/vis4d_cuda_ops.git /tmp/vis4d_cuda_ops
pip install /tmp/vis4d_cuda_ops --no-build-isolation --no-cache-dir
```

- 若 GitHub 仍不可用，可换镜像站或代理后再执行 `git clone`。
- 编译需要本机已配置 **CUDA Toolkit** 及与当前 PyTorch 匹配的 **nvcc**；失败时根据日志检查 `CUDA_HOME`、`gcc` 版本等。

### 3. 安装本仓库（3D-MOOD / `opendet3d`）

在仓库根目录：

```bash
pip install -v -e .
```

本仓库的 `requirements.txt` 已将 **`transformers` 限制为 4.x**（`<5`）。若未使用该约束而安装了 **Transformers 5.x**，运行 Demo 时可能在 BERT 分词处报错：

```text
AttributeError: BertTokenizer has no attribute batch_encode_plus
```

处理方式：

```bash
pip install 'transformers>=4.36.0,<5'
```

### 4. 运行 Demo

在仓库根目录：

```bash
python scripts/demo.py
```

成功时会在 `assets/demo/output.png` 生成可视化结果。

## Hugging Face 与权重下载

Demo 会：

1. 从 Hugging Face 加载 **`bert-base-uncased`**（经 `transformers`）。
2. 通过 **PyTorch `torch.hub` / HTTP** 从官方 URL 下载 Swin-B 权重（约 1GB），缓存目录一般为：
   `~/.cache/torch/hub/checkpoints/gdino3d_swin-b_120e_omni3d_834c97.pt`

### 无法访问 `huggingface.co` 时

- **仅 BERT 等 Hub 资源**：可设置镜像（示例为社区镜像，请自行评估可用性与合规性）：

  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  python scripts/demo.py
  ```

- **权重 URL 仍指向 `huggingface.co`**：`load_model_checkpoint` 使用 `torch` 的 HTTP 下载，**不会**自动走 `HF_ENDPOINT`。可 **手动下载** 到上述缓存文件名（镜像示例）：

  ```bash
  mkdir -p ~/.cache/torch/hub/checkpoints
  curl -L -o ~/.cache/torch/hub/checkpoints/gdino3d_swin-b_120e_omni3d_834c97.pt \
    "https://hf-mirror.com/RoyYang0714/3D-MOOD/resolve/main/gdino3d_swin-b_120e_omni3d_834c97.pt"
  ```

  然后再运行 `python scripts/demo.py`，将直接使用缓存，不再请求外网。

## 驱动与 PyTorch CUDA 版本（常见告警）

若日志中出现类似：

```text
CUDA initialization: The NVIDIA driver on your system is too old ...
torch 2.11.0+cu130
torch.cuda.is_available() -> False
```

说明当前安装的 PyTorch 为 **CUDA 13.0 构建**，而本机 **驱动过旧**，无法初始化 GPU。可选方案：

1. **升级 NVIDIA 驱动** 到官方文档中与所用 PyTorch CUDA 版本匹配的版本。
2. **改装与驱动匹配的 PyTorch**（示例：驱动对应 CUDA 12.4 时，可选用 cu124 构建的 2.4.x；版本号请对照 [PyTorch 官网](https://pytorch.org/get-started/locally/)）：

   ```bash
   pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
   ```

   **注意**：更换 PyTorch 大版本后，必须 **在具备 nvcc 的前提下重新编译** `vis4d_cuda_ops`（见下一节「GPU 与 `vis4d_cuda_ops` 真·CUDA 编译」）。

在未修复 GPU 前，Demo 会退化为 **CPU 推理**，耗时会明显增加。

## GPU 上跑 Demo（PyTorch cu124 + 重编译 `vis4d_cuda_ops`）

在 **驱动可用**（`nvidia-smi` 正常）且 **`torch.cuda.is_available()` 为 True** 的前提下，若前向仍报错：

```text
RuntimeError: Not compiled with GPU support
```

说明当前安装的 `vis4d_cuda_ops` 是按 **CPU 路径**编出来的（构建时 `CUDA_HOME` 为空、或本机没有 `nvcc`、或当时 `torch.cuda.is_available()` 为 False）。需要 **带 CUDA 的 toolkit + 正确的环境变量** 后 **强制重装** 扩展。

### 1. 用 Conda 装 CUDA 编译链（示例：与 cu124 PyTorch 对齐）

```bash
conda install -y cuda-nvcc=12.4 cuda-toolkit=12.4 -c nvidia
```

### 2. 安装与驱动匹配的 PyTorch（示例）

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
```

### 3. 准备 `CUDA_HOME` 与头文件路径（Conda 常见坑）

NVIDIA Conda 包把头文件放在 `$CONDA_PREFIX/targets/x86_64-linux/include`，而 PyTorch 扩展编译默认加 `-I$CUDA_HOME/include`。若直接 `export CUDA_HOME=$CONDA_PREFIX` 后出现 **`cuda_runtime.h` 找不到**，可将该目录下的头文件 **符号链接** 到 `$CONDA_PREFIX/include`（或只链接缺失项）。本次环境还曾缺 **`nv/`、`thrust/`、`cub/`** 子树，可从同一 `targets/.../include` 链入：

```bash
T="$CONDA_PREFIX/targets/x86_64-linux/include"
D="$CONDA_PREFIX/include"
ln -sfn "$T/nv" "$D/nv"
ln -sfn "$T/thrust" "$D/thrust"
ln -sfn "$T/cub" "$D/cub"
# 若仍报缺 .h/.hpp，可将 $T 下其余文件同步链到 $D（注意与 Conda 已有头文件重名冲突）
```

### 4. 重编译并验证

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
# 可选：限制并行，避免编译占满机器
export MAX_JOBS=8

python -c "import torch; assert torch.cuda.is_available()"

pip install /tmp/vis4d_cuda_ops --no-build-isolation --no-cache-dir --force-reinstall
```

成功时生成的 wheel 体积会 **明显大于** 仅 CPU 构建（例如数百 KB 量级 vs 更小体积的“假 GPU”包）。然后：

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 若需镜像访问 Hub
python scripts/demo.py
```

`torch.cuda.is_available()` 为 True 且不再出现 `Not compiled with GPU support` 即表示 **已在 GPU 上执行**。

## 批量推理与耗时统计（约 100 张、多类别可视化）

脚本 `scripts/batch_benchmark.py` 会从公开数据集抽样（默认 **Caltech-101**，按类别分层约 100 张），对每张图跑一次前向，并统计 **平均 / 标准差 / 中位数推理耗时**（不含首次模型与权重加载；脚本内另有 `model_load_seconds`）。

在仓库根目录：

```bash
conda activate 3dmood
export HF_ENDPOINT=https://hf-mirror.com   # 若需镜像访问 BERT 等
python scripts/batch_benchmark.py --num_images 100
```

- 数据默认落在 **`--data_root`（默认 `data/`）** 下：Caltech 为 `data/caltech101/`，Oxford Pet 为 `data/oxford-iiit-pet/`（与 **torchvision** 目录名一致）。
- 若下载失败但本机已有完整数据，将数据放到上述路径后 **再跑一次**（已存在目录时会跳过下载）。
- 输出目录默认 **`runs/batch_<source>/`**：`vis/*_compare.png`（输入 | 输出）、`category_grid.png`（每类一图拼图）、`summary.json`（含 `per_class` 平均耗时）。

小数据集快测（需 `pip install datasets`，且能访问 Hugging Face）：

```bash
python scripts/batch_benchmark.py --source beans --num_images 30
```

自有图片目录：

```bash
python scripts/batch_benchmark.py --source folder --image_dir /path/to/images --text_prompt chair.table.car
```

## FastAPI 服务与 Docker

仓库现在提供：

- `scripts/serve_fastapi.py`：FastAPI 单进程服务
- `Dockerfile`：固定 `CUDA 12.4 + torch 2.4.1 cu124` 的 GPU 镜像

### 本地启动 FastAPI

先补服务依赖：

```bash
pip install fastapi uvicorn python-multipart
```

启动：

```bash
python scripts/serve_fastapi.py --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

推理请求示例：

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "files=@assets/demo/rgb.png" \
  -F "prompt=chair.table" \
  -F "return_vis=false"
```

### 返回格式

`POST /predict` 直接返回 JSON。每张图片在 `results` 中有一项，主要字段为：

- `file_name`
- `inference_seconds`
- `num_predictions`
- `cuboids`

其中 `cuboids` 为数组，每个 cuboid 包含：

- `score`
- `class_id`
- `label`
- `bbox_2d_xyxy`
- `center_cam`
- `dimensions_wlh`
- `dimensions_whl`
- `rotation_quat`
- `depth`

若请求里设置 `return_vis=true`，则每张图额外返回 `vis_image_base64`。

### 批量调用脚本

如果你想对一批图片批量请求服务、并把每张图的 JSON 和预测图分别落盘，可以直接用 `scripts/batch_service.py`。它支持两种模式：

```bash
# 多图一次 POST，推荐
python scripts/batch_service.py \
  --image_dir data/benchmark_datasets/highway_camera_norway_100/images \
  --prompt car.vehicle.truck.bus.van.motorcycle.person.traffic.light.traffic.sign.road \
  --request_mode batch \
  --batch_size 8 \
  --output_dir runs/service_highway_camera_norway_100

# 单图逐个 POST，便于排查
python scripts/batch_service.py \
  --image_dir data/benchmark_datasets/highway_camera_norway_100/images \
  --prompt car.vehicle.truck.bus.van.motorcycle.person.traffic.light.traffic.sign.road \
  --request_mode single \
  --output_dir runs/service_highway_camera_norway_100_single
```

输出目录下会生成：

- `json/`：每张图对应的响应 JSON
- `vis/`：每张图对应的预测可视化 PNG
- `batches/`：每次 POST 的原始响应 JSON
- `summary.json`：汇总统计

脚本的全部可选参数如下：

- `--base_url`：FastAPI 服务基地址
- `--endpoint`：请求路径，默认 `/predict`
- `--image_dir`：待推理图片目录
- `--image_glob`：筛选图片的 glob
- `--recursive`：是否递归搜索子目录
- `--num_images`：最多处理多少张图，`0` 表示全部
- `--shuffle`：处理前是否打乱顺序
- `--seed`：打乱时的随机种子
- `--output_dir`：输出目录
- `--prompt`：点号分隔的文本 prompt
- `--intrinsics_json`：可选的 3x3 相机内参 JSON
- `--return_vis`：是否让服务返回 base64 可视化图
- `--request_mode`：`batch` 或 `single`
- `--batch_size`：`batch` 模式下每次 POST 的图片数
- `--timeout`：单次 HTTP 超时秒数
- `--skip_existing`：已存在 JSON 时跳过
- `--save_batch_json`：是否保存每次 POST 的原始响应
- `--score_threshold`：覆盖服务端的分数阈值
- `--max_per_image`：覆盖服务端的单图最大框数
- `--nms`：覆盖服务端的 NMS 开关
- `--class_agnostic_nms`：覆盖服务端的跨类 NMS 开关
- `--nms_iou_threshold`：覆盖服务端的 NMS IoU 阈值
- `--resize_h`：覆盖服务端的预处理高度
- `--resize_w`：覆盖服务端的预处理宽度

### Docker 构建与运行

```bash
docker build -t 3d-mood-fastapi .
```

如果你更想用一条命令快速启动，可以直接用仓库根目录的 `compose.yaml`：

```bash
docker compose up --build
```

运行前需保证宿主机已安装 **NVIDIA Container Toolkit**，且驱动满足 `cu124` 的最低要求。启动示例：

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/torch:/root/.cache/torch \
  3d-mood-fastapi
```

### 完全离线（无公网）运行

容器内若无法访问外网，首次需在**能联网的机器**上准备好：

- 3D-MOOD Swin-B 权重：默认会缓存到 `~/.cache/torch/hub/checkpoints/gdino3d_swin-b_120e_omni3d_834c97.pt`
- 文本侧 `bert-base-uncased`：在 Hugging Face 缓存目录中应有 `models--bert-base-uncased`（例如 `~/.cache/huggingface/hub/`）

启动时挂载上述缓存为只读，并开启 Transformers 离线模式，同时用 `--ckpt` 指向容器内挂载路径，避免启动阶段再拉权重：

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -v ~/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints:ro \
  -v ~/.cache/huggingface:/root/.cache/huggingface:ro \
  3d-mood-fastapi \
  python scripts/serve_fastapi.py --host 0.0.0.0 --port 8000 \
    --ckpt /root/.cache/torch/hub/checkpoints/gdino3d_swin-b_120e_omni3d_834c97.pt
```

镜像默认 `CMD` 不带 `--ckpt`；仅在需要「容器内零下载」时用上述方式覆盖启动命令。

若你的机器无法兼容 `cu124`，建议另做一个与本机驱动匹配的镜像 tag，而不是强行复用同一个 Dockerfile 输出。

## pip 使用清华源时的说明

若 `pip` 配置了清华等镜像，安装 `vis4d`、PyTorch 通常仍正常。若个别包版本解析异常，可对单条命令临时指定官方索引或使用 PyTorch 专用 wheel 源（见上节）。

## 官方测试命令（需数据集）

完整 `vis4d test` 需按 [README](../README.md) 与 [DATA.md](./DATA.md) 准备 HDF5 等数据，不在本文 Demo 范围内。

---

**验证记录摘要**：浅克隆安装 `vis4d_cuda_ops`；`transformers<5`；HF 不可达时用 `HF_ENDPOINT` + 必要时手动缓存 Swin-B 权重；**GPU**：`torch 2.4.1+cu124` + Conda `cuda-toolkit`/`cuda-nvcc` + `CUDA_HOME` 与 `targets/.../include` 头文件处理 + `--force-reinstall` 重编译 `vis4d_cuda_ops` 后，`scripts/demo.py` 可在 RTX 3090 上正常运行。
