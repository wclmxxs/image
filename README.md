# Image Model Lab — AWS 8×H200

在一台 8×H200 Linux 机器上，用模型名调用生图／编辑模型。提供统一 REST API、固定版本权重下载、独立 Docker 推理环境、串行队列和 1K／2K 测试工具。

**交付状态：部署代码和 CPU 控制层测试已完成；尚未在 H200 上构建容器或完成真实推理验收。** GPU 数量是初始配置，不是性能实测。Mage 的官方代码已接入，但目前未核实到可下载的 Edit 权重，默认只准备其余五个模型。

## 支持的模型

| 调用名 | 型号 | 本仓库适配能力 | 可见 GPU | 默认采样 |
| --- | --- | --- | --- | --- |
| `cosmos` | Cosmos3-Super-Text2Image | 文生图，vLLM-Omni | 0–3 | 50 步，CFG 4 |
| `flux` | FLUX.2 dev | 文生图、参考图编辑 | 0 | 50 步，CFG 4 |
| `ideogram` | Ideogram 4 官方 FP8 | 文生图 | 0 | QUALITY 48 步 |
| `hunyuan` | HunyuanImage-3.0 | 文生图 | 0–3 | 50 步 |
| `hunyuan-distil` | HunyuanImage-3.0-Instruct-Distil | 文生图、参考图编辑 | 0–7 | 8 步，think_recaption |
| `mage` | Mage-Flow-Edit | 参考图编辑，需要本地权重 | 0 | 30 步，CFG 5 |

也接受完整型号，例如 `"FLUX.2 dev"`。别名、版本 SHA、GPU 分配和参数白名单见 [config/models.json](config/models.json)。Ideogram 未接入未经核实的自托管编辑接口；Mage Edit 必须提供参考图。不会用 fal 或其他型号替代不可用模型。

默认一个 worker 常驻。连续请求同一模型复用权重，切换模型时释放旧 worker 的 GPU 后加载下一个。Hunyuan 使用官方 `device_map=auto`，可见 GPU 数量不代表等量切分或张量并行加速。

## 一键启动

准备 Ubuntu GPU DLAMI、x86_64、完整的 8×H200 141GB、可用驱动、充足系统内存，以及约 **2TB 专用数据盘**。数据盘与 Docker 所在盘都需留空间。脚本不会安装／替换显卡驱动，也不会格式化磁盘。

**启动默认接管整台机器的 8 张 GPU：会停止已有 GPU 任务。** 权重下载和镜像构建完成后才执行清理，随后检查 CUDA 并启动 API。只想检查时用 `--check`；要保留其他任务时用 `--no-gpu-cleanup`，GPU 忙碌会直接退出。

**临时凭据状态：** 本次公开提交的 `.hf-token.env` 已被 Hugging Face 标记为 `Invalidated`，接口返回 HTTP 401；它不能用于下载。必须配置新的、未公开的账号 Token 才能启动受限模型。

启动读取优先级为调用 shell 的非空 `HF_TOKEN` → `.env` 的非空 `HF_TOKEN` → `.hf-token.env`。因此，在 `.env` 中提供有效 Token 即可覆盖失效的临时值。`.hf-token.env` 从 Docker 构建上下文排除。

自行配置账号或更换凭据时：

1. 用同一个 Hugging Face 账号取得 [FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev)、[Ideogram 4 FP8](https://huggingface.co/ideogram-ai/ideogram-4-fp8)、[Cosmos Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail) 的访问权。需要审批的仓库须等待批准，再从该账号的 [Token 设置](https://huggingface.co/settings/tokens) 创建读取 Token；如果使用 fine-grained Token，还需允许读取账号有权访问的 public gated repositories。
2. 首次执行 `cp config/env.example .env`，编辑 `HF_TOKEN` 和 `DATA_ROOT`；已有 `.env` 时直接编辑，避免覆盖配置。推荐把数据目录设成已挂载的数据盘目录，例如 `/mnt/nvme/image-lab`。`API_KEY` 留空时由脚本生成并保存在 `.env`，不会打印。
3. 在 AWS 上进入仓库，执行：

```bash
./start.sh --profile aws-8xh200
```

`install.sh` 是同一入口的别名。首次启动会检查机器和全部所选权重的访问权限，然后下载、构建镜像和启动 API；耗时取决于网络、存储和编译。模型在首次请求时加载。

如果 Ubuntu AMI 缺少 Docker／NVIDIA Container Toolkit，可用 `./start.sh --install-runtime` 安装。存在其他运行中容器时不会自动重启 Docker。账号没有 Docker 权限时，按提示用 `sudo ./start.sh`，或配置 Docker 用户组后重新登录。

```bash
./start.sh --models flux,hunyuan-distil   # 只准备部分模型
./start.sh --check-access               # 只检查下载权限，不查询／停止 GPU 任务
./start.sh --models ideogram --check     # 主机／访问权预检，不下载大权重、不启动服务
./start.sh --no-gpu-cleanup              # 不清理其他任务；GPU 被占用时拒绝启动
./status.sh
./stop.sh
```

`--models` 选择本次准备和构建的模型，其他已缓存模型仍保留。重跑启动会重建 API，当前运行任务标为 `interrupted`，排队任务恢复。停止只清理本部署的容器，保留权重与结果。

遇到 `401/403` 或 `GatedRepoError`，先执行 `./start.sh --check-access`。此命令准备配置／数据目录、构建 CPU 下载助手镜像并验证未缓存权重的下载权限，不下载模型大权重，也不检查或清理 GPU。输出区分缺少 Token、Token 被拒绝和仓库权限不足，并显示当前认证用户名及需要申请访问的仓库链接。所有所选模型通过权限检查后，正常启动才开始下载权重。

下载助手读取 `.env` 中的 `HF_TOKEN`，也支持 `export HF_TOKEN`；调用脚本的 shell 中非空 Token 优先于 `.env`，引号由 shell 正确解析。仅在宿主机执行 `hf auth login` 不会把凭据传入容器；通过 `sudo` 启动时建议把 Token 配在 `.env`，避免环境变量被过滤。不要把 Token 粘贴到日志或提交到仓库。已准备完整的本地缓存无需重新验证 Token。

等待受限仓库批准期间，可以先运行 `./start.sh --models hunyuan,hunyuan-distil`，只准备两款公开的 Hunyuan 权重；正常启动仍会在下载和构建后执行前述 GPU 清理。

GPU 清理会先记录 PID、容器、systemd 服务和显存占用：

- GPU 容器：记录原 restart policy，将其设为 `no`，再执行 `docker stop`；不删除容器、镜像或数据卷，也不停止未使用 GPU 的容器。
- 独立 systemd GPU 服务：停止对应服务，阻止 `Restart=always` 立即拉起；不 disable 服务或停止 Docker／SSH／显卡基础服务。
- 普通 GPU 进程：先 TERM，等待 20 秒仍未退出才 KILL；用 Linux pidfd 和进程启动时间校验，避免误杀复用的 PID。
- 最多重新扫描三轮，处理自动重启的任务；每轮持续观察 30 秒，最后连续三次无计算进程且每卡显存占用不超过 1024MiB 才继续。异常残留不会伪装成清理成功，也不执行 GPU reset。

清理需要 root 权限，普通账号会通过 `sudo -n` 执行（AWS Ubuntu 通常配置无密码 sudo）。诊断记录保存在 `DATA_ROOT/diagnostics/gpu-cleanup-*.json`，包含恢复 Docker restart policy 所需的原值。需要恢复旧任务时，根据记录手动恢复策略并启动相应容器／服务。Kubernetes、Swarm 或独立用户 systemd 管理的 GPU 任务会明确报错，要求先停止其工作负载控制器；脚本不会盲目清理整个集群或用户会话。

默认 API：`http://127.0.0.1:18080`；交互式文档：`http://127.0.0.1:18080/docs`。`cached` 表示权重已准备；`active_model` 表示当前常驻 worker。**API 就绪不代表所有模型已通过真实出图验收。**

远程可用 `ssh -L 18080:127.0.0.1:18080 ubuntu@YOUR_AWS_HOST` 转发。设置 `BIND_HOST=0.0.0.0` 时需配置 AWS 安全组。业务接口要求 Bearer API key；服务供可信用户评测使用，控制容器挂载 Docker socket，不适合作为多租户公网服务。

## 按模型名测试

`./lab` 自动读取服务器 `.env`，只依赖系统 Python：

```bash
./lab models
./lab generate --model 'FLUX.2 dev' \
  --prompt '一只红狐坐在木桌旁，清晨自然光，摄影' \
  --width 1024 --height 1024 --seed 42 --output results/flux-1k.png

./lab generate --model hunyuan-distil \
  --prompt '将背景换成雪山，保持主体的外观' \
  --image /absolute/path/reference.png --output results/edit.png

./lab generate --model ideogram --prompt 'A coffee shop poster with the title MORNING' \
  --parameters '{"preset":"V4_QUALITY_48","prompt_mode":"template"}'

./lab job JOB_ID
./lab cancel JOB_ID
```

每次保存 PNG 和同名 JSON。多参考图重复传 `--image`。远程客户端可复制 `scripts/client.py`，设置环境变量 `API_KEY`，使用 `python3 client.py --url http://HOST:18080 ...`。

| 方法与路径 | 用途 |
| --- | --- |
| `GET /health` | 控制服务存活检查，无鉴权 |
| `GET /v1/models` | 型号、版本、参数、可用状态 |
| `POST /v1/uploads` | multipart `file`，返回参考图 ID |
| `POST /v1/jobs` | 异步提交，HTTP 202，返回任务 ID |
| `GET /v1/jobs/{id}` | 状态、耗时、结果和错误 |
| `POST /v1/jobs/{id}/cancel` | 取消任务 |
| `GET /v1/jobs/{id}/image` | 鉴权下载 PNG |
| `POST /v1/images/generations` | 简化的同步文生图兼容入口 |

异步请求体示例：

```json
{
  "model": "hunyuan-distil",
  "prompt": "一只红狐坐在木桌旁",
  "width": 1024,
  "height": 1024,
  "seed": 42,
  "images": [],
  "parameters": {"steps": 8, "bot_task": "think_recaption"}
}
```

`images` 填上传 ID，不接受任意服务器路径或 URL。参考图上限 25MiB／2400 万像素。每任务一张输出，`n > 1`、不支持的能力、未知参数和错误尺寸均报错。

兼容入口接受 `model/prompt/size/seed/n/parameters/response_format`。`size` 如 `1024x1024`，等待超过 55 秒会返回 **202 + job_id/poll_url**，任务继续执行；不能假定所有 OpenAI SDK 都能直接处理该响应。冷启动和跑批优先使用 `/v1/jobs`。`response_format=url` 返回需要鉴权的相对路径。

## 1K／2K 跑批与时间口径

```bash
./smoke-test.sh flux,hunyuan-distil
./lab benchmark --models flux,hunyuan-distil --repeat 3 --warmup 1
./lab benchmark --repeat 3 --warmup 1
```

[config/benchmark.jsonl](config/benchmark.jsonl) 包含五个文生图模型各一条 1024×1024、2048×2048 请求，保留各模型默认质量参数。每例先 warmup，再生成三次；按模型排序减少切换。输出 `summary.csv`、`results.jsonl`、逐次 PNG／JSON；warmup 明确标记。失败用例记录错误后继续其他用例，不重复失败请求。

编辑评测使用 [config/benchmark-edit.example.jsonl](config/benchmark-edit.example.jsonl)，先填真实参考图路径，再执行 `./lab benchmark --cases your-edit-cases.jsonl`。

**尺寸不能等同于能力承诺：** API 接受每边 256–2048、16 的倍数，超过约 1MP 标记为实验分辨率。Hunyuan 官方实现按分辨率集合选尺寸，适配器会提前拒绝不匹配的请求；所有输出 PNG 也再次验证尺寸。不会把 1K 上采样后当成原生 2K，也不会把失败耗时算作出图成绩。

| 记录字段 | 含义 |
| --- | --- |
| `queue_seconds` | 等待此前任务 |
| `load.load_seconds` | 模型切换、容器创建、权重加载和初始化；热复用为 0 |
| `generation_seconds` | 后端调用总墙钟时间，含编码、采样、解码及适用的扩写／审核；CUDA 同步计时 |
| `prompt_seconds` | 可独立计量的外置提示词处理；Hunyuan 原生改写无法拆分时为 null |
| `inference_seconds` | 后端时间减去可拆分的提示词处理；并非纯去噪内核耗时 |
| `client_seconds` | 上传、排队、加载、生成到收到完成状态，不含最后下载 PNG |

Cosmos 在 vLLM 子进程中执行，wrapper 的 PyTorch 峰值不能代表其显存；`nvidia-smi` 记录是完成后的快照，不是全程峰值。这里不提供未经 H200 实测的秒数。

## 提示词、质量与外部依赖

Ideogram 4 使用结构化 JSON。默认 `prompt_mode=template` 生成最小合法结构，方便普通文本调用，但没有 LLM 扩写能力，不能等同官方在线产品效果。质量评测建议提交完整 JSON（`prompt_mode=json`），或配置 `.env` 的 `IDEOGRAM_API_KEY` 并显式选 `prompt_mode=magic`。后者把提示词发送给 Ideogram 扩写服务，图像仍本地生成。官方当前说明扩写接口免费，详情及账号条件见 [官方提示词文档](https://github.com/ideogram-oss/ideogram4/blob/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2/docs/prompting.md)。

Cosmos 默认普通文本，官方推荐结构化 JSON 提高质量。可用 `parameters={"prompt_mode":"json"}` 提交已扩写内容；仓库不自动调用远端 LLM。其审核保持开启，Guardrail／Qwen 辅助权重一并缓存和固定版本。

Ideogram 可配置官方 Hive 接口：`HIVE_TEXT_MODERATION_KEY`、`HIVE_VISUAL_MODERATION_KEY`。命中审核则任务失败；未配置项在结果标为 `not_configured`。启用相应外部服务会发送对应内容给服务商。

## Mage 和商业许可

[Mage 官方源码](https://github.com/microsoft/Mage/tree/76bec2bb3818863f470de7e867c2dc7f1d0bfd83/mage_flow) 已接入；本次未能匿名获取 `microsoft/Mage-Flow-Edit` 的有效权重修订。取得授权的完整 checkpoint 后，在 `.env` 设置 `MAGE_LOCAL_PATH=/absolute/path/to/checkpoint`：

```bash
./start.sh --models mage
./lab generate --model Mage-Flow-Edit --prompt 'Change the background to a snowy mountain' \
  --image /absolute/path/reference.png --output results/mage.png
```

checkpoint 需包含 `model_index.json`、Edit transformer、文本编码器、VAE 等完整组件。首次复制到 `DATA_ROOT/models/mage` 并记录 SHA256 清单；直接放在该目标目录可避免复制。准备成功后才开放调用，兼容性仍需首次 GPU 推理验证。

仓库不打包权重或授予第三方商业使用权；下载门控授权不等于商业自托管授权：

- Cosmos3：核对 [官方模型卡](https://huggingface.co/nvidia/Cosmos3-Super-Text2Image) 的 OpenMDW 条款。
- FLUX.2 dev：权重有非商业许可，商业自托管需取得 BFL 适用授权，不能用 fal 调用资格代替。
- Ideogram 4：核对 [权重许可](https://huggingface.co/ideogram-ai/ideogram-4-fp8/blob/main/LICENSE.md) 和商业方案。
- Hunyuan：核对 [社区许可](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/6e9113a692a27a0751d82aba3b2015a876646c03/LICENSE) 的地域、规模等限制。
- Mage：源码 MIT 不自动覆盖所获权重的许可。

脚本不会自动购买或代为同意许可。本地出图没有 fal 调用费；AWS、存储及选用的外部服务按各自账户计费。

## 调优、诊断与验收

模型隔离 CUDA／PyTorch／Transformers 环境；权重 SHA、源码 commit、CUDA／Cosmos 镜像 digest 固定。结果记录实际依赖版本和镜像 ID。构建执行 `pip check` 与入口导入检查。

`LOAD_TIMEOUT` 默认 3600 秒，`JOB_TIMEOUT` 默认 1800 秒。首次 FlashInfer／CUDA 编译可能较长。OOM／超时后自动释放当前 worker，下个任务重新加载，不自动修改精度、步数或型号。FLUX 2K 若 OOM，可将 `config/models.json` 中其 `gpus` 改为 `[0,1]` 后重启，使用 `device_map=balanced`，随后重新实测。Cosmos 固定官方 4×H200 并行配置，不能仅改 GPU 数量。

```bash
docker compose logs --tail 100 api
./lab job JOB_ID
./stop.sh
```

`DATA_ROOT/sessions/<session>/worker.log` 是模型日志；`jobs/<job-id>/` 存结果；`models/` 存权重；`cache/` 存辅助权重和编译缓存。日志可能含测试提示词，勿连同 `.env` 分享。删除相应 `prepared/<model>.json` 后重跑启动可重新准备模型；不提供自动清盘操作。

本机 CPU 测试：

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest -q
.venv/bin/ruff check image_lab scripts/*.py tests
bash -n start.sh install.sh stop.sh status.sh smoke-test.sh lab scripts/*.sh
```

模拟执行器只存在于 `tests/`，生产无模拟 GPU 自动降级选项。AWS 验收顺序：访问权预检 → 镜像构建／导入检查 → smoke test → 1K／2K 和参考图编辑跑批；实际报告保存后，才能确认机器上的可用性和速度。
