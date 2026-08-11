# # SmolVLA × LIBERO-plus Spatial LoRA Fine-tuning
# 
# `lerobot/smolvla_libero_plus`を初期重みとして、
# LIBERO-Spatialの10タスクをLoRAで追加学習します。
# 
# 学習後はLoRAを元モデルへマージし、次の2モデルを
# 同じLIBERO-plus Spatial環境で比較します。
# 
# - 追加学習前のLIBERO-plus重み
# - Spatial追加学習後のマージ済みモデル
# 
# **既定条件**
# 
# - Spatial 10タスク × 各5エピソード
# - 3,000 training steps
# - 100 stepsごとにlossを表示
# - 評価は10タスク × 各3エピソード
# ## 1. Colabランタイムを確認する
# 
# ColabのランタイムをGPUへ変更してから実行してください。
import importlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# ★★★ PREP_ONLY モード（CPU ランタイムでダウンロードのみ実行）★★★
# ============================================================
#
# True  : CPU ランタイムでも動くように、GPU チェックを緩和
#         Cell 7.5 まで実行してキャッシュを作成し、Drive に保存して終了
#         Cell 8 以降は実行しない前提
#
# False : GPU (A100/L4/T4) で通常実行、学習まで実施
#
# CPU ランタイム設定: ランタイム → ランタイムのタイプを変更 →
#                    ハードウェアアクセラレータ = None
# ============================================================
PREP_ONLY = False  # ★★★ CPU で DL 準備するときは True にする ★★★

# Colab 環境なら Python の厳密なバージョンチェックを緩和して警告のみ出す
IN_COLAB = False
try:
    import google.colab  # type: ignore
    IN_COLAB = True
except Exception:
    IN_COLAB = False

if sys.version_info < (3, 12):
    if IN_COLAB:
        print(
            "[WARN] Colab の Python が 3.12 未満です。動かない機能があるかもしれませんが継続します."
        )
    else:
        raise RuntimeError("Python 3.12以上が必要です。")

if PREP_ONLY:
    print(">>> PREP_ONLY モード: CPU ランタイムで DL 準備のみ実施")
    if torch.cuda.is_available():
        print(f"    (GPU 検出: {torch.cuda.get_device_name(0)} — CPU 相当で処理)")
    else:
        print(f"    (GPU なし — 想定通り)")
else:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPUランタイムを選択してください。"
            "PREP_ONLY=True にすれば CPU でも動作します。"
        )
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ## 2. システムパッケージを準備する
# 
# LeRobot、動画デコード、MuJoCoで必要になるパッケージを導入します。
# ## 1.5. Google Drive をマウントし、永続キャッシュを設定する
try:
    from google.colab import drive as _colab_drive
    import shutil as _shutil

    DRIVE_MOUNTPOINT = Path("/content/drive")
    if not (DRIVE_MOUNTPOINT / "MyDrive").is_dir():
        _colab_drive.mount(str(DRIVE_MOUNTPOINT))

    # 永続キャッシュ（Drive 上）
    DRIVE_CACHE_ROOT = DRIVE_MOUNTPOINT / "MyDrive" / "parc2026_cache"
    DRIVE_HF_CACHE = DRIVE_CACHE_ROOT / "hf"
    DRIVE_LEROBOT_CACHE = DRIVE_CACHE_ROOT / "lerobot"
    DRIVE_HF_CACHE.mkdir(parents=True, exist_ok=True)
    DRIVE_LEROBOT_CACHE.mkdir(parents=True, exist_ok=True)

    # 高速キャッシュ（/content 上）
    LOCAL_HF_CACHE = Path("/content/hf_cache")
    LOCAL_LEROBOT_CACHE = Path("/content/lerobot_cache")
    LOCAL_HF_CACHE.mkdir(parents=True, exist_ok=True)
    LOCAL_LEROBOT_CACHE.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(LOCAL_HF_CACHE)
    os.environ["HF_LEROBOT_HOME"] = str(LOCAL_LEROBOT_CACHE)

    # 完了マーカー（DL が確実に終わった証明）
    CACHE_MARKER = DRIVE_CACHE_ROOT / "download_complete.txt"

    usage = _shutil.disk_usage(str(DRIVE_MOUNTPOINT / "MyDrive"))
    free_gb = usage.free / (1024 ** 3)
    print(f"[Drive] free = {free_gb:.1f} GB")

    if free_gb < 5.0:
        print(f"[WARN] Drive 残 {free_gb:.1f} GB は Spatial (4GB) にもギリギリ")
        print("       Google One 100GB (¥290/月) 加入を強く推奨")
    elif free_gb < 20.0:
        print(f"[OK] Drive 残 {free_gb:.1f} GB: Spatial+Object は保存可能、FULL は不可")
    else:
        print(f"[OK] Drive 残 {free_gb:.1f} GB: FULL データセットも保存可能")


    def restore_cache_from_drive():
        import time as _time
        if not CACHE_MARKER.exists():
            print("[Restore] マーカー無し。Drive にキャッシュが確実な形で無いのでスキップ")
            return False

        for src, dst, label in [
            (DRIVE_HF_CACHE, LOCAL_HF_CACHE, "hf_cache"),
            (DRIVE_LEROBOT_CACHE, LOCAL_LEROBOT_CACHE, "lerobot_cache"),
        ]:
            if not any(src.iterdir()):
                continue
            t0 = _time.time()
            print(f"[Restore] {label}: Drive → /content 復元中...", flush=True)
            subprocess.run(
                ["rsync", "-a", f"{src}/", f"{dst}/"],
                check=False,
            )
            print(f"[Restore] {label}: 完了 ({_time.time() - t0:.0f}s)", flush=True)
        return True


    def save_cache_to_drive():
        import time as _time
        for src, dst, label in [
            (LOCAL_HF_CACHE, DRIVE_HF_CACHE, "hf_cache"),
            (LOCAL_LEROBOT_CACHE, DRIVE_LEROBOT_CACHE, "lerobot_cache"),
        ]:
            if not any(src.iterdir()):
                continue
            t0 = _time.time()
            print(f"[Save] {label}: /content → Drive 保存中...", flush=True)
            subprocess.run(
                ["rsync", "-a", f"{src}/", f"{dst}/"],
                check=False,
            )
            print(f"[Save] {label}: 完了 ({_time.time() - t0:.0f}s)", flush=True)

        # マーカーを最後に書く（途中失敗したら restore がスキップされる）
        CACHE_MARKER.write_text(f"completed at {_time.time()}")
        print(f"[Save] マーカー書込 → 次回セッションで復元可能")


    # 復元試行（初回は何も起きない）
    restored = restore_cache_from_drive()
    if not restored:
        print(">>> 初回セッション or マーカー無し。Cell 7.5 で DL 発生")
    else:
        print(">>> キャッシュ復元完了。Cell 7.5 は即完了する見込み")
except Exception:
    # 非 Colab 環境では google.colab が import できず、ここはスキップされる
    pass
def run_quiet(
    command: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    if check and result.returncode != 0:
        raise RuntimeError(result.stdout[-6000:])

    return result


run_quiet(["apt-get", "update", "-qq"])
run_quiet(
    [
        "apt-get",
        "install",
        "-y",
        "-qq",
        "ffmpeg",
        "git",
        "unzip",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libexpat1",
        "libfontconfig1-dev",
        "libmagickwand-dev",
    ]
)

print("System packages ready.")

# ## 3. LeRobotをインストールする
# 
# LeRobot `v0.6.0`を使用します。
# ColabでのLoRA学習に必要な互換性調整もこのセルで適用します。
LEROBOT_TAG = "v0.6.0"
LEROBOT_DIR = Path("/content/lerobot")
LEROBOT_SRC = LEROBOT_DIR / "src"

run_quiet(
    [
        sys.executable,
        "-m",
        "pip",
        "uninstall",
        "-y",
        "lerobot",
        "torchao",
    ],
    check=False,
)

shutil.rmtree(LEROBOT_DIR, ignore_errors=True)

run_quiet(
    [
        "git",
        "clone",
        "--quiet",
        "--depth",
        "1",
        "--branch",
        LEROBOT_TAG,
        "https://github.com/huggingface/lerobot.git",
        str(LEROBOT_DIR),
    ]
)

smolvlm_source = (
    LEROBOT_SRC
    / "lerobot"
    / "policies"
    / "smolvla"
    / "smolvlm_with_expert.py"
)

if not torch.cuda.is_bf16_supported():
    source = smolvlm_source.read_text(encoding="utf-8")
    source = source.replace(
        'torch_dtype="bfloat16",',
        'torch_dtype="float16",',
        1,
    )
    smolvlm_source.write_text(
        source,
        encoding="utf-8",
    )

train_script = (
    LEROBOT_SRC
    / "lerobot"
    / "scripts"
    / "lerobot_train.py"
)
source = train_script.read_text(encoding="utf-8")
source = source.replace(
    "logging.info(pformat(cfg.to_dict()))",
    "logging.debug(pformat(cfg.to_dict()))",
    1,
)
source = source.replace(
    "disable=inside_slurm(),",
    "disable=True,",
    1,
)
train_script.write_text(
    source,
    encoding="utf-8",
)

run_quiet(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--upgrade",
        "-e",
        f"{LEROBOT_DIR}[training,smolvla,peft]",
    ]
)

run_quiet(
    [
        sys.executable,
        "-m",
        "pip",
        "uninstall",
        "-y",
        "torchao",
    ],
    check=False,
)

for module_name in list(sys.modules):
    if (
        module_name == "lerobot"
        or module_name.startswith("lerobot.")
        or module_name == "torchao"
        or module_name.startswith("torchao.")
    ):
        del sys.modules[module_name]

sys.path = [
    item
    for item in sys.path
    if item not in {
        str(LEROBOT_DIR),
        str(LEROBOT_SRC),
    }
]
sys.path.insert(0, str(LEROBOT_SRC))
importlib.invalidate_caches()

try:
    importlib.metadata.version("torchao")
except importlib.metadata.PackageNotFoundError:
    pass
else:
    raise RuntimeError("torchaoの削除に失敗しました。")

import lerobot
import peft

if (
    LEROBOT_SRC.resolve()
    not in Path(lerobot.__file__).resolve().parents
):
    raise RuntimeError("LeRobotの読込先が正しくありません。")

print("LeRobot ready.")

# ## 4. 学習・評価条件を設定する
# 
# Spatialの10タスクから各5エピソードを選び、
# 合計50エピソードで追加学習します。
# 
# 評価を正式な10エピソード/taskへ近づける場合は、
# `EVAL_EPISODES_PER_TASK = 10`へ変更してください。
BASE_MODEL_REPO = "lerobot/smolvla_libero_plus"
BASE_MODEL_REVISION = (
    "7bb70aa5bc92b82c9239142775d3a173103567ff"
)

VLM_REPO = (
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
)

DATASET_REPO = "lerobot/libero_plus"
DATASET_REVISION = (
    "f3f49f426d75030177b18778374005bc12ccd588"
)

# ===== 修正: SPATIAL_TASK_NAMES を廃止し、全 40 タスク使用 =====
# 参照用に残す（Cell 13 の比較表用）。Cell 6 の選定では使用しない
SPATIAL_TASK_NAMES = [
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
]

# ===== データ拡張パラメータ =====
TRAIN_EPISODES_PER_TASK = 10
EXPECTED_MIN_TASKS = 30

# ===== 学習ハイパラ（A100 向け） =====
STEPS = 15000
LOG_FREQ = 500
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
FINAL_LEARNING_RATE = 3e-5
WARMUP_STEPS = 500

# ===== LoRA 設定拡張 =====
LORA_R = 32
LORA_ALPHA = 64
LORA_TARGET_MODULES_REGEX = (
    r"(model\.vlm_with_expert\.lm_expert\..*\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj))|"
    r".*\.(action_in_proj|action_time_mlp_in|action_time_mlp_out)"
)
LORA_FULL_TRAINING_MODULES = ["state_proj", "action_out_proj"]

SEED = 42

# ---- 評価設定（変更なし）----
EVAL_TASK_IDS = list(range(10))
EVAL_EPISODES_PER_TASK = 3
EVAL_SEED = 2026
FINETUNED_EVAL_DIR = Path(
    "/content/eval/spatial_lora"
)
COMPARISON_CSV_PATH = Path(
    "/content/libero_spatial_comparison.csv"
)
MERGED_ZIP_PATH = Path(
    "/content/smolvla_libero_plus_spatial_lora_merged.zip"
)

# ===== 追加: 出力/モデル保存ディレクトリ（Colab 実行向け） =====
OUTPUT_DIR = Path("/content/outputs/smolvla_libero_plus_spatial_lora")
MERGED_MODEL_DIR = Path("/content/smolvla_libero_plus_spatial_lora_merged")
BASELINE_MODEL_DIR = Path("/content/smolvla_libero_plus_baseline")
BASE_EVAL_DIR = Path("/content/eval/base")

MIXED_PRECISION = (
    "bf16"
    if torch.cuda.is_bf16_supported()
    else "fp16"
)

# ============================================================
# ★★★ SMOKE_TEST モード（最短完走確認用） ★★★
# ============================================================
#
# True  : Spatial 10 task × 2 ep = 20 episode / 200 steps / batch=4 / LoRA r=16
#         合計 40〜100 分で完走見込み。スコアは度外視、完走確認優先。
# False : 上の本番定数（400 episode / 15000 steps / batch=16 / LoRA r=32）
#
# 本番へ戻すときは、この 1 行を False に変えるだけ。
# ============================================================
SMOKE_TEST = True

if SMOKE_TEST:
    print(">>> SMOKE_TEST モード: 最短完走確認用に軽量構成へ切り替えます")

    # ---- Cell 6 でタスクを SPATIAL 10 に絞るための allowlist ----
    SMOKE_TEST_TASK_ALLOWLIST = set(SPATIAL_TASK_NAMES)

    # ---- データ選定 ----
    TRAIN_EPISODES_PER_TASK = 2       # 10 × 2 = 20 episode
    EXPECTED_MIN_TASKS = 5            # 10 タスク前提の検証を緩和

    # ---- 学習ハイパラを軽量化 ----
    STEPS = 200
    LOG_FREQ = 50
    BATCH_SIZE = 4
    LEARNING_RATE = 3e-4
    FINAL_LEARNING_RATE = 3e-5
    WARMUP_STEPS = 20

    # ---- LoRA を軽く（trainable と初期化時間を減らす）----
    LORA_R = 16
    LORA_ALPHA = 32

    # ---- 評価も最短（10 task × 1 ep = 10 rollout / モデル）----
    EVAL_EPISODES_PER_TASK = 1

    print(f"    tasks         : SPATIAL 10 tasks only")
    print(f"    episodes/task : {TRAIN_EPISODES_PER_TASK}")
    print(f"    total episodes: ~{TRAIN_EPISODES_PER_TASK * len(SPATIAL_TASK_NAMES)}")
    print(f"    steps         : {STEPS}")
    print(f"    batch_size    : {BATCH_SIZE}")
    print(f"    LoRA r/alpha  : {LORA_R}/{LORA_ALPHA}")
    print(f"    eval ep/task  : {EVAL_EPISODES_PER_TASK}")
else:
    print(">>> 本番構成で実行します（400 episode / 15000 steps / batch=16）")
    SMOKE_TEST_TASK_ALLOWLIST = None  # None = フィルタなし = 全 40 タスク使用

# ## 5. 公開ファイルの取得処理を用意する
# 
# キャッシュを優先し、匿名アクセスの制限時は自動的に再試行します。
import random
import time
from collections.abc import Callable
from typing import TypeVar

import httpx
from huggingface_hub import snapshot_download
from huggingface_hub.errors import (
    HfHubHTTPError,
    LocalEntryNotFoundError,
)

T = TypeVar("T")


def run_hf_with_retry(
    operation: Callable[[], T],
) -> T:
    last_error: BaseException | None = None

    for attempt in range(6):
        try:
            return operation()
        except (
            HfHubHTTPError,
            httpx.HTTPStatusError,
        ) as error:
            last_error = error
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)

            if status != 429 and "429" not in str(error):
                raise

            if attempt == 5:
                break

            headers = getattr(response, "headers", {}) or {}
            try:
                delay = float(
                    headers.get("Retry-After", 15)
                ) + 1
            except (TypeError, ValueError):
                delay = min(
                    120,
                    15 * (2**attempt) + random.random(),
                )

            time.sleep(delay)

    raise RuntimeError(
        "Hugging Faceからの取得に失敗しました。"
    ) from last_error


def cached_or_downloaded_snapshot(
    repo_id: str,
    revision: str,
    *,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> Path:
    try:
        return Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                token=False,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                local_files_only=True,
            )
        )
    except (
        LocalEntryNotFoundError,
        FileNotFoundError,
    ):
        return Path(
            run_hf_with_retry(
                lambda: snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    token=False,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    max_workers=1,
                )
            )
        )

# ## 6. Spatial学習データを選ぶ
# 
# 10タスクから各5エピソードを等間隔に選択します。
import re
from collections import defaultdict

from lerobot.datasets.dataset_metadata import (
    LeRobotDatasetMetadata,
)


def normalize_task_name(value: str) -> str:
    value = value.lower().replace("_", " ")
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def task_name_from_cell(value) -> str:
    if isinstance(value, str):
        return value

    try:
        if len(value) > 0:
            return str(value[0])
    except TypeError:
        pass

    return str(value)


def choose_evenly_spaced(
    episode_indices: list[int],
    count: int,
) -> list[int]:
    positions = [
        round(
            index
            * (len(episode_indices) - 1)
            / (count - 1)
        )
        for index in range(count)
    ]

    return [
        episode_indices[position]
        for position in positions
    ]


dataset_metadata = run_hf_with_retry(
    lambda: LeRobotDatasetMetadata(
        DATASET_REPO,
        revision=DATASET_REVISION,
    )
)

task_to_episodes: dict[str, list[int]] = defaultdict(list)

for episode_index, task_cell in enumerate(
    dataset_metadata.episodes["tasks"]
):
    task_to_episodes[
        task_name_from_cell(task_cell)
    ].append(int(episode_index))

available_by_normalized = {
    normalize_task_name(task_name): task_name
    for task_name in task_to_episodes
}

# ===== 修正版：全 libero_plus タスク or SPATIAL 10 タスクのみ =====
selected_by_task: dict[str, list[int]] = {}

for task_name, episode_list in task_to_episodes.items():
    # SMOKE_TEST 時は SPATIAL_TASK_NAMES に含まれるタスクのみ選ぶ
    if SMOKE_TEST_TASK_ALLOWLIST is not None and task_name not in SMOKE_TEST_TASK_ALLOWLIST:
        continue

    if len(episode_list) < TRAIN_EPISODES_PER_TASK:
        print(f"[SKIP] {task_name}: only {len(episode_list)} episodes")
        continue

    selected_by_task[task_name] = choose_evenly_spaced(
        episode_list,
        TRAIN_EPISODES_PER_TASK,
    )

EPISODE_INDICES = sorted(
    episode_index
    for episode_indices in selected_by_task.values()
    for episode_index in episode_indices
)

n_tasks_selected = len(selected_by_task)
n_episodes_total = len(EPISODE_INDICES)

print(f"Training data: {n_tasks_selected} tasks × ~{TRAIN_EPISODES_PER_TASK} ep = {n_episodes_total} episodes")

# もし SMOKE_TEST 時に Spatial タスクが 0 件だった場合の diagnostic
if SMOKE_TEST_TASK_ALLOWLIST is not None and n_tasks_selected == 0:
    print("[DEBUG] SPATIAL タスク名が一致しません。actual task names:")
    for name in list(task_to_episodes.keys())[:20]:
        print(f"  - {name!r}")
    raise RuntimeError("SPATIAL タスク名が libero_plus と一致しませんでした。")

# 検証
if n_tasks_selected < EXPECTED_MIN_TASKS:
    raise RuntimeError(
        f"Task coverage too small: {n_tasks_selected} < {EXPECTED_MIN_TASKS} tasks. "
        f"Expected around 40 tasks from libero_plus."
    )

# ## 7. 初期重みを準備する
BASE_MODEL_LOCAL = cached_or_downloaded_snapshot(
    BASE_MODEL_REPO,
    BASE_MODEL_REVISION,
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "train_config.json",
        "policy_preprocessor.json",
        "policy_preprocessor*.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor*.safetensors",
    ],
    ignore_patterns=[
        "README.md",
        "eval/**",
    ],
)

if not (
    BASE_MODEL_LOCAL / "model.safetensors"
).is_file():
    raise FileNotFoundError("Base model not found.")

print("Base model ready.")

# ## 7.5. データセットを事前構築する（進捗監視付き）
# 
# lerobot-train の内部で自動的に走るデータセット構築を、明示的に前倒しで実施。
# ここで HF Hub からの video/parquet DL と torchcodec の初期化が起きるので、
# 別スレッドで /content/hf_cache と /content/lerobot_cache のサイズを
# 30 秒毎に表示し、進捗を可視化する。
import threading
import time
import gc as _gc
from lerobot.datasets.lerobot_dataset import LeRobotDataset

CACHE_DIRS_TO_WATCH = [
    Path(os.environ["HF_HOME"]),
    Path(os.environ["HF_LEROBOT_HOME"]),
]

_dataset_ready_event = threading.Event()


def _du_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    try:
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total / (1024 ** 3)


def _cache_monitor():
    interval = 30  # 30 秒毎
    start = time.time()
    while not _dataset_ready_event.is_set():
        elapsed = int(time.time() - start)
        parts = " | ".join(
            f"{d.name}={_du_gb(d):.2f}GB"
            for d in CACHE_DIRS_TO_WATCH
        )
        print(
            f"[Dataset {elapsed // 60:02d}:{elapsed % 60:02d}] {parts}",
            flush=True,
        )
        if _dataset_ready_event.wait(interval):
            break


_monitor_thread = threading.Thread(target=_cache_monitor, daemon=True)
_monitor_thread.start()

print(
    f">>> Building LeRobotDataset "
    f"(repo={DATASET_REPO}, revision={DATASET_REVISION[:8]}, "
    f"episodes={len(EPISODE_INDICES)})",
    flush=True,
)

_ds_start = time.time()
_dataset = None

try:
    # v0.6.0 で受理される標準的なシグネチャで試す
    try:
        _dataset = run_hf_with_retry(
            lambda: LeRobotDataset(
                repo_id=DATASET_REPO,
                revision=DATASET_REVISION,
                episodes=EPISODE_INDICES,
                video_backend="torchcodec",
            )
        )
    except TypeError as _te:
        # video_backend 引数が拒否された場合のフォールバック
        print(f"[FALLBACK] first init failed: {_te}", flush=True)
        _dataset = run_hf_with_retry(
            lambda: LeRobotDataset(
                repo_id=DATASET_REPO,
                revision=DATASET_REVISION,
                episodes=EPISODE_INDICES,
            )
        )
finally:
    _dataset_ready_event.set()
    _monitor_thread.join(timeout=5)

_ds_elapsed = time.time() - _ds_start
print(
    f">>> Dataset built in {int(_ds_elapsed) // 60:02d}:{int(_ds_elapsed) % 60:02d} "
    f"len={len(_dataset)}",
    flush=True,
)

# 動画デコーダを暖める（初回サンプル取得）
print(">>> Warming up video decoder (fetching first sample)...", flush=True)
_sample_start = time.time()
try:
    _ = _dataset[0]
    _sample_elapsed = time.time() - _sample_start
    print(f">>> First sample fetched in {_sample_elapsed:.2f}s", flush=True)
except Exception as _e:
    print(f"[WARN] First sample fetch failed: {_e}", flush=True)
    print("      Cell 8 で lerobot-train が再度初期化を試みます", flush=True)

# メモリ解放（学習セル本体で再構築されるので、ここでは持ち続けない）
del _dataset
_gc.collect()

print(">>> Dataset cache is warm. Cell 8 (training) will start faster.")

# ## 8. LoRA学習を実行する
# 
# 100 stepごとに平均lossとlearning rateを表示します。
import re
import json
from collections import deque

episodes_json = (
    "["
    + ",".join(map(str, EPISODE_INDICES))
    + "]"
)

command = [
    "lerobot-train",
    f"--policy.path={BASE_MODEL_LOCAL}",
    f"--policy.vlm_model_name={VLM_REPO}",
    "--policy.push_to_hub=false",
    "--policy.repo_id=null",
    "--policy.input_features=null",
    "--policy.output_features=null",
    "--policy.empty_cameras=0",
    "--policy.freeze_vision_encoder=true",
    "--policy.train_expert_only=true",
    f"--policy.optimizer_lr={LEARNING_RATE}",
    "--policy.optimizer_weight_decay=1e-4",
    f"--policy.scheduler_decay_lr={FINAL_LEARNING_RATE}",
    f"--policy.scheduler_warmup_steps={WARMUP_STEPS}",
    f"--policy.scheduler_decay_steps={STEPS}",
    f"--dataset.repo_id={DATASET_REPO}",
    f"--dataset.revision={DATASET_REVISION}",
    f"--dataset.episodes={episodes_json}",
    "--dataset.use_imagenet_stats=false",
    "--dataset.video_backend=torchcodec",
    f"--output_dir={OUTPUT_DIR}",
    "--job_name=smolvla_libero_plus_spatial_lora",
    f"--steps={STEPS}",
    f"--batch_size={BATCH_SIZE}",
    "--num_workers=4",
    "--persistent_workers=true",
    "--env_eval_freq=0",
    "--eval_steps=0",
    f"--seed={SEED}",
    "--save_checkpoint=true",
    f"--save_freq={STEPS // 4}",
    "--save_checkpoint_to_hub=false",
    f"--log_freq={LOG_FREQ}",
    "--wandb.enable=false",
    "--peft.method_type=LORA",
    f"--peft.r={LORA_R}",
    f"--peft.lora_alpha={LORA_ALPHA}",
    f"--peft.target_modules={LORA_TARGET_MODULES_REGEX}",
    "--peft.full_training_modules=" + json.dumps(LORA_FULL_TRAINING_MODULES),
]

training_env = os.environ.copy()
training_env["PYTHONPATH"] = (
    str(LEROBOT_SRC)
    + os.pathsep
    + training_env.get("PYTHONPATH", "")
)
training_env["ACCELERATE_MIXED_PRECISION"] = (
    MIXED_PRECISION
)
training_env["PYTHONUNBUFFERED"] = "1"
training_env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
training_env["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
training_env["HF_HUB_VERBOSITY"] = "error"
training_env["TQDM_DISABLE"] = "1"
training_env["PYTHONWARNINGS"] = "ignore"

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

print("Preparing data and starting training...")

process = subprocess.Popen(
    command,
    cwd=LEROBOT_DIR,
    env=training_env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

recent_lines: deque[str] = deque(maxlen=80)
report_step = LOG_FREQ

assert process.stdout is not None

# セットアップ主要マーカー（step:/loss: 以外にも表示するもの）
SETUP_MARKERS = (
    "Creating dataset",
    "Creating policy",
    "Wrapping model",
    "Using PEFT",
    "Creating optimizer",
    "Output dir:",
    "num_learnable_params",
    "num_total_params",
    "Start offline training",
    "Loading",
    "Downloaded",
    "Downloading",
    "Resolving data files",
    "Fetching",
)

for raw_line in process.stdout:
    line = raw_line.replace("\r", "").strip()
    if not line:
        continue
    recent_lines.append(line)

    # セットアップ主要マーカー
    if any(marker in line for marker in SETUP_MARKERS):
        print(f"[SETUP] {line[:200]}", flush=True)
        continue

    # 学習中のログ
    if "step:" in line and "loss:" in line:
        loss_match = re.search(r"loss:([0-9.eE+-]+)", line)
        lr_match = re.search(r"lr:([0-9.eE+-]+)", line)
        loss = loss_match.group(1) if loss_match else "n/a"
        lr = lr_match.group(1) if lr_match else "n/a"
        print(
            f"step {report_step:4d}/{STEPS}  "
            f"loss={loss}  lr={lr}",
            flush=True,
        )
        report_step += LOG_FREQ

return_code = process.wait()

if return_code != 0:
    print("\n".join(recent_lines))
    raise RuntimeError(
        f"Training failed: {return_code}"
    )

print("Training complete.")

# ## 9. LoRAをマージしてモデル全体を保存する
# 
# LoRA差分を元weightへ統合し、通常のLeRobotモデルとして保存します。
import contextlib
import gc
import io
import json

from peft import PeftModel
from safetensors import safe_open
from lerobot.configs import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
)

checkpoint_dir = (
    OUTPUT_DIR
    / "checkpoints"
    / f"{STEPS:06d}"
    / "pretrained_model"
)

if not (
    checkpoint_dir / "adapter_model.safetensors"
).is_file():
    raise FileNotFoundError("Final adapter not found.")

gc.collect()
torch.cuda.empty_cache()

merge_config = PreTrainedConfig.from_pretrained(
    checkpoint_dir
)
merge_config.device = "cpu"
merge_config.pretrained_path = BASE_MODEL_LOCAL
merge_config.use_peft = False

quiet_output = io.StringIO()

with (
    contextlib.redirect_stdout(quiet_output),
    contextlib.redirect_stderr(quiet_output),
):
    base_policy = SmolVLAPolicy.from_pretrained(
        BASE_MODEL_LOCAL,
        config=merge_config,
        strict=False,
    )

    peft_policy = PeftModel.from_pretrained(
        base_policy,
        checkpoint_dir,
        is_trainable=False,
        torch_device="cpu",
    )

    merged_policy = peft_policy.merge_and_unload(
        safe_merge=True
    )

shutil.rmtree(MERGED_MODEL_DIR, ignore_errors=True)
MERGED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

merged_policy.config.use_peft = False
merged_policy.config.pretrained_path = None
merged_policy.config.push_to_hub = False
merged_policy.config.repo_id = None
merged_policy.config.device = None
merged_policy.config.load_vlm_weights = False
merged_policy.config.vlm_model_name = VLM_REPO

merged_policy.save_pretrained(MERGED_MODEL_DIR)

for pattern in [
    "policy_preprocessor.json",
    "policy_preprocessor*.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor*.safetensors",
]:
    for source_path in checkpoint_dir.glob(pattern):
        shutil.copy2(
            source_path,
            MERGED_MODEL_DIR / source_path.name,
        )

merged_weights_path = (
    MERGED_MODEL_DIR / "model.safetensors"
)

with safe_open(
    merged_weights_path,
    framework="pt",
    device="cpu",
) as weights:
    if any(
        "lora_" in key.lower()
        for key in weights.keys()
    ):
        raise RuntimeError(
            "LoRA parameters remain after merge."
        )

del peft_policy
del base_policy
del merged_policy

gc.collect()
torch.cuda.empty_cache()

print("Merged model ready.")

# ## 10. 比較用ベースラインを準備する
# 
# 公開weightを追加学習モデルと同じ入力schema・processorへ揃えます。
baseline_config = PreTrainedConfig.from_pretrained(
    MERGED_MODEL_DIR
)
baseline_config.device = "cpu"
baseline_config.pretrained_path = BASE_MODEL_LOCAL
baseline_config.use_peft = False
baseline_config.load_vlm_weights = False

quiet_output = io.StringIO()

with (
    contextlib.redirect_stdout(quiet_output),
    contextlib.redirect_stderr(quiet_output),
):
    baseline_policy = SmolVLAPolicy.from_pretrained(
        BASE_MODEL_LOCAL,
        config=baseline_config,
        strict=False,
    )

shutil.rmtree(BASELINE_MODEL_DIR, ignore_errors=True)
BASELINE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

baseline_policy.config.use_peft = False
baseline_policy.config.pretrained_path = None
baseline_policy.config.push_to_hub = False
baseline_policy.config.repo_id = None
baseline_policy.config.device = None
baseline_policy.config.load_vlm_weights = False
baseline_policy.config.vlm_model_name = VLM_REPO
baseline_policy.save_pretrained(BASELINE_MODEL_DIR)

for pattern in [
    "policy_preprocessor.json",
    "policy_preprocessor*.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor*.safetensors",
]:
    for source_path in MERGED_MODEL_DIR.glob(pattern):
        shutil.copy2(
            source_path,
            BASELINE_MODEL_DIR / source_path.name,
        )

del baseline_policy
gc.collect()
torch.cuda.empty_cache()

print("Baseline ready.")

# ## 11. LIBERO-plus評価環境を準備する
# 
# MuJoCo、LIBERO-plus fork、評価assetsを導入します。
from huggingface_hub import hf_hub_download

LIBERO_PLUS_SHA = "4976dc3"
LIBERO_PLUS_DIR = Path("/content/LIBERO-plus")
LIBERO_PLUS_PACKAGE_ROOT = (
    LIBERO_PLUS_DIR / "libero" / "libero"
)
LIBERO_PLUS_ASSETS_DIR = (
    LIBERO_PLUS_PACKAGE_ROOT / "assets"
)

os.environ["MUJOCO_GL"] = "egl"

run_quiet(
    [
        sys.executable,
        "-m",
        "pip",
        "uninstall",
        "-y",
        "hf-libero",
        "libero",
        "robosuite",
    ],
    check=False,
)

run_quiet(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "robosuite==1.4.1",
        "bddl==1.0.1",
        "easydict==1.13",
        "mujoco==3.7.0",
        "matplotlib==3.10.8",
        "Wand==0.6.13",
        "scikit-image==0.25.2",
        "gym==0.26.2",
    ]
)

if (
    importlib.metadata.version("robosuite")
    != "1.4.1"
):
    raise RuntimeError(
        "robosuite 1.4.1 is required."
    )

if not (LIBERO_PLUS_DIR / ".git").is_dir():
    shutil.rmtree(
        LIBERO_PLUS_DIR,
        ignore_errors=True,
    )
    run_quiet(
        [
            "git",
            "clone",
            "--quiet",
            "https://github.com/sylvestf/LIBERO-plus.git",
            str(LIBERO_PLUS_DIR),
        ]
    )

checkout = run_quiet(
    [
        "git",
        "-C",
        str(LIBERO_PLUS_DIR),
        "checkout",
        "--quiet",
        LIBERO_PLUS_SHA,
    ],
    check=False,
)

if checkout.returncode != 0:
    run_quiet(
        [
            "git",
            "-C",
            str(LIBERO_PLUS_DIR),
            "fetch",
            "--quiet",
            "--depth",
            "1",
            "origin",
            LIBERO_PLUS_SHA,
        ]
    )
    run_quiet(
        [
            "git",
            "-C",
            str(LIBERO_PLUS_DIR),
            "checkout",
            "--quiet",
            LIBERO_PLUS_SHA,
        ]
    )

run_quiet(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--no-deps",
        "-e",
        str(LIBERO_PLUS_DIR),
    ]
)

if not LIBERO_PLUS_ASSETS_DIR.is_dir():
    assets_root = Path(
        "/content/libero_plus_assets"
    )
    archive_path = Path(
        run_hf_with_retry(
            lambda: hf_hub_download(
                repo_id="Sylvest/LIBERO-plus",
                repo_type="dataset",
                filename="assets.zip",
                local_dir=assets_root,
                token=False,
            )
        )
    )
    extract_dir = assets_root / "extract"

    shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    run_quiet(
        [
            "unzip",
            "-q",
            str(archive_path),
            "-d",
            str(extract_dir),
        ]
    )

    candidates = sorted(
        [
            path
            for path in extract_dir.rglob("assets")
            if path.is_dir()
        ],
        key=lambda path: len(path.parts),
    )

    if not candidates:
        raise FileNotFoundError(
            "LIBERO-plus assets not found."
        )

    LIBERO_PLUS_ASSETS_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    shutil.move(
        str(candidates[0]),
        str(LIBERO_PLUS_ASSETS_DIR),
    )
    shutil.rmtree(assets_root, ignore_errors=True)

libero_config_dir = Path.home() / ".libero"
libero_config_dir.mkdir(
    parents=True,
    exist_ok=True,
)
(libero_config_dir / "config.yaml").write_text(
    "\n".join(
        [
            f"assets: {LIBERO_PLUS_ASSETS_DIR}",
            (
                "bddl_files: "
                f"{LIBERO_PLUS_PACKAGE_ROOT / 'bddl_files'}"
            ),
            (
                "datasets: "
                f"{LIBERO_PLUS_PACKAGE_ROOT.parent / 'datasets'}"
            ),
            (
                "init_states: "
                f"{LIBERO_PLUS_PACKAGE_ROOT / 'init_files'}"
            ),
        ]
    )
    + "\n",
    encoding="utf-8",
)

eval_script = (
    LEROBOT_SRC
    / "lerobot"
    / "scripts"
    / "lerobot_eval.py"
)
source = eval_script.read_text(encoding="utf-8")

source = source.replace(
    "logging.info(pformat(asdict(cfg)))",
    "logging.debug(pformat(asdict(cfg)))",
    1,
)
source = source.replace(
    "max_episodes_rendered = 0 if cfg.eval.recording else 10",
    "max_episodes_rendered = 0",
    1,
)
source = source.replace(
    "disable=inside_slurm()",
    "disable=True",
)

progress_state = (
    '_EVAL_PROGRESS = {"task_index": 0, "task_total": 0}'
)
if progress_state not in source:
    import_anchor = "from tqdm import trange\n"
    if import_anchor not in source:
        raise RuntimeError(
            "Evaluation progress import anchor not found."
        )
    source = source.replace(
        import_anchor,
        import_anchor + "\n" + progress_state + "\n",
        1,
    )

task_loop_anchor = (
    "        for i, (task_group, task_id, env) "
    "in enumerate(tasks):\n"
)
task_loop_patch = (
    task_loop_anchor
    + '            _EVAL_PROGRESS["task_index"] = i + 1\n'
    + '            _EVAL_PROGRESS["task_total"] = len(tasks)\n'
)
if (
    '_EVAL_PROGRESS["task_index"] = i + 1'
    not in source
):
    if task_loop_anchor not in source:
        raise RuntimeError(
            "Evaluation task-loop anchor not found."
        )
    source = source.replace(
        task_loop_anchor,
        task_loop_patch,
        1,
    )

episode_loop_anchor = "    for batch_ix in progbar:\n"
episode_progress_line = (
    '        print('
    'f"EVAL_PROGRESS '
    "task={_EVAL_PROGRESS['task_index']}/"
    "{_EVAL_PROGRESS['task_total']} "
    'episode={batch_ix + 1}/{n_batches}", '
    "flush=True)\n"
)
if "EVAL_PROGRESS task=" not in source:
    if episode_loop_anchor not in source:
        raise RuntimeError(
            "Evaluation episode-loop anchor not found."
        )
    source = source.replace(
        episode_loop_anchor,
        episode_loop_anchor + episode_progress_line,
        1,
    )

eval_script.write_text(
    source,
    encoding="utf-8",
)

libero_plus_path = str(LIBERO_PLUS_DIR)
sys.path = [
    item
    for item in sys.path
    if item != libero_plus_path
]
sys.path.insert(0, libero_plus_path)

for module_name in list(sys.modules):
    if (
        module_name == "libero"
        or module_name.startswith("libero.")
        or module_name == "robosuite"
        or module_name.startswith("robosuite.")
    ):
        del sys.modules[module_name]

importlib.invalidate_caches()

import libero
from libero.libero import benchmark

search_paths = [
    Path(path).resolve()
    for path in getattr(libero, "__path__", [])
]

if not any(
    LIBERO_PLUS_DIR.resolve() in path.parents
    or path == LIBERO_PLUS_DIR.resolve()
    for path in search_paths
):
    raise RuntimeError(
        "LIBERO-plus fork was not loaded."
    )

benchmark_path = Path(
    benchmark.__file__
).resolve()

if (
    LIBERO_PLUS_DIR.resolve()
    not in benchmark_path.parents
):
    raise RuntimeError(
        "LIBERO-plus benchmark was not loaded."
    )

print("LIBERO-plus ready.")

# ## 12. 追加学習前後を評価する
# 
# 追加学習前後の2モデルを、同じ10タスク・同じseedで評価します。
# 評価は1モデルにつき30 rollout、2モデル合計で60 rolloutです。
import json
import re
from collections import deque

EVAL_CAMERA_MAPPING = {
    "agentview_image": "front",
    "robot0_eye_in_hand_image": "wrist",
}


def build_eval_command(
    policy_path: Path,
    output_dir: Path,
) -> list[str]:
    return [
        "lerobot-eval",
        f"--policy.path={policy_path}",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--env.type=libero",
        "--env.is_libero_plus=true",
        "--env.task=libero_spatial",
        (
            "--env.task_ids="
            + json.dumps(
                EVAL_TASK_IDS,
                separators=(",", ":"),
            )
        ),
        (
            "--env.camera_name_mapping="
            + json.dumps(
                EVAL_CAMERA_MAPPING,
                separators=(",", ":"),
            )
        ),
        "--env.observation_height=256",
        "--env.observation_width=256",
        "--env.control_mode=relative",
        "--env.max_parallel_tasks=1",
        "--eval.batch_size=1",
        (
            "--eval.n_episodes="
            f"{EVAL_EPISODES_PER_TASK}"
        ),
        "--eval.use_async_envs=false",
        "--eval.recording=false",
        f"--seed={EVAL_SEED}",
        f"--output_dir={output_dir}",
    ]


def run_evaluation(
    policy_path: Path,
    output_dir: Path,
    label: str,
) -> dict:
    shutil.rmtree(
        output_dir,
        ignore_errors=True,
    )

    eval_env = os.environ.copy()
    eval_env["MUJOCO_GL"] = "egl"
    eval_env["PYTHONPATH"] = (
        str(LIBERO_PLUS_DIR)
        + os.pathsep
        + str(LEROBOT_SRC)
        + os.pathsep
        + eval_env.get("PYTHONPATH", "")
    )
    eval_env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    eval_env["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    eval_env["HF_HUB_VERBOSITY"] = "error"
    eval_env["TQDM_DISABLE"] = "1"
    eval_env["PYTHONWARNINGS"] = "ignore"
    eval_env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        build_eval_command(
            policy_path,
            output_dir,
        ),
        cwd=LEROBOT_DIR,
        env=eval_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    recent_lines: deque[str] = deque(
        maxlen=120
    )

    progress_pattern = re.compile(
        r"^EVAL_PROGRESS "
        r"task=(\d+)/(\d+) "
        r"episode=(\d+)/(\d+)$"
    )

    assert process.stdout is not None

    for raw_line in process.stdout:
        line = (
            raw_line
            .replace("\r", "")
            .strip()
        )

        if not line:
            continue

        recent_lines.append(line)
        match = progress_pattern.match(line)

        if match:
            (
                task_index,
                task_total,
                episode_index,
                episode_total,
            ) = match.groups()

            print(
                f"{label:<13} | "
                f"task {task_index}/{task_total} | "
                f"episode {episode_index}/{episode_total}"
            )

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            "\n".join(recent_lines)
        )

    result_path = (
        output_dir
        / "eval_info.json"
    )

    if not result_path.is_file():
        raise FileNotFoundError(
            result_path
        )

    return json.loads(
        result_path.read_text(
            encoding="utf-8"
        )
    )


BASE_EVAL_INFO = run_evaluation(
    BASELINE_MODEL_DIR,
    BASE_EVAL_DIR,
    "Base model",
)

FINETUNED_EVAL_INFO = run_evaluation(
    MERGED_MODEL_DIR,
    FINETUNED_EVAL_DIR,
    "Spatial LoRA",
)

print("Evaluation complete.")

# ## 13. 成功率を比較する
# 
# `Δ (pp)`は、追加学習後から追加学習前を引いた成功率差です。
import pandas as pd
from IPython.display import display


def per_task_success(
    eval_info: dict,
) -> dict[int, float]:
    result: dict[int, float] = {}

    for task_info in eval_info["per_task"]:
        task_id = int(task_info["task_id"])
        successes = task_info["metrics"]["successes"]
        result[task_id] = (
            100.0
            * sum(bool(value) for value in successes)
            / len(successes)
        )

    return result


base_per_task = per_task_success(BASE_EVAL_INFO)
finetuned_per_task = per_task_success(
    FINETUNED_EVAL_INFO
)

rows = []

for task_id in EVAL_TASK_IDS:
    base_score = base_per_task[task_id]
    finetuned_score = finetuned_per_task[task_id]

    rows.append(
        {
            "Task ID": task_id,
            "Task": SPATIAL_TASK_NAMES[task_id],              # 【修正 F6】
            "Base (%)": base_score,
            "Spatial LoRA (%)": finetuned_score,
            "Δ (pp)": finetuned_score - base_score,
        }
    )

base_overall = float(
    BASE_EVAL_INFO["overall"]["pc_success"]
)
finetuned_overall = float(
    FINETUNED_EVAL_INFO["overall"]["pc_success"]
)

rows.append(
    {
        "Task ID": "Overall",
        "Task": "LIBERO-Spatial",
        "Base (%)": base_overall,
        "Spatial LoRA (%)": finetuned_overall,
        "Δ (pp)": finetuned_overall - base_overall,
    }
)

comparison_df = pd.DataFrame(rows)
comparison_df.to_csv(
    COMPARISON_CSV_PATH,
    index=False,
)

display(comparison_df.round(1))

print(
    f"Overall: {base_overall:.1f}% → "
    f"{finetuned_overall:.1f}% "
    f"({finetuned_overall - base_overall:+.1f} pp)"
)

# ## 14. 学習済みモデルと比較結果をダウンロードする
from zipfile import ZIP_STORED, ZipFile
try:
    from google.colab import files as _colab_files
    _IS_COLAB = True
except Exception:
    _IS_COLAB = False


def push_outputs_to_github(
    paths: list[Path],
    github_remote: str | None = None,
    token_env: str = "GITHUB_TOKEN",
    branch: str = "main",
):
    """Copy small output files into a temporary git repo and push to GitHub.

    Skips files larger than 90 MiB to avoid GitHub size limits; requires
    `GITHUB_TOKEN` in env for authenticated HTTPS push when needed.
    """
    if github_remote is None:
        github_remote = os.environ.get("GITHUB_REMOTE")

    if not github_remote:
        print("GITHUB_REMOTE not set; skipping push to GitHub.")
        return

    token = os.environ.get(token_env)

    # Choose a working directory that's writable both on Colab and local runs
    if Path("/content").exists():
        repo_tmp = Path("/content/gh_outputs_repo")
    else:
        repo_tmp = Path.cwd() / "gh_outputs_repo"

    shutil.rmtree(repo_tmp, ignore_errors=True)
    repo_tmp.mkdir(parents=True, exist_ok=True)

    def git(cmd: list[str]):
        return run_quiet(cmd, check=True)

    try:
        git(["git", "init"])
        # Configure a benign committer if not present in env
        git(["git", "config", "user.email", os.environ.get("GIT_AUTHOR_EMAIL", "lerobot@example.com")])
        git(["git", "config", "user.name", os.environ.get("GIT_AUTHOR_NAME", "lerobot")])

        # Copy files, skipping large blobs
        skipped: list[Path] = []
        for p in paths:
            if not p.exists():
                continue
            dst = repo_tmp / p.name
            if p.is_dir():
                for src in sorted(p.rglob("*")):
                    if src.is_file():
                        size = src.stat().st_size
                        if size > 90 * 1024 * 1024:
                            skipped.append(src)
                            continue
                        target = repo_tmp / src.relative_to(p.parent)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, target)
            else:
                size = p.stat().st_size
                if size > 90 * 1024 * 1024:
                    skipped.append(p)
                    continue
                shutil.copy2(p, dst)

        # Add remote (inject token for HTTPS if provided)
        remote = github_remote
        if token and remote.startswith("https://") and "@" not in remote:
            remote = remote.replace("https://", f"https://{token}@")

        git(["git", "remote", "add", "origin", remote])

        git(["git", "add", "."])
        from datetime import datetime

        commit_msg = f"Add training outputs: {datetime.utcnow().isoformat()}Z"
        git(["git", "commit", "-m", commit_msg])

        # Try to pull remote changes first to avoid push rejection
        try:
            git(["git", "pull", "--rebase", "origin", branch])
        except RuntimeError:
            print("Remote pull failed or not necessary; continuing to push.")

        try:
            git(["git", "push", "-u", "origin", branch])
            print("Pushed outputs to GitHub (origin/%s)." % branch)
        except RuntimeError as e:
            print("Push failed:", e)
            print("Skipped files:", [str(x) for x in skipped])

    except Exception as exc:
        print("Failed to push outputs to GitHub:", exc)


def save_outputs_locally(paths: list[Path], dest_dir: Path):
    """Copy specified output files into `dest_dir`.

    Keeps file names and overwrites existing files with the same name.
    """
    try:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        for p in paths:
            if not p.exists():
                print(f"Skipping missing path: {p}")
                continue

            target = dest_dir / p.name
            if p.is_dir():
                # If a directory is provided, create a zip of it instead
                archive = dest_dir / (p.name + ".zip")
                if archive.exists():
                    archive.unlink()
                with ZipFile(archive, mode="w", compression=ZIP_STORED, allowZip64=True) as zf:
                    for src in sorted(p.rglob("*")):
                        if src.is_file():
                            zf.write(src, arcname=src.relative_to(p))
                print(f"Saved directory as zip: {archive}")
            else:
                shutil.copy2(p, target)
                print(f"Saved: {target}")

    except Exception as exc:
        print("Failed to save outputs locally:", exc)


if MERGED_ZIP_PATH.exists():
    MERGED_ZIP_PATH.unlink()

with ZipFile(
    MERGED_ZIP_PATH,
    mode="w",
    compression=ZIP_STORED,
    allowZip64=True,
) as archive:
    for file_path in sorted(
        MERGED_MODEL_DIR.rglob("*")
    ):
        if file_path.is_file():
            archive.write(
                file_path,
                arcname=(
                    Path(MERGED_MODEL_DIR.name)
                    / file_path.relative_to(
                        MERGED_MODEL_DIR
                    )
                ),
            )

print(f"Saved: {MERGED_ZIP_PATH}")
# 現行の files.download 呼び出しを保護
if _IS_COLAB:
    _colab_files.download(str(MERGED_ZIP_PATH))
    _colab_files.download(str(COMPARISON_CSV_PATH))
else:
    print(f"[手元環境] Colab download をスキップ。ローカル保存済み: {MERGED_ZIP_PATH}")

# === Save outputs locally ===
# User-requested path for saving merged outputs
AUTO_SAVE_DIR = Path(r"C:\Users\J48359\Documents\parc2026_submission\auto_save_mergefile")
try:
    save_outputs_locally([MERGED_ZIP_PATH, COMPARISON_CSV_PATH], AUTO_SAVE_DIR)
except Exception as e:
    print("save_outputs_locally raised:", e)

# If running in Google Colab, also copy outputs to a persistent folder under /content
if Path("/content").exists():
    colab_save = Path("/content/parc2026_submission/auto_save_mergefile")
    try:
        colab_save.mkdir(parents=True, exist_ok=True)
        for p in [MERGED_ZIP_PATH, COMPARISON_CSV_PATH]:
            if p.exists():
                shutil.copy2(p, colab_save / p.name)
                print(f"Saved to Colab: {colab_save / p.name}")
    except Exception as e:
        print("Failed to save outputs to Colab:", e)
