# E. Железо: что влезает в Quadro M2000M 4 ГБ (Maxwell sm_50), Windows

Дата сбора: 2026-09-16. Все числа — либо со ссылкой, либо с показанным расчётом.
Расчётные скрипты воспроизводимы (модель памяти и времени описана в §3.1 и §6.1).

---

## 0. Паспорт карты — из чего считаем

| параметр | значение | источник |
|---|---|---|
| Чип | GM107GLM, Maxwell 1-го поколения, 28 нм | [Notebookcheck](https://www.notebookcheck.net/NVIDIA-Quadro-M2000M.151581.0.html) |
| Compute capability | **5.0 (sm_50)** — не 5.2! GM107 = 5.0, 5.2 это GM204/206 | [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus) |
| CUDA-ядер | 640 (5 SM × 128) | [videocardz.net](https://videocardz.net/nvidia-quadro-m2000m) |
| Частота | база 1029 МГц, буст 1098 МГц | там же |
| fp32 пик | 640 × 2 × 1,098 ГГц = **1,405 TFLOPS** = 702,7 GMAC/с | расчёт |
| Память | 4096 МБ GDDR5, 128 бит | там же |
| ПСП | **80,2 ГБ/с** | там же |
| TDP | 55 Вт (мобильная — троттлит в длинных прогонах) | там же |
| Дата выпуска | 3 декабря 2015 | там же |

Две производные величины, которые дальше решают всё:

- **Точка баланса roofline** = 702,7 GMAC/с ÷ 80,2 ГБ/с = **8,76 MAC/байт**.
  Всё, что ниже 8,76 — упирается в память, а не в арифметику.
- **Реально доступная видеопамять**: 4096 МБ физически, минус WDDM-композитор
  и дисплей (~250–350 МБ на Windows), минус CUDA-контекст + библиотеки cuDNN/cuBLAS
  (~250–300 МБ, вне учёта аллокатора PyTorch) → **~3450–3600 МБ под тензоры**.

---

## 1. Версии софта, которые ОБЯЗАНЫ быть

### 1.1 Главный факт: sm_50 держится только в колёсах cu118 / cu124 / cu126

PyTorch собирает бинарные колёса под фиксированный список архитектур
(`TORCH_CUDA_ARCH_LIST`), и он **разный для Linux и Windows**. Это видно прямо
в CI-скриптах репозитория:

| канал | Windows `TORCH_CUDA_ARCH_LIST` | sm_50? | источник |
|---|---|---|---|
| cu118 (torch 2.0–2.7.1) | `3.7+PTX;5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0` | **да** | [`.ci/pytorch/windows/cuda118.bat` @ v2.7.1](https://github.com/pytorch/pytorch/blob/v2.7.1/.ci/pytorch/windows/cuda118.bat) |
| cu121 (2.1–2.5.1) | `5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0` | да | [`cuda121.bat`](https://github.com/pytorch/pytorch/blob/v2.5.1/.ci/pytorch/windows/cuda121.bat) |
| cu124 (2.4–2.6) | `5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0` | да | [`cuda124.bat` @ v2.6.0](https://github.com/pytorch/pytorch/blob/v2.6.0/.ci/pytorch/windows/cuda124.bat) |
| cu126 @ 2.6.0–2.7.1 | `5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0` | да | [`cuda126.bat` @ v2.7.1](https://github.com/pytorch/pytorch/blob/v2.7.1/.ci/pytorch/windows/cuda126.bat) |
| **cu126 @ 2.8.0** | `6.1;7.0;7.5;8.0;8.6;9.0` | **НЕТ — регрессия** | [`cuda126.bat` @ v2.8.0](https://github.com/pytorch/pytorch/blob/v2.8.0/.ci/pytorch/windows/cuda126.bat) |
| cu126 @ 2.9.0 … 2.14.0 | `5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0` | **да (починено)** | [`cuda126.bat` @ v2.9.0](https://github.com/pytorch/pytorch/blob/v2.9.0/.ci/pytorch/windows/cuda126.bat) |
| cu128 @ 2.7.1 | `5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0;10.0;12.0` | да, но CUDA 12.8 Maxwell уже не поддерживает | см. ниже |
| cu128 @ 2.8.0+ | `6.1;7.0;…` / `7.0;7.5;…` | нет | [`cuda128.bat` @ v2.9.0](https://github.com/pytorch/pytorch/blob/v2.9.0/.ci/pytorch/windows/cuda128.bat) |
| cu129 / cu130 / cu132 | `7.0;7.5;…` / `7.5;8.0;…` | нет | там же |

**Хронология поломки и починки — по PR, а не по слухам:**

- **PR [#152069](https://github.com/pytorch/pytorch/pull/152069)** «Drop legacy CUDA
  support to slim down the wheels» — вырезал `5.0;6.0` из Windows-скриптов
  `cuda124.bat / cuda126.bat / cuda128.bat`. Мотив автора дословно: колесо выросло
  «from ~2.5GB with Pytorch 2.6.0 all the way to ~3.1GB with pytorch 2.7.0 CUDA 12.8».
  Явно перечислены жертвы: «GPUs like the GTX960M, 950M, 940M, 930M and some other
  Quadro GPUs all the way from april 2016 like Quadro M500M».
- Итог: **torch 2.8.0 + cu126 на Windows НЕ запускается на sm_50/sm_52** —
  issue [#160575](https://github.com/pytorch/pytorch/issues/160575),
  `torch.cuda.get_arch_list()` вернул `['sm_61','sm_70','sm_75','sm_80','sm_86','sm_90']`.
  Комментарий мейнтейнера atalman 2025-08-13: *«Please note this affects only Windows
  builds. Linux binaries do build with following sm support: `5.0;6.0;7.0;7.5;8.0;8.6;9.0`»*.
- **PR [#160586](https://github.com/pytorch/pytorch/pull/160586)** вернул
  `5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0` в `cuda126.bat`. atalman 2025-10-22:
  *«Closing since addressed in release 2.9.0»*.
- Для CUDA 12.8/12.9 возврата не было и не будет: NVIDIA убрала Maxwell/Pascal
  из самого тулкита — [dev-discuss, atalman, 14.07.2025](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128),
  матрица — issue [#157517](https://github.com/pytorch/pytorch/issues/157517).

### 1.2 Дедлайн: PyTorch 2.14 — последняя версия с sm_50

- RFC [#190385](https://github.com/pytorch/pytorch/issues/190385) (17.07.2026) и
  [объявление в dev-discuss](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432):
  в релизе 2.15 CUDA 12.6 выводится из CI/CD, остаются только CUDA 13.0/13.2/13.4,
  а CUDA 13.x **не умеет компилировать** sm_50/sm_60/sm_70.
  Дословная рекомендация: *«users on these GPUs should remain on PyTorch 2.14
  (CUDA 12.6) or earlier»*.
- Проверено по индексу колёс на 2026-09-16
  (`https://download.pytorch.org/whl/cu126/torch/`, фильтр `win_amd64`):
  cu126 для Windows выпущен для **2.6.0 … 2.14.0**; `torch 2.14.0` вышел 2026-09-02
  (PyPI), это текущий последний релиз.

### 1.2b ВТОРОЙ, НЕЗАВИСИМЫЙ ОБРЫВ: cuDNN 9.11 выкинул SM < 7.5

Мало иметь `sm_50` в `get_arch_list()` — свёртки считает cuDNN, а он живёт своим
календарём. **cuDNN 9.11.0 убрал поддержку GPU с compute capability < 7.5**
(Maxwell, Pascal, Volta): «GPU architectures earlier than the NVIDIA Turing
architecture … are no longer supported in cuDNN 9.11.0»
([cuDNN Backend Release Notes](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html)).
Симптом — не внятная ошибка, а «odd error messages that do not really give a clue»
(issue [pytorch#162574](https://github.com/pytorch/pytorch/issues/162574),
рекомендация автора дословно: *«Install a PyTorch build that bundles cuDNN ≤ 9.10.x»*).

Хорошая новость: **PyTorch специально держит cu126 как legacy-канал и пиннит там
старый cuDNN.** Проверено по CI-скрипту Windows-сборки
[`.ci/pytorch/windows/internal/cuda_install.bat` @ v2.14.0](https://github.com/pytorch/pytorch/blob/v2.14.0/.ci/pytorch/windows/internal/cuda_install.bat#L163-L185):

```bat
if %CUDA_VER% EQU 126 ( set CUDNN_FOLDER=cudnn-windows-x86_64-9.10.2.21_cuda12-archive
                        set EXPECTED_CUDNN_VERSION=9.10.2 )
if %CUDA_VER% EQU 128 ( set CUDNN_FOLDER=cudnn-windows-x86_64-9.24.0.43_cuda12-archive
                        set EXPECTED_CUDNN_VERSION=9.24.0 )
if %CUDA_VER% EQU 130 ( ... 9.24.0 )
if %CUDA_VER% EQU 132 ( ... 9.24.0 )
```

**9.10.2 < 9.11.0 → Maxwell жив.** Таблица по версиям (Windows, канал cu126):

| torch | cuDNN в Windows-колесе cu126 | SM < 7.5? |
|---|---|---|
| 2.7.1 | 9.5.0.50 | да |
| 2.9.0 / 2.9.1 | 9.7.0.66 | да |
| 2.10.0 | 9.7.0.66 | да |
| 2.11.0 … **2.14.0** | **9.10.2.21** (пиннится явно) | **да** |
| любой cu128/129/130/132 | 9.19 / 9.20 / **9.24.0.43** | **нет** |

И финальное подтверждение, что в 2.14.0 sm_50 в cu126 действительно есть — уже из
унифицированной таблицы сборки
[`.ci/manywheel/build_env_setup.py` @ v2.14.0](https://github.com/pytorch/pytorch/blob/v2.14.0/.ci/manywheel/build_env_setup.py#L72-L89):

```python
TORCH_CUDA_ARCH_LIST_TABLE = {
    "12.6": {"x86_64": {50, 60, 70, 75, 80, 86, 90}, "aarch64": {80, 90}},
    "13.0": {"x86_64": {75, 80, 86, 90, 100, 120}, ...},
    "13.2": {...}, "13.4": {...},
}
```

Комментарий в том же файле объясняет, почему возврата не будет:
*«CUDA 13.x dropped sm_50/60/70 … nvcc 13 rejects with "Unsupported gpu
architecture 'compute_50'"»*.

**Вывод: канал cu126 — единственный, где сходятся оба условия (SASS для sm_50
И cuDNN < 9.11). Он держится до torch 2.14.0 включительно и исчезает в 2.15.**

### 1.3 Практический вывод по версиям

**Ставить одно из двух, ничего больше:**

| вариант | команда | sm_50 | cuDNN | комментарий |
|---|---|---|---|---|
| **A — рекомендую** | `pip install torch==2.9.1+cu126 torchvision==0.24.1+cu126 --index-url https://download.pytorch.org/whl/cu126` | да | 9.7.0 | свежий API (`torch.amp`, SDPA, `torch.compile`), обкатан год, Python 3.10–3.14 |
| B — потолок | `torch==2.14.0+cu126 torchvision==0.29.0+cu126` | да | 9.10.2 | последний релиз с sm_50 вообще; берите, если библиотека требует torch ≥ 2.11 |
| C — запасной | `pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118` | да | 9.5.0 | CUDA 11.8 — здесь же живёт единственная сборка onnxruntime-gpu под sm_50 (§8.2) |

**Категорически НЕ ставить:** `2.8.0+cu126` (регрессия sm_50 на Windows),
любые `+cu128 / +cu129 / +cu130 / +cu132` (Maxwell выпилен из тулкита),
`torch ≥ 2.15` (когда выйдет — там не будет cu12x вообще).

Проверка после установки — одна строка, делать её обязательно перед любым обучением:

```python
import torch; print(torch.__version__, torch.cuda.get_arch_list(),
                    torch.cuda.get_device_capability(0))
# ожидаем: 'sm_50' в списке и (5, 0)
```

Если `sm_50` в списке нет, симптом будет такой:
`GPU with CUDA capability sm_50 is not compatible with the current PyTorch installation`
(текст из issue [#160575](https://github.com/pytorch/pytorch/issues/160575)) —
либо `CUDA error: no kernel image is available for execution on the device`.

### 1.4 Драйвер — отдельная бомба замедленного действия

- Ветка **R580 — последняя, поддерживающая Maxwell / Pascal / Volta**
  ([Phoronix](https://www.phoronix.com/news/NVIDIA-580-Linux-Driver-Last-HW),
  [TechPowerUp](https://www.techpowerup.com/338497/nvidias-v580-driver-branch-ends-support-for-maxwell-pascal-and-volta-gpus)).
  Для Quadro: обновления ветки 580 — до конца июля 2026, дальше только security-fix.
- Следствие: драйвер на узле **обновлять нельзя**. Зафиксировать текущий, записать
  его версию в журнал проекта. Любое «обновим драйвер до последнего» = потеря CUDA.
- CUDA 12.6 runtime при minor-version-compatibility требует драйвер ≥ 527.41 (Windows);
  ветка 580 это покрывает с запасом.

### 1.5 Что из библиотек может не завестись

| библиотека | требование | вердикт на torch 2.9.1 |
|---|---|---|
| `torchvision` | строгая пара к torch | 0.24.1 ↔ 2.9.1 — ок |
| `timm` | ≥ 0.9 требует torch ≥ 1.13 | ок; даёт `set_grad_checkpointing()` из коробки (§4.2) |
| `transformers` (для CLIP/DINOv2) | свежие требуют torch ≥ 2.1 | ок по версии, но модели не влезут по памяти (§8) |
| `torchreid` (deep-person-reid) | чистый torch, без CUDA-расширений | ок |
| `fast-reid` | `apex` опционально, `faiss` для eval | ок без apex; apex на sm_50 собирать не надо |
| `bitsandbytes` (8-bit Adam) | **Compute Capability 6.0+** | **не встанет на sm_50** (§4.6) |
| `flash-attn` | sm_80+ | **не встанет**, и не нужен |
| `xformers` | собирается под sm_70+ в бинарях | не нужен, в torch есть mem-efficient SDPA от sm_50 |

---

## 2. fp16 / AMP на Maxwell: памяти — да, скорости — нет

### 2.1 Аппаратной fp16-арифметики на sm_50 НЕТ — это записано в таблице NVIDIA

Таблица «Throughput of Native Arithmetic Instructions (Number of Results per Clock
Cycle per Multiprocessor)» из CUDA C++ Programming Guide, §8.4.1 Arithmetic Instructions
([online](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions),
исторический срез с теми же числами — [CUDA 9.1 PG, стр. 85](https://docs.nvidia.com/cuda/archive/9.1/pdf/CUDA_C_Programming_Guide.pdf)):

| Compute capability | 3.5/3.7 | **5.0 / 5.2** | 5.3 | 6.0 | 6.1 | 6.2 | 7.0 | 8.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **16-bit FP** add / mul / FMA | N/A | **N/A** | 256 | 128 | 2 | 256 | 128 | 256 |
| 32-bit FP add / mul / FMA | 192 | **128** | 128 | 64 | 128 | 128 | 64 | 128 |
| 64-bit FP add / mul / FMA | 64 | **4** | 4 | 32 | 4 | 4 | 32 | 32 |

**У 5.0 в строке fp16 стоит `N/A` — блоков fp16-арифметики в кремнии нет вообще.**
Нативный fp16 появляется с 5.3 (Tegra X1, 2× от fp32) и 6.0 (P100, 2×);
на 6.1 (GTX 10xx) он специально кастрирован до 2/такт = 1/64 от fp32.

Порог для интринсиков `__hadd2`/`__hfma2` — `__CUDA_ARCH__ >= 530`; ниже (то есть
и на sm_50) компилятор подставляет эмуляцию, дословно из
[NVIDIA blog «Mixed-Precision Programming with CUDA 8»](https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/):

```cuda
y[i] = __float2half(__half2float(a) * __half2float(x[i]) + __half2float(y[i]));
```

То есть на sm_50 каждая fp16-операция стоит **дороже** fp32: две конвертации плюс
сама операция. Конвертационные интринсики есть на всех архитектурах — поэтому
fp16 **как формат хранения** работает нормально, а как формат вычислений — нет.

**cuDNN** различает два режима:
- `PSEUDO_HALF_CONFIG` — тензоры `CUDNN_DATA_HALF`, счёт в `CUDNN_DATA_FLOAT`. **Работает на sm_50** — это то, что реально включает AMP.
- `TRUE_HALF_CONFIG` — и данные, и счёт в fp16. **Требует cc ≥ 5.3.**
  Дословно из [cuDNN 7.6.4 Release Notes](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-880/release-notes/rel_7xx.html):
  *«will now return CUDNN_STATUS_ARCH_MISMATCH for true-half configuration on devices
  with compute capability less than 5.3 … which do not have native hardware support
  for true half computation»*.

**Tensor Cores** — с Volta (sm_70). **TF32** — только Ampere+ и это не формат хранения,
а усечение мантиссы внутри Tensor Core
([PyTorch CUDA notes](https://docs.pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)).
**bfloat16** аппаратно — с sm_80; `torch.cuda.is_bf16_supported(including_emulation=False)`
на sm_50 вернёт `False`, а `autocast("cuda", dtype=torch.bfloat16)` бросит
`RuntimeError: Current CUDA Device does not support bfloat16. Please switch dtype to float16.`
([`torch/cuda/__init__.py`](https://github.com/pytorch/pytorch/blob/main/torch/cuda/__init__.py)).

### 2.2 Скорость от AMP: измерено, что МЕДЛЕННЕЕ

Замер на Tesla M60 (GM204, sm_52 — то же поколение, что наша карта), BERT-бенчмарк
([discuss.pytorch.org](https://discuss.pytorch.org/t/amp-autocast-not-faster-than-fp32/111757)):

| режим | время шага |
|---|---:|
| FP32 | **79,00 мс** |
| FP16 autocast | **92,23 мс** |

**AMP на 17 % медленнее.** Ответ мейнтейнера ptrblck там же: *«I wouldn't expect to
see a large difference between AMP and FP32 on Maxwell GPUs»*.

Официальная дока PyTorch не обещает иного:
*«Mixed precision primarily benefits Tensor Core-enabled architectures (Volta, Turing,
Ampere). This recipe should show significant (2-3X) speedup on those architectures.
On earlier architectures (Kepler, Maxwell, Pascal), you may observe a modest speedup»*
([AMP recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)).

### 2.3 Память от AMP: выигрыш есть, но НЕ гарантирован

Первоисточник — Micikevicius et al., «Mixed Precision Training», ICLR 2018
([arXiv:1710.03740](https://arxiv.org/abs/1710.03740)), §3.1, дословно:

> *«Even though maintaining an additional copy of weights increases the memory
> requirements for the weights by **50 %** compared with single precision training,
> impact on overall memory usage is much smaller. For training memory consumption is
> dominated by activations … Since activations are also stored in half-precision
> format, the overall memory consumption for training deep neural networks is
> **roughly halved**.»*

Бухгалтерия для нашей конфигурации (числа из §3.2):

| статья | fp32 | AMP fp16 | дельта |
|---|---:|---:|---:|
| веса (fp32 master + fp16 копия) | 98,8 | 148,2 | **+49,4** |
| градиенты | 98,8 | 98,8 (unscale в fp32) | 0 |
| Adam | 197,7 | 197,7 | 0 |
| **активации** | **2276,9** | **~1400–1550** | **−730…−880** |
| транзиент | 341,5 | ~220 | −120 |
| **ИТОГО** | **3133,8** | **~2185–2335** | **−800…−950 МБ** |

Оценка активаций: в fp16 уходят выходы conv (`PSEUDO_HALF_CONFIG`), это ~2/3 объёма;
входы нормализаций и сами BN остаются fp32 по autocast-политике
([op reference](https://docs.pytorch.org/docs/stable/amp.html#cuda-op-specific-behavior)).

**Контрпример, который обязательно надо проверить у себя.** Замер на RTX 6000
([discuss.pytorch.org](https://discuss.pytorch.org/t/increased-memory-usage-with-amp/125486)):

| | `max_memory_allocated` | `max_memory_reserved` |
|---|---:|---:|
| FP32 | 3176 МБ | 3454 МБ |
| AMP | **3520 МБ (+11 %)** | 3646 МБ |

Причины роста: кэш fp16-кастов весов внутри autocast-региона и фрагментация
аллокатора. ptrblck там же: *«model parameters are still kept in fp32 … The reduction
comes from not keeping the activations in fp32»*.

### 2.4 Как этим пользоваться

AMP на M2000M — это **бюджет памяти, а не ускорение**. Включать имеет смысл только
чтобы поднять батч (32 → 48–64), потому что для triplet-лосса число ID в батче
важнее скорости (§4.3). Порядок действий:

1. замерить `torch.cuda.max_memory_allocated()` на 20 шагах без AMP и с AMP;
2. если выигрыш < 500 МБ — выключить AMP и брать gradient checkpointing,
   он на Maxwell предсказуемее;
3. если выигрыш есть — поднять батч и **сравнивать время эпохи**, а не время шага.

## 3. Сколько памяти реально нужно — расчёт

### 3.1 Модель расчёта (не догадка)

Пиковая память = **параметры + градиенты + состояние оптимизатора + активации +
транзиент backward + workspace cuDNN**.

- **Параметры / градиенты** = `N × 4 Б` каждое.
- **Adam** хранит `exp_avg` и `exp_avg_sq` = `2 × N × 4 Б`
  ([`torch.optim.Adam`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)).
  SGD+momentum = `1 × N × 4 Б`, SGD без momentum = 0.
- **Активации** считаем не «на глазок», а перечислением тензоров, которые autograd
  обязан сохранить до backward. Правила взяты из `tools/autograd/derivatives.yaml`:

  | операция | что сохраняет |
  |---|---|
  | `convolution` | **вход** (нужен для `grad_weight`) |
  | `native_batch_norm` / `instance_norm` | **вход** + `save_mean`, `save_invstd` |
  | `relu(inplace=True)` | **результат** (тот же буфер, что вход → нового не создаёт) |
  | `relu(inplace=False)` | **результат** (вход при этом освобождается!) |
  | `add`, `cat`, `split`, `.contiguous()` | **ничего** |
  | `mul` | **оба операнда** |
  | `sigmoid`, `softmax` | результат |
  | `max_pool2d_with_indices` | вход + `indices` (int64 = 2× fp32) |
  | `linear`, `layer_norm`, `gelu` | вход |
  | `bmm` / `matmul` | оба операнда |

  Тензор занимает память до backward **только если его кто-то сохранил**.
  Для цепочки `conv → bn → relu` живут ровно два буфера: выход conv (его держит BN)
  и выход BN (его держит ReLU). Это даёт **×2 от суммы карт признаков** — ровно
  то, что даёт прямой подсчёт карт ResNet-50.
- **Транзиент backward** (градиенты активаций, живущие короткими окнами) — 15 % от
  активаций, эмпирическая надбавка.
- **Workspace cuDNN** — 120 МБ (типично для `IMPLICIT_PRECOMP_GEMM` при
  `cudnn.benchmark=True`).

**Валидация модели на трёх независимых точках:**

1. Параметры backbone ResNet-50 расчётом = **23 508 032**; с `fc(2048→1000)` =
   **25 557 032** — в точности каноническое число torchvision.
2. Прямые MAC ResNet-50 @224 расчётом = **4,09 GMAC/изобр.** — каноническое 4,1.
3. Параметры OSNet-AIN x1.0 расчётом (с головой на 1171 класс) = 2,79 M;
   минус классификатор `512×1171+1171` = **2,19 M** — в точности число из брифа
   и из torchreid (2 193 616).
4. MAC OSNet-AIN расчётом при 256×128 = **0,978 GMAC** — в точности значение
   из [torchreid MODEL_ZOO](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html) (0,978 GFLOPs).
5. ViT-B/16 @224 расчётом = **17,56 GMAC** — каноническое 17,6.

### 3.2 ТАБЛИЦА ПАМЯТИ (fp32, Adam, голова на 1171 класс)

Все значения в МБ. «ИТОГО» — то, что увидит аллокатор PyTorch;
бюджет карты **~3450–3600 МБ**.

| конфигурация | параметры, M | веса | градиенты | Adam | активации | транзиент | workspace | **ИТОГО, МБ** | влезает? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| **R50 @208 B=32 Adam** *(текущая)* | 25,91 | 98,8 | 98,8 | 197,7 | 2276,9 | 341,5 | 120 | **3133,8** | да, запас ~350 МБ |
| R50 @208 B=32 **SGD+momentum** | 25,91 | 98,8 | 98,8 | 98,8 | 2276,9 | 341,5 | 120 | **3035,0** | да, +99 МБ запаса |
| R50 @208 B=16 Adam | 25,91 | 98,8 | 98,8 | 197,7 | 1138,5 | 170,8 | 120 | **1824,6** | да, вольготно |
| R50 @208 B=48 Adam | 25,91 | 98,8 | 98,8 | 197,7 | 3415,4 | 512,3 | 120 | **4443,0** | **нет** |
| R50 @224 B=32 Adam | 25,91 | 98,8 | 98,8 | 197,7 | 2622,0 | 393,3 | 120 | **3530,7** | на грани |
| R50 @256 B=32 Adam | 25,91 | 98,8 | 98,8 | 197,7 | 3424,5 | 513,7 | 120 | **4453,6** | **нет** |
| R50 @256 B=24 Adam | 25,91 | 98,8 | 98,8 | 197,7 | 2568,4 | 385,3 | 120 | **3469,0** | на грани |
| R50 @256 B=16 Adam | 25,91 | 98,8 | 98,8 | 197,7 | 1712,2 | 256,8 | 120 | **2484,5** | да |
| **R50-IBN-a @208 B=32 Adam** | 25,91 | 98,8 | 98,8 | 197,7 | 2276,9 | 341,5 | 120 | **3133,8** | **да — столько же, сколько R50** |
| R50-IBN-a @208 B=24 Adam | 25,91 | 98,8 | 98,8 | 197,7 | 1707,7 | 256,2 | 120 | **2479,2** | да |
| R50-IBN-a @256 B=32 Adam | 25,91 | 98,8 | 98,8 | 197,7 | 3424,5 | 513,7 | 120 | **4453,6** | **нет** |
| R50-IBN-a @256 B=16 Adam | 25,91 | 98,8 | 98,8 | 197,7 | 1712,2 | 256,8 | 120 | **2484,5** | да |
| **OSNet-AIN @208 B=32 Adam** | 2,79 | 10,7 | 10,7 | 21,3 | **3256,7** | 488,5 | 120 | **3907,8** | **нет!** |
| OSNet-AIN @208 B=24 Adam | 2,79 | 10,7 | 10,7 | 21,3 | 2442,5 | 366,4 | 120 | **2971,6** | да |
| OSNet-AIN @256×128 B=32 | 2,79 | 10,7 | 10,7 | 21,3 | 2466,8 | 370,0 | 120 | **2999,4** | да |
| OSNet-AIN @256×128 B=64 | 2,79 | 10,7 | 10,7 | 21,3 | 4933,5 | 740,0 | 120 | **5836,2** | нет |
| ViT-B/16 @224 B=32 Adam | 86,70 | 330,7 | 330,7 | 661,5 | 4283,5 | 642,5 | 120 | **6368,9** | **нет** |
| ViT-B/16 @224 B=16 Adam | 86,70 | 330,7 | 330,7 | 661,5 | 2141,8 | 321,3 | 120 | **3905,9** | **нет** |
| ViT-B/16 @224 B=8 Adam | 86,70 | 330,7 | 330,7 | 661,5 | 1070,9 | 160,6 | 120 | **2674,4** | влезет, но бесполезно (§8) |

### 3.3 Три вывода из таблицы, которые меняют планы

**(1) IBN-a бесплатен по памяти.** Это неочевидно и это проверено по коду.
`IBN.forward` ([XingangPan/IBN-Net `ibnnet/modules.py`](https://github.com/XingangPan/IBN-Net/blob/master/ibnnet/modules.py)):

```python
split = torch.split(x, self.half, 1)
out1 = self.IN(split[0].contiguous())
out2 = self.BN(split[1].contiguous())
out = torch.cat((out1, out2), 1)
```

`torch.split` и `.contiguous()` **не сохраняют вход** для backward, поэтому выход
conv1 освобождается сразу; вместо него живут две contiguous-копии (в сумме = одна
полная карта) и результат `cat` (ещё одна). Итого **2 карты — ровно столько же,
сколько у обычной цепочки conv→BN→ReLU**. Параметров тоже поровну: IBN заменяет
`BatchNorm2d(planes)` на `IN(planes/2) + BN(planes/2)`, аффинных параметров те же `2×planes`.
Транзиентная надбавка (момент, когда живы и копии, и `out1/out2`, и `cat`) — одна
карта самого крупного IBN-слоя: 32×64×52×52×4 Б = **22,1 МБ**. Шум.

→ **Переход R50 → R50-IBN-a при батче 32 и 208×208 не требует ничего менять.**

**(2) OSNet при батче 32 и 208×208 НЕ ВЛЕЗЕТ,** хотя весит в 12 раз меньше.
3908 МБ против бюджета 3450–3600. Причина — архитектура:
- финальная карта у OSNet на /16 (13×13 при 208), у ResNet-50 на /32 (7×7);
- 4 параллельных потока × глубины 1..4 = 10 `LightConv3x3` на блок, каждый
  из pointwise + depthwise + BN + ReLU;
- `ChannelGate` делает `input * gate` — `mul` сохраняет **оба операнда**;
- **471 сохранённый тензор против 112 у ResNet-50**.

Лечится тривиально: **батч 24** (2972 МБ) или **вход 256×128** вместо 208×208
(2999 МБ при батче 32). Второе ещё и ближе к тому, на чём OSNet обучался.

**(3) Adam стоит ровно 197,7 МБ, и это подтверждается размерами ваших файлов.**
`ckpt.pt / final.pt = 308 885 785 / 103 142 354 = 2,9948 ≈ 3`. Разность
`ckpt − final = 205 743 431 Б = 8·T` → `T = 25 717 929` обучаемых параметров,
`4T = 102 871 716 Б`; остаток `final − 4T = 270 638 Б` — это буферы BN
(53 слоя, 26 560 каналов: `2×26 560×4 + 53×8 = 212 904 Б`) плюс ~58 кБ заголовков zip.
Расчёт сходится до 0,3 %.

---

## 4. Приёмы экономии памяти и их цена

### 4.1 Сводка: сколько даёт и сколько стоит

Отсортировано по отношению «выигрыш / риск». База: R50 @208 B=32 Adam = 3133,8 МБ.

| приём | экономия | цена | вердикт |
|---|---:|---|---|
| **`Adam(..., foreach=False)`** | **~98,8 МБ** | ~0 | **включить первым — бесплатно** |
| `cudnn.benchmark=True` + `drop_last=True` | — | −(долгий 1-й шаг) | **включить**, даёт +5…40 % скорости |
| **Gradient checkpointing** `layer3+layer4` | **~1400 МБ** (активации 2277 → ~850) | **+25…30 % времени** | **главный рычаг** |
| **AMP fp16** (§2) | ~800–950 МБ, но не гарантированно | 0…−17 % скорости | второй по силе, обязательно мерить |
| Заморозка `conv1+bn1+layer1` (+`.eval()` на их BN) | ~350–450 МБ активаций + 3,5 МБ весов/Adam | риск недообучения низких слоёв | рабочий, если префикс **непрерывный от входа** |
| SGD+momentum вместо Adam | 98,8 МБ | вся сетка lr из BoT — под Adam | не стоит того |
| `torch.optim.Adafactor` (в ядре PyTorch) | ~190 МБ (состояние O(R+C) вместо O(R·C)) | другой режим сходимости | запасной |
| Батч 32 → 24 | 569 МБ активаций | **−4,5 п.п. mAP** (§4.3) | последний рычаг |
| `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` | борется с фрагментацией | небольшое замедление | держать в рукаве |
| `zero_grad(set_to_none=True)` | 0 — **уже default** в современном PyTorch | — | просто не передавать `False` |
| `nn.ReLU(inplace=True)` | 0 — **уже включён** в torchvision | — | ничего не делать |
| Gradient accumulation | **0 по памяти и 0 по триплетам** | — | **для re-ID бесполезно** |
| `channels_last` | 0 | **до ×2,4 медленнее** в fp32 | **не включать** |
| `expandable_segments:True` | — | **на Windows молча игнорируется** | недоступно |
| `bitsandbytes` 8-bit Adam | ~150 МБ | **требует CC ≥ 6.0** | **не запустится на sm_50** |
| `torch.cuda.empty_cache()` в цикле | 0 для своего процесса | замедляет | только между train и eval |

### 4.2 Gradient checkpointing: числа из первоисточника

**Chen et al., «Training Deep Nets with Sublinear Memory Cost»,
[arXiv:1604.06174](https://arxiv.org/abs/1604.06174).** В статье нет таблиц с
ResNet-50/101/152 — есть три рисунка (MXNet, Titan X, batch 32, вход 3×224×224).
Дословные утверждения:

- *«we design an algorithm that costs **O(sqrt(n)) memory** to train a n layer network,
  **with only the computational cost of an extra forward pass per mini-batch**»*;
- *«reduce the memory cost of a 1,000-layer deep residual network **from 48G to 7G**
  on ImageNet problems … with only **30 percent** additional running time cost»*
  (**×6,9 по памяти за +30 % времени**);
- Fig. 5: без приёма «it is only possible to train a **200 layer** ResNet», с приёмом —
  «**1000 layer** ResNet using **less than 7GB**»;
- Fig. 7: *«sub-linear memory plan incurs **roughly 30 %** of additional runtime cost»*;
- §4.2 — что именно выбрасывать: *«We can always keep the result of convolution, but
  **drop the result of the batch normalization, activation function and pooling**»*.

**Независимый замер в PyTorch** (BERT-base, batch 64, T4/V100,
[residentmario](https://residentmario.github.io/pytorch-training-performance-guide/gradient-checkpoints.html) /
[spell.ml](https://spell.ml/blog/gradient-checkpointing-pytorch-YGypLBAAACEAefHs)):
*«reduced **peak model memory usage by about 60 %**»* при *«increasing model training
time by **about 25 %**»*.

**Перенос на нашу конфигурацию (расчёт).** Активации R50@208 B=32 = 2276,9 МБ,
из них на `layer1..layer4` — около 2100 МБ. Если чекпойнтить только `layer3+layer4`
(они дешевле по пересчёту: 6+3 боттлнека на разрешении /16 и /32), живыми остаются
входы этих блоков плюс активации одного пересчитываемого:

| вариант | активации | ИТОГО | время эпохи |
|---|---:|---:|---|
| без чекпойнтинга, B=32 | 2276,9 | 3133,8 | 245…353 с |
| чекпойнт `layer3+layer4`, B=32 | ~1450 | ~2180 | ~290…420 с (+18 %) |
| чекпойнт `layer1..layer4`, B=32 | ~850 | ~1430 | ~320…460 с (+30 %) |
| **чекпойнт `layer1..layer4`, B=64** | ~1700 | **~2600** | **320…495 с** |

**Три ловушки, которые надо знать ДО, а не ПОСЛЕ:**

1. **BatchNorm обновляется дважды — известный открытый баг.**
   [pytorch#96136](https://github.com/pytorch/pytorch/issues/96136) (с 06.03.2023,
   **до сих пор открыт**), дословно: *«if the wrapped module contains `BatchNorm`
   layers, the **batch norm will be updated twice** because the forward pass has to
   be ran twice»*. Затрагивает **и** `use_reentrant=True`, **и** `False`.
   Эффективная `momentum` у BN удваивается. Для нас это критично: у BoT-модели
   BNNeck, и его `running_mean/var` идут прямо в инференс-эмбеддинг.
   **Лечение:** уменьшить `bn.momentum` вдвое (0,1 → 0,05) **или** не оборачивать
   BNNeck и держать чекпойнт только внутри `layer3/layer4`.
2. **Только `use_reentrant=False`.** С PyTorch 2.9 отсутствие явного аргумента —
   исключение; `True` ломает `torch.autograd.grad`, не поддерживает detached-тензоры
   и требует хотя бы один вход с `requires_grad=True`
   ([дока](https://docs.pytorch.org/docs/stable/checkpoint.html)).
3. **`preserve_rng_state=True`** (по умолчанию) нужен, если внутри блока есть
   dropout или случайные аугментации; он стоит немного времени, но выключать его
   нельзя — иначе второй forward даст другой dropout-маску и **молча неверные
   градиенты** (дока прямо предупреждает про *«silently incorrect gradients»*).

**Как применять к torchvision.** Встроенной поддержки **нет**: в
[`torchvision/models/resnet.py`](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
слова `checkpoint` не встречается. Оборачивать вручную:

```python
from torch.utils.checkpoint import checkpoint
class CkptBlock(nn.Module):
    def __init__(self, block): super().__init__(); self.block = block
    def forward(self, x):     return checkpoint(self.block, x, use_reentrant=False)

for name in ("layer3", "layer4"):            # по одному bottleneck — гранулярнее
    layer = getattr(model, name)
    for i, blk in enumerate(layer): layer[i] = CkptBlock(blk)
```

В **timm** это есть из коробки: `model.set_grad_checkpointing(True)` →
`checkpoint_seq([layer1..layer4], x, flatten=True)`
([`timm/models/resnet.py`](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py)).
Если берёте ResNet-50-IBN-a из timm — просто вызовите этот метод.

### 4.3 Gradient accumulation: для re-ID это пустышка

Накопление по `k` микро-батчам даёт эффективный батч `k·B` **по статистике
градиента**, и на этом всё:

1. **Память не экономится вообще** — пик определяется микро-батчем.
2. **BatchNorm считает статистики по микро-батчу.** Открытый feature request
   [pytorch#46516](https://github.com/pytorch/pytorch/issues/46516): нормализация
   внутри forward остаётся по микро-батчу, а `running_mean/var` обновляются
   `k` раз за шаг оптимизатора.
3. **Triplet-лосс считается внутри микро-батча.** Batch Hard из Hermans et al.
   ([arXiv:1703.07737](https://arxiv.org/abs/1703.07737), формула 5): *«for each
   sample a in the batch, we can select the hardest positive and the hardest negative
   samples **within the batch**»*. Число валидных триплетов для P идентичностей
   по K снимков = `P·K·(K−1)·(P−1)·K`:

   | P × K | батч | триплетов |
   |---|---:|---:|
   | 2 × 4 | 8 | **96** |
   | 6 × 4 | 24 | 1 440 |
   | **8 × 4** | **32** | **2 688** ← сейчас |
   | **16 × 4** | **64** | **11 520** (×4,3) |
   | 32 × 4 | 128 | 47 616 |

   Накопление 2 микро-батчей по 32 даёт 2×2688 = 5376 триплетов, но **ни одного
   кросс-микробатчевого** — то есть 47 % от настоящего батча 64 и, что важнее,
   **вдвое меньше идентичностей** для поиска hard-negative.

**Сколько это стоит по метрике.** BoT (Luo et al.,
[arXiv:1903.07071](https://arxiv.org/abs/1903.07071), Табл. 5):

| P × K (батч) | Market1501 r1 / mAP | DukeMTMC r1 / mAP |
|---|---|---|
| 8×3 (24) | 92,6 / 79,2 | 84,4 / 68,1 |
| **8×4 (32)** | **92,9 / 80,0** | **84,7 / 69,4** |
| 16×3 (48) | 93,8 / 83,1 | 86,8 / 72,1 |
| **16×4 (64)** | **93,8 / 83,7** | **86,6 / 73,0** |
| 32×4 (128) | 93,2 / 82,8 | 86,5 / 73,1 |

Переход **32 → 64 даёт +3,7 mAP на Market1501 и +3,6 на DukeMTMC**; дальше (128)
уже хуже. При нашем `sd(mAP) = 0,0134` это **в 2,7 раза больше порога значимости** —
то есть единственный приём из всего §4, который покупает не мегабайты, а метрику.

**Вывод: настоящий батч 64 через gradient checkpointing стоит +25…30 % времени
и даёт ~+3,7 mAP. Виртуальный батч 64 через accumulation стоит столько же времени
и не даёт ничего.**

### 4.4 Заморозка слоёв: правило, которое обычно формулируют неверно

Экономит ли заморозка активации — **зависит от того, где заморожен участок**.
Правило от core-разработчика PyTorch (albanD,
[discuss.pytorch.org](https://discuss.pytorch.org/t/requires-grad-false-does-not-save-memory/21936)):
*«if you don't require gradients for the input, then you won't need to backprop
through conv2 and **intermediary results won't be saved**»*.

| что заморожено | активации | градиенты | состояние Adam |
|---|---|---|---|
| **непрерывный префикс от входа** (`conv1+bn1+layer1`) | **экономятся** — граф не строится | экономятся | экономится |
| блок в середине (`layer2` при обучаемом `layer1`) | **НЕ экономятся** — градиент идёт насквозь | экономятся | экономится |

Для нас: заморозка `conv1+bn1+layer1` убирает самые «толстые» по HxW активации.
Расчёт по §3.1: доля `conv1+bn1+maxpool+layer1` в 2276,9 МБ ≈ **450 МБ**;
параметров там 0,22 M → ещё 0,9 МБ градиентов + 1,8 МБ Adam.
**Итого ~453 МБ.** Обязательно `.eval()` на замороженных BN, иначе их
`running_mean/var` продолжат дрейфовать и заморозка станет фиктивной.

### 4.5 Скрытый пожиратель: `foreach=True`

По умолчанию `torch.optim.Adam` и `SGD` на CUDA используют `foreach`-реализацию,
и [дока](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)
предупреждает прямо: *«the foreach implementation uses **~ sizeof(params) more peak
memory** than the for-loop version due to the intermediates being a tensorlist vs
just one tensor»*. Для нашей модели это **~98,8 МБ пика** — ровно столько же, сколько
даёт замена Adam на SGD, но **без потери качества и без пересчёта lr**.

```python
optimizer = torch.optim.Adam(params, lr=3.5e-4, foreach=False)
```

### 4.6 Оптимизаторы по памяти

ResNet-50 + BNNeck + голова 1171 = 25,91 M параметров, 98,8 МБ в fp32.

| оптимизатор | состояние | params+grads+state | доступен на sm_50? |
|---|---|---:|---|
| SGD, momentum=0 | 0 | 197,6 МБ | да |
| SGD + momentum | 1×N | 296,4 МБ | да |
| **Adam / AdamW** | 2×N (`exp_avg`, `exp_avg_sq`) | **395,3 МБ** | да ← сейчас |
| Adam, `amsgrad=True` | 3×N | 494,1 МБ | да, но не нужно |
| **`torch.optim.Adafactor`** | ~O(R+C) вместо O(R·C) | ~**210 МБ** | **да** — чистый PyTorch, есть в ядре |
| bitsandbytes 8-bit Adam | 2×N в int8 | ~247 МБ | **НЕТ — требует CC ≥ 6.0** |

Про bitsandbytes дословно из
[официальной установки](https://huggingface.co/docs/bitsandbytes/main/en/installation):
*«bitsandbytes is currently supported on NVIDIA GPUs with **Compute Capability 6.0+**»*;
PyPI-колёса собраны под `sm60, sm70, sm75, sm80, sm86, sm89, sm90` — **sm_50 нет**,
а поддержка Maxwell выведена из колёс с v0.48.0.

Про SGD vs Adam в re-ID: в BoT слово «SGD» **не встречается ни разу** — сравнения
нет. Что есть (§2, п.8): *«**Adam** method is adopted … The initial learning rate
is set to be **0,00035** and is decreased by 0,1 at the 40th epoch and 70th epoch …
Totally there are **120 training epochs**»* плюс warmup с 3,5·10⁻⁵ до 3,5·10⁻⁴
за 10 эпох. Меняя Adam на SGD ради 98,8 МБ, вы выбрасываете всю выверенную сетку
гиперпараметров. **`foreach=False` даёт ту же экономию бесплатно.**

### 4.6b Аллокатор на Windows

- **`expandable_segments:True` на Windows не работает — молча.** В
  [`c10/cuda/CUDAAllocatorConfig.cpp`](https://github.com/pytorch/pytorch/blob/release/2.10/c10/cuda/CUDAAllocatorConfig.cpp)
  функция возвращает `false`, если не определён `PYTORCH_C10_DRIVER_API_SUPPORTED`,
  а в [`c10/cuda/CMakeLists.txt`](https://github.com/pytorch/pytorch/blob/release/2.10/c10/cuda/CMakeLists.txt#L71-L74)
  этот макрос ставится только `if(NOT WIN32)`. Поддержка Windows появилась
  в ветке `main` уже **после** того, как оттуда выпилили sm_50 — то есть нам
  недоступна в принципе.
- **Что работает:** `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8`
  ([дока](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management)):
  *«This can reduce fragmentation and may allow some borderline workloads to complete
  without running out of memory»*. Применять, если видите
  `OOM … reserved but unallocated`.
- **`torch.cuda.empty_cache()`** «releases all **unused** cached memory … However,
  the occupied GPU memory by tensors **will not be freed**» — своему процессу
  не помогает, в цикле обучения только замедляет. Вызывать один раз между
  train и eval.

### 4.7 channels_last на Maxwell — минус, а не плюс

NHWC ускоряет, потому что этого требуют **Tensor Cores**. Официальный туториал
([memory format](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html))
не скрывает: *«the most significant performance gains are observed on NVIDIA's
hardware **with Tensor Cores support running on reduced precision (torch.float16)**»*,
«over 22 % … while utilizing AMP», «8 %–35 % on **Volta** devices».

Прямой замер в fp32 **без** Tensor Cores ([pytorch#50036](https://github.com/pytorch/pytorch/issues/50036),
TITAN RTX, fp32):

| ядро / формат тензора | время |
|---|---:|
| NCHW / NCHW | **0,97 с** |
| NCHW / NHWC | 2,21 с |
| NHWC / NHWC | **2,35 с (×2,4 хуже)** |
| NHWC / NCHW | 2,77 с |

На Maxwell Tensor Cores нет вовсе, а NHWC-оптимизированных cuDNN-ядер под sm_50
не существует. Плюс `channels_last` **не экономит ни байта** — это тот же объём,
другой порядок strides. **Не включать.**

### 4.8 `cudnn.benchmark`

`torch.backends.cudnn.benchmark = True` на первом шаге каждой новой формы входа
перебирает алгоритмы свёртки и кэширует лучший. Разброс выигрыша по замерам
([discuss.pytorch.org](https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936)):
от **+5 %** («free ~1.05x») до **+25 %** и **+30…40 %** на отдельных сетях;
у одного пользователя на torchvision ResNet-101 получилось **хуже** — надо мерить.

**Цена, и она для нас нетривиальна:** во время перебора cuDNN аллоцирует workspace
под каждый кандидат, часть из которых требует сотен МБ. На карте с 4 ГБ это может
дать OOM **именно на первой итерации**, при том что стабильный шаг влезал бы.
Правило: **если OOM ловится на шаге 1 — первым делом выключить `benchmark`.**

Вход у нас фиксирован (208×208), поэтому перебор случится один раз — **при условии
`drop_last=True`**, иначе хвостовой неполный батч даст вторую форму и второй
перебор. В P×K-сэмплере `drop_last=True` и так обязателен.

## 5. Скорость: сколько эпох реально успеть

### 5.1 Модель времени

Roofline по шагу обучения:
`t_шаг = max(3·MAC / 702,7e9 ,  2,5·трафик / 80,2e9)`
(множитель 3 по арифметике — forward + два backward-прохода; 2,5 по трафику —
запись в forward плюс чтение+запись в backward).

Трафик оцениваем как `2 × Σ(размер всех созданных тензоров)` — каждый тензор
хотя бы раз пишется и хотя бы раз читается.

**Калибровка по известной точке.** R50@208 B=32, 7248 кадров = 226,5 шагов:
идеальная эпоха = **112,1 с**, наблюдаемая **245–353 с** →
коэффициент реализации **k = 2,18…3,15**, то есть карта отдаёт
**32–46 % от roofline**. Это нормальная цифра для cuDNN fp32 на Maxwell
под WDDM и с загрузчиком данных на Windows. Всё остальное масштабируем этим же k.

### 5.2 Арифметическая интенсивность — главное число раздела

| конфигурация | MAC/шаг | трафик/шаг, МБ | t_арифм, мс | t_память, мс | сохранённых тензоров | интенсивность, MAC/байт | что упирается |
|---|---:|---:|---:|---:|---:|---:|---|
| R50 @208 B=32 | 116,0 G | 4874 | 165 | 61 | 112 | **22,7** | арифметика |
| R50-IBN-a @208 B=32 | 116,0 G | 5614 | 165 | 70 | 164 | 19,7 | арифметика |
| R50 @256 B=32 | 170,9 G | 7329 | 243 | 92 | 112 | 22,2 | арифметика |
| OSNet-AIN @208 B=32 | 41,4 G | 10 359 | 59 | 129 | **471** | **3,8** | **память** |
| OSNet-AIN @256×128 B=32 | 31,3 G | 7846 | 45 | 98 | 471 | 3,8 | память |

Точка баланса карты — 8,76 MAC/байт. **OSNet с интенсивностью 3,8 сидит глубоко
в memory-bound зоне**: у него в 2,8 раза меньше MAC, но в 2,1 раза больше трафика
и в 4,2 раза больше запусков ядер, чем у ResNet-50.

**Важная поправка к формулировке брифа:** OSNet-AIN меньше ResNet-50
**в 11,7 раза по параметрам** (2,19 M против 25,56 M), но только
**в 2,8 раза по FLOPs** (1,29 против 3,62 GMAC при 208×208). «В ~10 раз меньше
FLOPs» — неверно; в 10 раз меньше вес файла, а не вычислений.

### 5.3 ТАБЛИЦА ВРЕМЕНИ ЭПОХИ

Датасет 7248 кадров. Диапазон = калибровочный коридор k = 2,18…3,15.

| конфигурация | эпоха, с | эпох за **3 ч** | эпох за **8 ч** | эпох за 24 ч |
|---|---:|---:|---:|---:|
| **R50 @208 B=32** *(измерено)* | **245…353** | 30…44 | 81…117 | 244…352 |
| **R50-IBN-a @208 B=32** | 245…353 (+0…15 % на IN-ядра) | 27…44 | 72…117 | 216…352 |
| R50 @224 B=32 | 276…398 | 27…39 | 72…104 | 216…312 |
| **R50 @256 B=24** | 361…520 | 20…29 | 55…79 | 166…239 |
| R50 @256 B=16 | 361…520 | 20…29 | 55…79 | 166…239 |
| **OSNet-AIN @208 B=24** | 168…241 | 44…64 | 119…171 | 357…515 |
| OSNet-AIN @256×128 B=32 | 127…183 | 59…85 | 157…226 | 472…680 |
| R50 + gradient checkpointing @208 B=64 | 320…495 | 21…33 | 58…90 | 174…270 |

**Удвоенная выборка (свои 7248 + столько же чужих = 14 496 кадров/эпоху):**

| конфигурация | эпоха, с | эпох за **3 ч** | эпох за **8 ч** | эпох за 24 ч |
|---|---:|---:|---:|---:|
| R50 @208 B=32 | 490…706 | 15…22 | 40…58 | 122…176 |
| R50-IBN-a @208 B=32 | 490…706 | 15…22 | 40…58 | 122…176 |
| R50 @256 B=24 | 722…1040 | 10…14 | 27…39 | 83…119 |
| OSNet-AIN @208 B=24 | 335…483 | 22…32 | 59…85 | 178…257 |
| OSNet-AIN @256×128 B=32 | 254…366 | 29…42 | 78…113 | 236…340 |

**Чтение таблицы.** Первая попытка была 30 эпох — это **2,0…2,9 часа**.
То есть **весь цикл обучения помещается в один прогон длиной 3 часа**, а в 8 часов
влезает 2–3 полных цикла с разными гиперпараметрами. Дефицита времени у задачи нет
— узкое место это память и риск падения узла, а не скорость.

**Про OSNet.** Roofline обещает 168…241 с/эпоху при батче 24 — быстрее ResNet-50.
Но: (а) depthwise-свёртки в cuDNN на Maxwell работают заметно ниже roofline,
(б) 471 тензор против 112 означает ~1400 запусков ядер на шаг против ~360, а
накладные расходы на запуск под WDDM — 8–20 мкс на ядро, то есть **+14 мс/шаг**
против +4 мс у ResNet-50. Честная оценка: **паритет с ResNet-50 или чуть быстрее,
не в 2,8 раза**. Считать по таблице, но замерить на 20 шагах перед планированием.

---

## 6. Устойчивость к падению узла

### 6.1 Что сохранять в чекпойнт

Узел падал дважды. Минимум, при котором возобновление **воспроизводит** прогон,
а не «продолжает чем-то похожим»:

```python
state = {
    "epoch":        epoch,
    "global_step":  step,
    "model":        model.state_dict(),
    "optimizer":    optimizer.state_dict(),
    "scheduler":    scheduler.state_dict(),          # часто забывают
    "scaler":       scaler.state_dict(),             # если AMP
    "sampler_epoch": epoch,                          # для set_epoch() у P×K-сэмплера
    "rng": {
        "python":  random.getstate(),
        "numpy":   np.random.get_state(),
        "torch":   torch.get_rng_state(),
        "cuda":    torch.cuda.get_rng_state_all(),
    },
    "best_map":     best_map,
    "config_hash":  cfg_hash,   # чтобы не возобновить с чужим конфигом
}
```

Без `rng` и `sampler_epoch` возобновление даёт другую последовательность батчей и
другие аугментации — результаты прогона перестают быть сравнимыми, а при нашем
`sd(mAP) = 0,0134` это прямо съедает порог значимости.

### 6.2 Атомарная запись — обязательна

`torch.save` **не атомарен**: он пишет zip-архив потоком, и падение в середине
оставляет обрезанный файл. Ровно этот сценарий разбирается в
[Lightning issue #19970 «Make checkpoint saving fully atomic»](https://github.com/Lightning-AI/pytorch-lightning/issues/19970).
Правильный шаблон (работает и на Windows, `os.replace` там атомарен для файлов
на одном томе):

```python
import os, torch
tmp = path + ".tmp"
with open(tmp, "wb") as f:
    torch.save(state, f)
    f.flush()
    os.fsync(f.fileno())      # без fsync атомарен только rename, но не содержимое
os.replace(tmp, path)         # атомарная подмена
```

Плюс: держать **две ротируемые копии** (`ckpt_a.pt` / `ckpt_b.pt`, писать в ту,
что старее) — тогда даже повреждение файловой системы в момент `replace`
оставляет рабочий чекпойнт на эпоху назад. При эпохе 245–353 с цена — одна эпоха.

### 6.3 Размер чекпойнта — что сколько стоит

Измерено на ваших файлах и подтверждено расчётом (§3.3):

| что храним | размер | множитель |
|---|---:|---:|
| только веса (`final.pt`) | 103 142 354 Б ≈ **98,4 МБ** | ×1 |
| веса + SGD без momentum | ≈ 98,4 МБ | ×1 |
| веса + SGD с momentum | ≈ **196,6 МБ** | ×2 |
| **веса + Adam** (`ckpt.pt`) | 308 885 785 Б ≈ **294,6 МБ** | **×3** |
| веса + Adam + EMA | ≈ 393 МБ | ×4 |
| только EMA-веса | ≈ 98,4 МБ | ×1 |

30 эпох × 294,6 МБ = 8,6 ГБ, если хранить всё. Практика: держать
**последний чекпойнт (ротация из двух) + лучший по mAP + финальные веса**.
Это 3×294,6 + 98,4 ≈ **982 МБ** на прогон.

### 6.4 Стоит ли хранить только EMA-веса

**Для возобновления — нет.** EMA-веса не содержат состояния Adam; возобновление
с них равносильно рестарту оптимизатора, что при re-ID с triplet-лоссом
даёт заметный провал на 2–5 эпох.

**Для отдачи результата — да.** EMA (`decay = 0,999`, что при 226,5 шагах/эпоху
даёт окно ≈ 1000 шагов ≈ 4,4 эпохи) обычно чуть стабильнее последней точки и
стоит всего +98,4 МБ на диск и +98,4 МБ в RAM (EMA-копию держать на CPU!,
`ema_model.to('cpu')` — иначе съест 98,8 МБ видеопамяти из нашего запаса в 350 МБ).

Схема, которая ничего не ломает: `ckpt.pt` (веса+Adam+RNG, ротация ×2) для
возобновления, `ema.pt` (только EMA-веса) — для оценки и выдачи.

### 6.5 Запуск

WMI-запуск (`Win32_Process.Create`) уже проверен — процесс переживает разрыв SSH,
потому что не является дочерним для сессии. К этому стоит добавить:

- `stdout/stderr` в файл с `flush` (иначе при падении узла последние строки лога
  теряются в буфере) — `python -u train.py > train.log 2>&1`;
- файл-маркер `heartbeat.txt` с меткой времени, обновляемый каждый шаг: по нему
  снаружи видно, жив прогон или висит;
- `RUNNING.pid` и проверка при старте — чтобы два прогона не начали писать
  в один чекпойнт;
- планировать прогоны **по 3 часа** (§5.3) — полный цикл 30 эпох помещается
  целиком, и падение узла стоит максимум одного цикла.

---

## 7. Что точно НЕ влезет — с расчётом

Бюджет: **~3450–3600 МБ**. Все цифры — из модели §3.1, fp32 + Adam.

| модель | параметры | веса+град.+Adam | активации | **ИТОГО, МБ** | превышение |
|---|---:|---:|---:|---:|---|
| **ViT-B/16 @224 B=32, full FT** | 86,70 M | 1322,9 | 4283,5 | **6368,9** | **×1,8** |
| ViT-B/16 @224 B=16, full FT | 86,70 M | 1322,9 | 2141,8 | **3905,9** | ×1,1 |
| ViT-B/16 @224 B=8, full FT | 86,70 M | 1322,9 | 1070,9 | 2674,4 | влезает, но **бесполезно** |
| **TransReID ViT-B @256×128 B=32** (patch 16, stride 12 → 211 токенов) | 86,71 M | 1323,1 | 4638,5 | **6777,4** | **×1,9** |
| TransReID ViT-B B=16 | 86,71 M | 1323,1 | 2319,3 | **4110,2** | ×1,2 |
| **CLIP-ReID ViT-B/16 @256×128 B=32** (129 токенов), только stage-2 | 86,65 M | 1322,1 | 2657,1 | **4497,8** | ×1,3 |
| CLIP-ReID stage-1 (+текстовый энкодер 63,4 M) | 150,1 M | 2290 | 2657 | **~5070** | ×1,4 |
| **DINOv2-L (ViT-L/14) @224 B=8** | 304,38 M | 4644,4 | 3878,7 | **9224,9** | **×2,6** |
| DINOv2-L @224 **B=1** | 304,38 M | **4644,4** | 484,8 | **5322,0** | **×1,5** |
| **fast-reid R50-IBN @256×256 B=64** | 24,69 M | 1,0×376,8 | 6849,0 | **8373,1** | **×2,3** |
| fast-reid R50-IBN @256×256 B=32 | 24,69 M | 376,8 | 3424,5 | **4434,9** | ×1,2 |

### 7.1 Приговоры по пунктам

**ViT-B/16 full FT — нет.** При батче 32 нужно 6369 МБ, это 1,8× бюджета.
Формально влезает при батче 8 (2674 МБ), но батч 8 для re-ID мёртв: P×K-сэмплинг
с K=4 даёт P=2, то есть **две машины в батче**, всего `2·4·3·1·4 = 96` триплетов
против 2688 при P=8/K=4 (−96 %). Плюс BatchNorm-статистики по 8 образцам — шум.
Плюс скорость: 17,56 GMAC/изобр. против 3,62 у R50 → **эпоха 1200…1700 с при
батче 8**, то есть 30 эпох = 10…14 часов, и это при заведомо плохом батче.

**DINOv2-L — нет, даже при батче 1.** Здесь нужен не «поджать активации», а
физически невозможное: только веса + градиенты + два момента Adam =
`304,38 M × 4 Б × 4 = 4644,4 МБ` > 4096 МБ физической памяти карты. Никакой
checkpointing, AMP или батч-1 не помогут — **состояние оптимизатора не влезает
в карту в принципе**. Вариант «заморозить backbone и учить голову»
(параметры в оптимизаторе ≈ 0) оставляет 1161 МБ весов + активации fp16 — это
теоретически возможно, но это уже не обучение DINOv2, а линейный пробинг поверх
замороженных признаков, и его скорость — 81 GMAC/изобр., то есть **22× медленнее
ResNet-50**, эпоха ~1,5–2 часа даже без backward по backbone.

**CLIP-ReID ViT-B — нет.** Stage-2 при батче 32 даёт 4498 МБ (×1,3). При батче 16
влезло бы (2970 МБ), но это снова P=4 и −75 % триплетов. Stage-1 требует держать
текстовый энкодер (63,4 M параметров, +242 МБ только весами) — 5070 МБ.

**TransReID — нет.** Хуже ViT-B из-за overlapping patches: stride 12 вместо 16
даёт 211 токенов вместо 129 при входе 256×128 → **+64 % активаций**. 6777 МБ при
батче 32. Модуль JPM (jigsaw patch module) добавляет ещё один проход по
трансформеру для перемешанных патчей — фактически **×2 к активациям последнего
слоя**. SIE (side information embedding) к тому же **требует `camera_id`
на инференсе**, которого у нас нет.

**fast-reid со штатным конфигом (256×256, батч 64) — нет.** 8373 МБ, ×2,3.
Влезает только урезанный вариант: `SOLVER.IMS_PER_BATCH 32`, `INPUT.SIZE_TRAIN [256,256]`
→ 4435 МБ… тоже не влезает; нужно либо 256×128 (половина пикселей → ~2400 МБ),
либо батч 16 при 256×256 (2485 МБ). То есть **готовый конфиг fast-reid нельзя
запустить как есть** — его надо переписывать под 4 ГБ, и веса `veriwild_bot`
(1 036 401 141 Б) при загрузке ещё и потребуют ~1 ГБ RAM на CPU (не видеопамяти).

### 7.2 Что при этом ВЛЕЗАЕТ и остаётся осмысленным

| конфигурация | МБ | запас | триплетов в батче (P=8,K=4) |
|---|---:|---:|---:|
| **R50-IBN-a @208 B=32 Adam** | 3134 | ~350 МБ | 2688 |
| R50 @208 B=32 Adam | 3134 | ~350 МБ | 2688 |
| R50-IBN-a @256×128 B=32 Adam | ~2400 | ~1100 МБ | 2688 |
| OSNet-AIN @256×128 B=32 Adam | 2999 | ~480 МБ | 2688 |
| OSNet-AIN @208 B=24 Adam (P=6,K=4) | 2972 | ~500 МБ | 1440 |
| **R50-IBN-a @208 B=64 + checkpointing** | ~2600 | ~900 МБ | **11 520** |

Последняя строка — самый интересный вариант: gradient checkpointing на
`layer1..layer4` освобождает достаточно, чтобы поднять батч до 64 (P=16, K=4),
что даёт **в 4,3 раза больше триплетов** (2688 → 11 520) ценой +25–30 % времени
эпохи (320…495 с вместо 245…353 с). По Табл. 5 BoT переход 8×4 → 16×4 стоит
**+3,7 mAP на Market1501 и +3,6 на DukeMTMC** (§4.3) — при нашем `sd(mAP) = 0,0134`
это в 2,7 раза больше порога значимости. Для задачи, где главная беда — что сеть
учит камеру, а не машину, больший P означает ещё и больше кросс-камерных негативов
в каждом батче. **Это единственный приём из всего списка, который покупает метрику,
а не мегабайты.**

---

## 8. ONNX-экспорт и инференс на sm_50

### 8.1 Экспорт

- `torch.onnx.export` в torch 2.x требует **явного `opset_version`** — умолчание
  было объявлено устаревшим в 1.12 и меняется от версии к версии.
- В torch ≥ 2.5 появился новый экспортёр `dynamo=True`; **для sm_50 он не нужен и
  рискован** — старый TorchScript-экспортёр (`dynamo=False`) стабильнее для CNN.
- **Рекомендация: `opset_version=13`.** Причины:
  - opset 13 покрывает всё, что есть в ResNet/IBN/OSNet (Conv, BatchNormalization,
    InstanceNormalization, Relu, MaxPool, GlobalAveragePool, Gemm, Add, Mul,
    Concat, Split, Sigmoid, ReduceMean);
  - opset 13 читают **все** версии onnxruntime начиная с 1.8 — то есть и та
    старая сборка под CUDA 11.8, которая единственная работает на sm_50 (§8.2);
  - ваши эталонные веса `vehicle-reid-0001` собраны экспортёром `pytorch 1.3`,
    то есть в районе opset 9–11 — держаться низкого opset значит оставаться
    совместимым с тем же инференс-стеком.
- Экспортировать с **динамической осью батча** (`dynamic_axes={'input': {0: 'N'},
  'output': {0: 'N'}}`), вход НЕ зашивать: у `vehicle-reid-0001` вход тоже
  динамический, 208 не зашито — сохраняем это свойство.
- `InstanceNormalization` (нужен для IBN-a и OSNet-AIN) есть в ONNX с opset 6 —
  ограничением не является.

### 8.2 Инференс: CUDA EP на sm_50 — ловушка

Список архитектур в сборке onnxruntime задан в
[`cmake/CMakeLists.txt`](https://github.com/microsoft/onnxruntime/blob/v1.22.0/cmake/CMakeLists.txt#L1565-L1580):

```cmake
if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12)
  # 37, 50 still work in CUDA 11 but are marked deprecated ...
  set(CMAKE_CUDA_ARCHITECTURES "37-real;50-real;52-real;60-real;70-real;75-real;80-real;86-real;89")
elseif (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
  set(CMAKE_CUDA_ARCHITECTURES "52-real;60-real;70-real;75-real;80-real;86-real;89-real;90")
else()
  set(CMAKE_CUDA_ARCHITECTURES "all")
```

Читать так: **сборка под CUDA 12.x содержит `52-real`, но НЕ `50`.** Суффикс
`-real` означает только SASS, без PTX, поэтому JIT вниз до sm_50 невозможен.
Quadro M2000M — это GM107, то есть **sm_50, а не sm_52**, и на CUDA-12-сборке
onnxruntime она упадёт с `cudaErrorNoKernelImageForDevice`.

| вариант инференса | работает на sm_50? | как ставить |
|---|---|---|
| `onnxruntime-gpu` c PyPI (≥1.19 = CUDA 12) | **нет** | — |
| `onnxruntime-gpu` сборка под **CUDA 11.8** (≤1.20.x) | **да** | `pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/` ([офиц. инструкция](https://onnxruntime.ai/docs/install/)) |
| **`onnxruntime-directml`** | **да, любая DX12-карта** | `pip install onnxruntime-directml` — **самый надёжный путь на Windows** |
| `onnxruntime` (CPU) | да | всегда работает, для 7248 кадров приемлемо |
| TensorRT EP | **нет** | TensorRT ≥ 8.6 не поддерживает sm_50 |
| OpenVINO | да (CPU/iGPU) | ваш эталон `vehicle-reid-0001` и так из Intel OMZ |

**Рекомендация:** экспорт `opset_version=13`, инференс через
**onnxruntime-directml** (единственный GPU-путь на Windows, не зависящий от
CUDA-судьбы Maxwell) с проверкой численного совпадения против PyTorch
(`np.allclose(rtol=1e-3, atol=1e-4)` на 50 изображениях) и против CPU EP.

---

## 9. Итог: максимальная разумная конфигурация

### 9.1 Базовая — запускать первой

```
модель        ResNet-50-IBN-a, вход 208×208
батч          32 (P=8 × K=4)
оптимизатор   Adam(lr=3.5e-4, foreach=False)
точность      fp32
память        3133,8 − 98,8 (foreach=False) ≈ 3035 МБ из ~3500 доступных
эпоха         245…353 с  →  30 эпох = 2,0…2,9 ч  →  один прогон
```

IBN-a **не стоит ни одного мегабайта дополнительно** по сравнению с обычным
ResNet-50 (доказательство в §3.3), поэтому переход бесплатен и по памяти,
и по числу параметров.

### 9.2 Целевая — если нужен больший батч (а он нужен)

```
модель        ResNet-50-IBN-a, вход 208×208
батч          64 (P=16 × K=4) через gradient checkpointing на layer1..layer4
              use_reentrant=False, bn.momentum 0,1 → 0,05 (баг #96136)
оптимизатор   Adam(lr=3.5e-4, foreach=False)
память        ~2600 МБ                       (запас ~900 МБ)
эпоха         320…495 с  →  30 эпох = 2,7…4,1 ч
выигрыш       триплетов 2688 → 11 520 (×4,3); по BoT Табл. 5 это +3,7 mAP
```

Это **максимальная разумная конфигурация**: больше батча (128) по BoT уже хуже,
больше входа (256) не влезает, более тяжёлая модель (ViT/DINOv2) не влезает
в принципе (§7).

### 9.3 Альтернатива, если берём OSNet

```
модель        OSNet-AIN x1.0, вход 256×128 — НЕ 208×208
              (при 208×208 и батче 32 нужно 3908 МБ — не влезает)
батч          32
память        2999 МБ
эпоха         127…183 с по roofline; ожидать паритет с R50
              (depthwise-свёртки + 471 тензор против 112 под WDDM)
```

### 9.4 Версии, от которых нельзя отступать

```
torch        2.9.1+cu126   (потолок 2.14.0+cu126;  запасной 2.7.1+cu118)
torchvision  0.24.1+cu126  (0.29.0+cu126        ;             0.22.1+cu118)
             pip install ... --index-url https://download.pytorch.org/whl/cu126
НЕ ставить   2.8.0+cu126 (регрессия sm_50 на Windows, PR #152069)
             любые +cu128 / +cu129 / +cu130 / +cu132 (там cuDNN ≥ 9.19 и нет sm_50)
             torch >= 2.15 (колёс cu12x не будет вообще)
драйвер      ветка R580 — ЗАФИКСИРОВАТЬ, не обновлять; это последняя с Maxwell
cuDNN        должен быть < 9.11.0 — в канале cu126 он пиннится (9.5 / 9.7 / 9.10.2)
onnxruntime  onnxruntime-directml для инференса (CUDA-сборки с PyPI не видят sm_50)
```

**Проверка перед любым обучением — три строки, обязательны:**

```python
import torch
print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())
print(torch.cuda.get_arch_list())            # должен содержать 'sm_50'
print(torch.cuda.get_device_capability(0))   # должно быть (5, 0)
# cudnn.version() -> 90500 / 90700 / 91002 — всё, что < 91100, годится
```

### 9.5 Настройки, которые включить сразу

```python
torch.backends.cudnn.benchmark = True        # вход фиксирован → +5…40 %
DataLoader(..., drop_last=True)              # иначе второй перебор алгоритмов cuDNN
optimizer = torch.optim.Adam(p, lr=3.5e-4, foreach=False)   # −98,8 МБ пика
# channels_last  — НЕ включать (нет Tensor Cores, замер: до ×2,4 медленнее в fp32)
# AMP            — только если нужен батч >32 и только после замера памяти
# accumulation   — для triplet-лосса бесполезно (§4.3)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # если OOM-фрагментация
```

### 9.6 Чек-лист устойчивости к падению узла

```
[ ] чекпойнт каждую эпоху: model + optimizer + scheduler + RNG + sampler_epoch
[ ] атомарная запись: torch.save во временный файл -> f.flush() -> os.fsync() -> os.replace()
[ ] ротация двух файлов ckpt_a.pt / ckpt_b.pt
[ ] отдельный ema.pt (только веса, 98,4 МБ) — для оценки и выдачи, НЕ для возобновления
[ ] EMA-копия модели на CPU, не на GPU (иначе −98,8 МБ из запаса)
[ ] python -u train.py > train.log 2>&1  (иначе последние строки лога теряются)
[ ] heartbeat.txt с меткой времени каждые N шагов
[ ] проверка RUNNING.pid при старте
[ ] прогоны по 3 часа — полный цикл 30 эпох помещается целиком
```

### 9.7 Три числа, которые стоит запомнить

1. **3500 МБ** — реальный бюджет карты. Всё, что в расчёте §3.2 больше — не влезет.
2. **8,76 MAC/байт** — точка баланса roofline. ResNet-50 имеет 22,7 и упирается
   в арифметику; OSNet имеет 3,8 и упирается в память, поэтому его «в 12 раз
   меньше параметров» не превращается в «в 12 раз быстрее».
3. **PyTorch 2.14** — последний релиз, где вообще есть sm_50. Дальше только
   сборка из исходников с `TORCH_CUDA_ARCH_LIST=5.0` и CUDA ≤ 12.6.
