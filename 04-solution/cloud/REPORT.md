# REPORT — подготовка GPU-машины `lct-gpu` в Yandex Cloud (без создания ресурсов)

Дата: 2026-09-17. CLI `yc 1.34.0`, профиль `lct2026`, облако `b1gqis7qavhsob55m0d1`, каталог `b1grruo0n4tcqspfngdf`.
Выполнялись только `list` / `get` / `--help` (и локальные проверки). Ни одна команда `create/update/delete/start/attach` не запускалась — см. раздел 4.

> ⚠ **Инцидент с ключом.** `yc config list` печатает SA-ключ целиком (он встроен в `~/.config/yandex-cloud/config.yaml`), и первый
> же запуск этой команды (для проверки профиля) вывел приватный ключ в терминал. Файл `~/.config/lct2026/yc-sa-key.json` я не открывал,
> ключ никуда не копировал; дальше использовал только `yc config get <поле>`. Но лог сессии `stream.jsonl`, который харнесс пишет в эту
> же папку, сохранил вывод команды — я вырезал из него тело ключа (6 вхождений заменены на `[REDACTED]`, JSONL остался валидным).
> Поскольку ключ побывал в терминале и в логе, **рекомендую ротировать ключ SA `lct-worker`** (это делается вручную, я ключи не создавал)
> и не запускать `yc config list` в логируемых сессиях.

## 1. Образ, платформа, зона, минимальная ступень

| Параметр | Значение | Откуда |
|---|---|---|
| **Образ (ID)** | `fd8qb5sstf8h1lk8pmb8` | `yc compute image get --id …` (docs/… → `image-candidates.yaml`) |
| Имя / family | `ubuntu-2204-lts-cuda-12-2-v20260813` / `ubuntu-2204-lts-cuda-12-2` | то же; `get-latest-from-family` подтверждает, что это последний образ семейства |
| Создан | 2026-08-13 10:15 UTC, `status: READY`, `min_disk_size` 30 ГиБ, `pooled: true`, Gen 1.2 (`PCI_TOPOLOGY_V2`) | `yc compute image get` |
| Предустановлено | «Образ содержит NVIDIA Driver 535 и CUDA 12.2» (Ubuntu 22.04 LTS); тарификация продукта Free | [Marketplace: Ubuntu 22.04 LTS GPU CUDA 12.2](https://yandex.cloud/ru/marketplace/products/yc/ubuntu-2204-lts-cuda-12-2) (обновлён 13.08.2026). В `yc` описание пустое: «ubuntu-2204-lts-cuda-12-2 image» |
| Почему этот | единственный актуальный образ с драйвером, который документация рекомендует для T4i (`gpu-os.md`); для T4 документация рекомендует `ubuntu-2004-lts-gpu`, но его последняя сборка — 2022-05 (Python 3.8) | [docs: GPU → образы ОС](https://yandex.cloud/ru/docs/compute/concepts/gpus#os) |
| **Платформа** | `standard-v3-t4` — Intel Ice Lake with NVIDIA® Tesla® T4, 16 ГБ GDDR6 (альтернатива `standard-v3-t4i` — T4i, 24 ГБ; переключается `PLATFORM=standard-v3-t4i ./up.sh`) | [docs: Платформы](https://yandex.cloud/ru/docs/compute/concepts/vm-platforms) |
| **Минимальная ступень** | **1 GPU · 4 vCPU · 16 ГБ RAM** (ступени обеих платформ: 4/16, 8/32, 16/64, 32/128; не более 1 GPU на ВМ) | [docs: Конфигурации GPU](https://yandex.cloud/ru/docs/compute/concepts/gpus#config) — таблицы «standard-v3-t4» и «standard-v3-t4i» (в источнике `ru/compute/concepts/gpus.md`, коммит cadf579 от 16.09.2026) |
| **Зона** | **`ru-central1-d`** — единственная зона, где поддержка T4/T4i заявлена явно (таблица «Ограничения»: T4 ✔ в `d`, ✘ в `e`); для `a`/`b` документация ничего не утверждает — см. §5 | [docs: Платформы → Ограничения](https://yandex.cloud/ru/docs/compute/concepts/vm-platforms#restrictions) |
| Зоны по `yc` | `yc compute zone list`: a, b, d, e, k — UP; c — DOWN | `yc` |
| Квоты для запроса | `compute.instanceT4Gpus.count` (T4 в облаке) **и** `compute.instanceT4Gpus.count.ru-central1-d` (T4 в зоне d); для T4i — `compute.instanceT4IGpus.count`. Также проверить `compute.instanceCores.count` (по умолчанию 32) и `compute.instanceMemory.size` (128 ГБ) | [docs: Квоты и лимиты](https://yandex.cloud/ru/docs/compute/concepts/limits#compute-quotas); `yc quota-manager quota-limit list` — PermissionDenied у SA (нужна роль на облако) |

Другие GPU-образы из `standard-images` (полный список — `std-images.json`, 8272 образа; сводка по семействам — в `journal.md`):
`ubuntu-2404-lts-secureboot-cuda-13-0` (`fd8d70pq4ici5rur6o45`, 2026-08-13, Ubuntu 24.04 = Python 3.12 «из коробки», CUDA 13.0, драйвер ≥580) — **Gen 2 / UEFI Secure Boot**; в документации такие образы привязаны к платформе `gpu-standard-v3i`, для `standard-v3-t4` совместимость не заявлена → не выбран. `ubuntu-2004-lts-secureboot-cuda-12-2`, `ubuntu-2204-lts-cuda-11-4`, `container-optimized-image-gpu` — не подходят по возрасту/назначению.

## 2. Цены и арифметика

Источник чисел: таблица «Цены для региона Россия» на [docs/compute/pricing](https://yandex.cloud/ru/docs/compute/pricing#prices) и [docs/vpc/pricing](https://yandex.cloud/ru/docs/vpc/pricing#prices-public-ip). Таблица рендерится компонентом `<PriceList>` из API `https://yandex.cloud/api/priceList/getPriceList?installationCode=ru&currency=RUB&services[]=dn22pas77ftg9h3f2djj&services[]=dn28okfvqh19eiue6l2m` (найдено по сетевому логу headless-Chrome); полный JSON сохранён: `docs/pricelist-compute.json` (125 SKU), `docs/pricelist-vpc.json`, TSV-выжимки рядом. Все цены — **RUB с НДС** ([«Все цены в рублях … указаны с НДС»](https://yandex.cloud/ru/docs/compute/pricing)), 1 ГБ = 2³⁰ байт. Действуют с 01.05.2026 (IP — с 01.01.2026, T4i preemptible GPU — с 30.07.2026). Машиночитаемая копия — `prices.json` (её читают `status.sh`/`up.sh`).

| Позиция | SKU (`name`) | Цена | Ед. |
|---|---|---|---|
| T4: 1 GPU | `dn20ml8ifdps6m7048an` compute_gpu.vm.gpu.standard.v3-t4 | **70,27 ₽** | GPU × час |
| T4: 100 % vCPU | `dn24b7m6qol7tb7tukga` compute_gpu.vm.cpu.c100.standard.v3-t4 | **1,15 ₽** | vCPU × час |
| T4: RAM | `dn2lg2hrvbn5b8lm7em4` compute_gpu.vm.ram.standard.v3-t4 | **0,3074 ₽** | ГБ × час |
| T4i: 1 GPU | `dn2hql9evci880d8jq7i` compute_gpu.vm.gpu.gpu-standard.t4i | 158,11 ₽ | GPU × час |
| T4i: 100 % vCPU / RAM | `dn242l2ivnhdd5so2oga` / `dn290pbmohupnus9ajb7` | 1,15 ₽ / 0,3074 ₽ | vCPU × час / ГБ × час |
| Сетевой SSD (`network-ssd`, «Быстрый диск (SSD)») | `dn27ajm6m8mnfcshbi61` nbs.network-nvme.allocated | **0,0199 ₽** | ГБ × час (= 14,328 ₽/ГБ за 30 дней, 14,8056 ₽ за 31 день — в прайсе цена только почасовая) |
| Публичный IP (динамический или статический), активный | `dn229q5mnmp58t58tfel` network.public_fips | **0,26352 ₽** | IP × час |
| Резервирование неактивного статического IP (доплата) | `dn2nqa0nqffbhihsq2n5` network.public_fips.deallocated | 0,34038 ₽ | IP × час (неактивный статический = 0,26352 + 0,34038 = 0,6039 ₽/ч) |
| Прерываемая T4: GPU / vCPU / RAM | `dn2cpk4mc82b1vib72e5` / `dn2lsfskfirek2985fnd` / `dn2im0g43iedeohe4sac` | 17,56 / 0,3184 / 0,0768 ₽ | за час |
| Прерываемая T4i: GPU / vCPU / RAM | `dn2qlml2u48bng4jgilh` / `dn2960mi7268n67o8iae` / `dn25rffeums4j1ku5649` | 21,09 / **1,15** / **0,3074** ₽ (vCPU и RAM без скидки) | за час |
| Снимок / образ (хранение) | `dn2d7s257l01hsl9rc6d` / `dn2gnu9gs67vf056r8dj` | 0,0051 ₽ | ГБ × час |
| Исходящий трафик | `dn28ml7sjbb5v98jkuj3` network.egress.inet | первые 100 ГБ/мес — 0, далее 1,42 ₽ | ГБ (входящий — бесплатно) |
| Сети, подсети, группы безопасности | — | 0 | — |

Правила: ВМ тарифицируется «с момента запуска (RUNNING) и до полной остановки. Время, которое ВМ была выключена, не тарифицируется»; «Диски тарифицируются независимо от того, запущена ВМ или нет» ([docs/compute/pricing](https://yandex.cloud/ru/docs/compute/pricing#disk)). Публичный IP: «При остановке ВМ с динамическим публичным IP-адресом её адрес освобождается» ([docs/vpc/concepts/address](https://yandex.cloud/ru/docs/vpc/concepts/address#public-addresses)) — у остановленной ВМ **с динамическим IP (наш up.sh) платится только диск**; статический IP у остановленной ВМ стоит 0,6039 ₽/ч.

**Арифметика (T4, минимальная ступень 4 vCPU / 16 ГБ, диск network-ssd 60 ГБ, динамический IP):**

- Час работы: GPU 1 × 70,27 + vCPU 4 × 1,15 + RAM 16 × 0,3074 = 70,27 + 4,60 + 4,9184 = **79,7884 ₽** (ресурсы ВМ);
  диск 60 × 0,0199 = **1,194 ₽**; IP 0,26352 ₽ → **итого 81,25 ₽/ч** (81,24592).
- Час простоя (ВМ остановлена): диск 1,194 ₽/ч (динамический IP освобождён) = **28,66 ₽/сутки = 859,68 ₽/30 дней**. Если IP сделать статическим: 1,194 + 0,6039 = 1,7979 ₽/ч = 43,15 ₽/сутки.
- Три часа работы + диск на сутки: 3 × 79,7884 + 3 × 0,26352 + 24 × 1,194 = 239,3652 + 0,7906 + 28,656 = **268,81 ₽**.
- Для сравнения T4i (4/16): 158,11 + 4,60 + 4,9184 = 167,6284 ₽ + диск + IP = **169,09 ₽/ч**; три часа + сутки диска = 532,33 ₽.
- Ступень 8 vCPU / 32 ГБ дороже на 4 × 1,15 + 16 × 0,3074 = 9,52 ₽/ч (90,76 ₽/ч работы).
- cloud-init после старта (≈12 мин, см. §3) — это ≈16 ₽ оплаченного времени (15 мин — 20 ₽).
- Ориентир бюджета: 5000 ₽ ≈ 61 час работы T4-машины без учёта простоя диска.

**Прерываемая ВМ (`PREEMPTIBLE=1 ./up.sh`):** скидка для GPU-платформ есть. T4: 17,56 + 4 × 0,3184 + 16 × 0,0768 = 20,0624 ₽/ч (−75 % к 79,79; с диском и IP 21,52 ₽/ч). T4i: 21,09 + 4,60 + 4,9184 = 30,6084 ₽/ч (GPU −86,7 %, vCPU/RAM без скидки; с диском и IP 32,07 ₽/ч). Риски по [docs: Прерываемые ВМ](https://yandex.cloud/ru/docs/compute/concepts/preemptible-vm): «могут быть принудительно остановлены в любой момент» — через 24 ч после запуска или при нехватке ресурсов в зоне; «максимальный срок жизни — 24 часа»; могут вообще не создаться (`not enough resources`); нет SLA; при остановке ОС получает ACPI shutdown и 30 с на завершение (рекомендуют `DefaultTimeoutStopSec=30s`). Данные на диске сохраняются, ВМ можно запустить снова (`./up.sh` это делает). Для наших задач (Re-ID прогон по 11 416 кропам, замер одного кадра) — подходит при условии, что результаты пишутся на диск по ходу; cloud-init (~12 мин) придётся переждать заново только при пересоздании, а не при рестарте.

## 3. Скрипты и cloud-init (в этой папке)

| Файл | Назначение |
|---|---|
| `lct.env` | все параметры (каталог, зона, имена, платформа, ступень, образ, диск, пользователь); переопределяются переменными окружения |
| `lct-lib.sh` | общие функции: проверка профиля/каталога, `resource_exists` (различает «not found» и другие ошибки), расчёт стоимости по `prices.json` |
| `up.sh` | сеть `lct-net` и подсеть `lct-subnet` (`ru-central1-d`, 10.10.0.0/24) — только если их нет; ВМ `lct-gpu`: `standard-v3-t4`, 4 vCPU/16 ГБ/1 GPU, `--create-boot-disk name=lct-gpu-boot,type=network-ssd,size=60,image-id=fd8qb5sstf8h1lk8pmb8,auto-delete=true`, `--network-interface subnet-name=lct-subnet,nat-ip-version=ipv4` (динамический публичный IPv4), `--metadata-from-file user-data=cloud-init.rendered.yaml`, `--preemptible` по флагу. Ключ `~/.ssh/id_ed25519.pub` подставляется в cloud-init (пользователь `artem`, sudo без пароля) — так надёжнее, чем совмещать `--ssh-key` с `user-data`. Если ВМ уже есть и остановлена — запускает. В конце печатает IP, пишет `lct-gpu.ip`, ждёт порт 22 и печатает подсказки. `--dry-run` — показать команды |
| `down.sh` | удаляет ВМ (диск `auto-delete=true` уходит с ней), затем диск `lct-gpu-boot`, если остался, адреса `lct-*`, подсеть, сеть, локальные `lct-gpu.ip`/`cloud-init.rendered.yaml`; чужое (`default`, чужие адреса/диски) не трогает, только предупреждает. Идемпотентен — на пустом каталоге проходит без ошибок (проверено). Спрашивает подтверждение (`--yes` — без вопроса), `--dry-run` |
| `status.sh` | контроль расходов: все ВМ/диски/адреса/снимки/образы/файловые хранилища каталога и их ₽/ч по `prices.json` (RUNNING-ВМ по платформе, прерываемые — по своему тарифу; остановленные — 0; IP; диски по типу), итог в час/сутки/30 дней; `--json` |
| `prices.json` | тарифы с SKU и ссылками (единственный источник цифр для скриптов) |
| `cloud-init.yaml` | шаблон (`__SSH_PUBKEY__` подставляет up.sh). Пользователь `artem`; `write_files` → `/usr/local/sbin/lct-provision.sh` (идемпотентный: метки шагов в `/var/lib/lct`, можно перезапускать), `/opt/lct/smoke.py`, `/opt/lct/blender-smoke.py`, `/etc/profile.d/lct.sh`; `runcmd` запускает провижининг, лог `/var/log/lct-provision.log`, результат самопроверки `/var/log/lct-smoke.log` (`SMOKE_REID OK`, `SMOKE_BLENDER OK`) |

Что ставит cloud-init: `apt update` + безопасный `upgrade` (пакеты `nvidia*`, `libnvidia*`, `cuda*`, ядро — на `apt-mark hold`, иначе апгрейд драйвера без перезагрузки ломает `nvidia-smi`); `time` (для `/usr/bin/time` из ЗАМЕР.md), `xz-utils`, X11/GL-библиотеки для headless Blender; **Python 3.12** через `uv` (`uv python install 3.12`, venv `/opt/reid/venv`, `python`/`pip` доступны через `/etc/profile.d/lct.sh`); **PyTorch 2.14.0+cu126 + torchvision** (индекс `download.pytorch.org/whl/cu126`); **`onnxruntime-gpu==1.26.0`** (последняя сборка PyPI под CUDA 12 — с 1.27 PyPI-сборка требует CUDA 13/драйвер ≥580), `onnx`, `timm`, `transformers`, `open_clip_torch`, `numpy`, `Pillow`, `safetensors`, `huggingface_hub`; **Blender 4.5.14 LTS** — `https://download.blender.org/release/Blender4.5/blender-4.5.14-linux-x64.tar.xz` (378 045 212 байт, опубликован 15.09.2026), SHA-256 `9ba871ff2ecd36526b77432745980b7e6664ecd0c7ca11c48849073dcfe06da3` из официального [`blender-4.5.14.sha256`](https://download.blender.org/release/Blender4.5/blender-4.5.14.sha256), распаковка в `/opt/blender`, симлинк `/usr/local/bin/blender`; каталоги `/data/{reid,render,hf}` (`HF_HOME=/data/hf`).

Совместимость драйвера 535 (CUDA 12.2) с этим стеком: по [NVIDIA minor version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) приложения на CUDA 12.x работают на драйвере ≥ 525, CUDA 13.x требует ≥ 580; ограничение — PTX JIT. Релизные wheel'ы PyTorch 2.14 cu126 собраны SASS-only с архитектурами {50,60,70,75,80,86,90} ([pytorch `.ci/manywheel/build_env_setup.py`, release/2.14](https://github.com/pytorch/pytorch/blob/release/2.14/.ci/manywheel/build_env_setup.py)) — sm_75 (T4) есть. Blender 4.5: CUDA — compute capability ≥ 3.0, OptiX — CC ≥ 5.0 и **драйвер ≥ 535** ([manual 4.5: GPU Rendering](https://docs.blender.org/manual/en/4.5/render/cycles/gpu_rendering.html)) — T4 (CC 7.5, RT-ядра) и драйвер 535 образа проходят впритык; сцена ЗАМЕР сама выбирает OPTIX → CUDA → CPU. Локально проверено: тарболл скачан, SHA-256 совпал, `blender -b --version` = «Blender 4.5.14 LTS», сцена из ЗАМЕР.md (`bashnya.glb`, 876 объектов) отрендерена на CPU в мини-настройках — `FILM OK … устройство CPU` (скрипт проекта совместим с 4.5.x).

Проверка синтаксиса (`checks.txt` — полный вывод):
- `bash -n up.sh down.sh status.sh lct-lib.sh lct.env` — OK; `shellcheck -x` 0.11.0 (статический бинарник в `tools/`, в системе не было) — **0 замечаний** на всех; встроенный `lct-provision.sh` извлечён из YAML — `bash -n` + `shellcheck` OK; `cloud-init.yaml` — валидный YAML (PyYAML), встроенные `.py` — `py_compile` OK.
- `--help` всех используемых команд `yc` сохранены в `yc-help/`; флаги в 1.34.0 существуют: `--create-boot-disk` (`name,type,size,image-id,auto-delete`), `--network-interface` (`subnet-name,nat-ip-version`), `--metadata-from-file`, `--metadata`, `--preemptible`, `--platform/--cores/--memory/--gpus/--hostname/--labels`, `vpc subnet create --network-name/--zone/--range`, `… delete --name|--id`, `compute image get-latest-from-family`, `compute disk-type list` (есть `network-ssd`).
- Прогон на заглушке `tests/fake-yc` (фикстуры: 2 RUNNING-ВМ T4/T4i-preemptible, остановленная ВМ, 3 диска, 2 адреса, снимок): `status.sh` = 114,50 ₽/ч, совпало с ручным расчётом; `up.sh --dry-run`/`down.sh --dry-run` печатают ожидаемые команды; заглушка падает на любой изменяющей команде — ни одна не вызвана.
- Реальные `./up.sh --dry-run`, `./down.sh --dry-run`, `./status.sh` (только get/list) — отработали на пустом каталоге, `status.sh` показывает 0 ₽/ч.
- **Прогон cloud-init-скрипта в контейнере** (`tests/container-test.sh`: podman, `ubuntu:22.04`, хранилище внутри этой папки и удалено; без GPU и без скачивания torch): apt upgrade + все зависимости, `uv` → Python 3.12.14, dry-run резолва torch (2.14.0+cu126, torchvision 0.29.0+cu126, nvidia-*-cu12 12.6.x, cudnn 9.10.2.21), установка `onnxruntime-gpu==1.26.0` + onnx/numpy/pillow (ORT видит `CUDAExecutionProvider`), Blender 4.5.14 (SHA-256 ок, `blender -b --version` ок, headless-библиотек хватает), `/usr/bin/time`, повторный запуск шагов → skip. Итог `CONTAINER_TEST_OK` за 4 м 44 с (лог `tests/ci-out/container-test.log`). Тест поймал реальную ошибку — в Ubuntu 22.04 нет пакета `libxrandr1` (нужен `libxrandr2`); исправлено. Не проверено локально только то, что требует GPU/torch: `torch.cuda.is_available()`, ORT CUDA-сессия, `SMOKE_BLENDER OK`.

**Время cloud-init после старта:** скачать ≈ 4,7 ГБ (torch + CUDA-библиотеки ≈ 3,8 ГБ, ORT 0,26 ГБ, Blender 0,36 ГБ, apt ≈ 0,2–0,3 ГБ, uv/python ≈ 0,08 ГБ) и записать ≈ 15 ГБ (кэш wheel'ов + распакованные ≈ 10 ГБ + Blender 1,1 ГБ + apt). Узкое место — диск: `network-ssd` 60 ГБ = 2 блока по 32 ГБ → лимит записи 2 × 15 МБ/с = 30 МБ/с ([docs: лимиты дисков](https://yandex.cloud/ru/docs/compute/concepts/limits#compute-limits-disks)) → ≈ 8 мин только на запись; сеть при 30–100 МБ/с — 1–3 мин параллельно. **Оценка: ≈ 12 мин (10–15) ≈ 16–20 ₽.** Ускорить: диск 96 ГБ (3 блока, 45 МБ/с, +0,72 ₽/ч) или после первого провижининга сделать снимок диска (~20 ГБ × 0,0051 = 0,10 ₽/ч) и поднимать ВМ из него — тогда старт ≈ 2 мин.

## 4. Подтверждение: в облаке ничего не создано

`final-check.txt` (снято в конце работы, 2026-09-17 11:58; там же `vpc subnet list`, `vpc address list`, `compute snapshot list`, `compute image list`, `compute filesystem list`):

```
$ yc compute instance list
+----+------+---------+--------+-------------+-------------+
| ID | NAME | ZONE ID | STATUS | EXTERNAL IP | INTERNAL IP |
+----+------+---------+--------+-------------+-------------+
+----+------+---------+--------+-------------+-------------+

$ yc compute disk list
+----+------+------+------+--------+--------------+-----------------+-------------+
| ID | NAME | SIZE | ZONE | STATUS | INSTANCE IDS | PLACEMENT GROUP | DESCRIPTION |
+----+------+------+------+--------+--------------+-----------------+-------------+
+----+------+------+------+--------+--------------+-----------------+-------------+

$ yc vpc network list
+----------------------+---------+
|          ID          |  NAME   |
+----------------------+---------+
| enpr7br554k7c8lafp0c | default |
+----------------------+---------+
```

Сеть `default` (с подсетями `default-ru-central1-{a,b,d}`) существовала до начала работы — это автосозданная сеть каталога, она бесплатна и не тронута; `lct-net`/`lct-subnet` не существуют. Также пусто: `yc vpc subnet list` (кроме default-*), `yc vpc address list`, `yc compute snapshot list`, `yc compute image list`, `yc compute filesystem list`. `./status.sh` → «платных ресурсов нет», 0 ₽/ч.

## 5. Что не удалось выяснить / допущения

1. **Зоны `ru-central1-a`/`-b` для T4/T4i** — документация явно подтверждает только `ru-central1-d` (и явно исключает `-e`); про `a`/`b` утверждений нет, `yc` списка платформ по зонам не даёт, конфиг калькулятора цен с зонами достать не удалось. Скрипты берут `ru-central1-d`; если квоту выдадут на другую зону — `ZONE=ru-central1-a ./up.sh`.
2. **Текущие квоты облака** — `yc quota-manager quota-limit list` → `PermissionDenied` (у `lct-worker` роль `editor` только на каталог). Имена квот для запроса — из документации (§1).
3. **Совместимость Gen 2-образа `ubuntu-2404-lts-secureboot-cuda-13-0` с `standard-v3-t4`** не подтверждена документацией — поэтому Ubuntu 22.04 + Python 3.12 через `uv`, а не 24.04.
4. **Точное содержимое образа** (версия драйвера до патча, набор CUDA-пакетов) видно только на Marketplace («NVIDIA Driver 535 и CUDA 12.2»); в `yc` описание образа пустое. Если в `lct-smoke.log` окажется `torch.cuda.is_available False` — план Б: `uv pip install --python /opt/reid/venv/bin/python --index-url https://download.pytorch.org/whl/cu121 "torch==2.5.1" "torchvision==0.20.1"` (сборка под CUDA 12.1; wheel'ы cp312 на индексе есть — проверено) — либо, вопреки рекомендации docs не менять драйвер, `apt install nvidia-driver-580` + перезагрузка.
5. **Цена «за ГБ в месяц» для SSD** в прайсе не публикуется — только 0,0199 ₽/ГБ·ч; месячная посчитана (14,328 ₽ за 30 дней).
6. **Пропускная способность интернета ВМ** в документации не нормирована — оценка времени cloud-init дана диапазоном.
7. Скрипты не запускались по-настоящему (запрещено) — первое реальное `./up.sh` стоит делать со `./status.sh` рядом; при любой ошибке `create` ресурсы не создаются, деньги не тратятся. Сомнительный момент, проверяемый только на практике: одновременная передача `--metadata-from-file user-data` и `--metadata serial-port-enable=1` (обе — штатные формы флагов).

## 6. Файлы в папке

`REPORT.md` (этот отчёт), `journal.md` (ход работы), `checks.txt` (66 проверок), `final-check.txt`;
скрипты: `up.sh`, `down.sh`, `status.sh`, `lct.env`, `lct-lib.sh`, `cloud-init.yaml`, `prices.json`;
данные: `std-images.json` (все 8272 образа standard-images), `image-candidates.yaml`, `docs/` (pricelist-*.json/tsv, скачанные md-страницы документации, sparse-клон `docs/yc-docs` на коммите cadf579), `yc-help/` (--help всех команд), `render-zamer/` (распакованный архив ЗАМЕР), `tests/` (заглушка yc, фикстуры, контейнерный тест и его лог), `tools/shellcheck`.

Порядок действий после получения квоты: `./status.sh` → `./up.sh` → через ~12 мин `ssh artem@$(cat lct-gpu.ip) cat /var/log/lct-smoke.log` (ждать `SMOKE_REID OK` и `SMOKE_BLENDER OK`) → работа (`scp -r render-zamer/render-zamer artem@IP:/data/render/`; команда из ЗАМЕР.md; Re-ID через `/opt/reid/venv/bin/python`) → `./down.sh` → `./status.sh` (должно быть 0 ₽/ч).
