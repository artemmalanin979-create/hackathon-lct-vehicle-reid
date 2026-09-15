# 05. Скорость/бюджет весов и лицензионная чистота (vehicle re-id, коммерческий заказчик)

Дата исследования: 2026-09-15. Метод: каждое число/условие лицензии открыто через WebFetch и помечено
`[ПРОВЕРЕНО: <url>]`. Всё, что не удалось открыть, помечено `[ПО ПАМЯТИ / НЕ ПРОВЕРЕНО]` или «расчёт».
Ограничения задачи: суммарный размер весов <= 2 ГБ, офлайн-инференс, метрики: (а) мс/объект при batch=1, (б) img/s при батче.
GPU организатора неизвестен (T4 / V100 / A100 / L4 / RTX).

---

## Часть А1. Размеры весов backbone (fp32 — официальные файлы; fp16 — арифметика /2)

| Backbone | Параметры | fp32, файл | fp16 (расчёт) | Источник |
|---|---|---|---|---|
| ResNet50 (torchvision IMAGENET1K_V1/V2) | 25 557 032 | **97.8 МБ** | ~49 МБ | [ПРОВЕРЕНО: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html] |
| ResNet50-IBN (a) | в README IBN-Net параметры НЕ опубликованы; по конструкции IN заменяет часть BN, число параметров ≈ как у ResNet50 (~25.6M → ~98 МБ fp32) | ~98 МБ (расчёт) | ~49 МБ | README проверен, цифр там нет [ПРОВЕРЕНО: https://raw.githubusercontent.com/XingangPan/IBN-Net/master/README.md]; оценка параметров — [ПО ПАМЯТИ / расчёт] |
| OSNet x1.0 | 2.2M (офиц. model zoo torchreid); x0.75 — 1.3M; x0.5 — 0.6M; x0.25 — 0.2M | ~8.8 МБ (расчёт 2.2M×4Б) | ~4.4 МБ | параметры [ПРОВЕРЕНО: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html]; размер файла в zoo не указан |
| EfficientNet-B0 | 5 288 548 | **20.5 МБ** | ~10 МБ | [ПРОВЕРЕНО: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html] |
| EfficientNet-B1 | 7 794 184 | **30.1 МБ** | ~15 МБ | [ПРОВЕРЕНО: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b1.html] |
| EfficientNet-B2 | 9 109 994 | **35.2 МБ** | ~18 МБ | [ПРОВЕРЕНО: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b2.html] |
| EfficientNet-B3 | 12 233 232 | **47.2 МБ** | ~24 МБ | [ПРОВЕРЕНО: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b3.html] |
| EfficientNet-B4 | 19 341 616 | **74.5 МБ** | ~37 МБ | [ПРОВЕРЕНО: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b4.html] |
| ViT-B/16 | 86 567 656 (SWAG E2E: 86 859 496) | **330.3 МБ** (SWAG E2E: 331.4 МБ) | ~165 МБ | [ПРОВЕРЕНО: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html] |
| ViT-L/16 (torchvision, для масштаба) | 304 326 632 | **1161.0 МБ** | ~580 МБ | [ПРОВЕРЕНО: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_l_16.html] |
| ViT-L/14 (DINOv2-large, только vision) | ~304M | **1.22 ГБ** (model.safetensors) | ~610 МБ | [ПРОВЕРЕНО: https://huggingface.co/facebook/dinov2-large/tree/main] |
| ViT-L/14 (OpenAI CLIP, vision+text целиком) | — | **1.71 ГБ** (pytorch_model.bin / safetensors) | ~855 МБ | [ПРОВЕРЕНО: https://huggingface.co/openai/clip-vit-large-patch14/tree/main] |
| Swin-B | 87 768 224 | **335.4 МБ** | ~168 МБ | [ПРОВЕРЕНО: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.swin_b.html] |
| ConvNeXt-B | 88 591 464 | **338.1 МБ** | ~169 МБ | [ПРОВЕРЕНО: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_base.html] |

Справочно: timm resnet50.a1_in1k — «Params (M): 25.6, GMACs: 4.1» [ПРОВЕРЕНО: https://huggingface.co/timm/resnet50.a1_in1k].

**Вывод по бюджету 2 ГБ:** любой одиночный backbone из списка укладывается с большим запасом; даже ViT-L/14 fp32 (~1.2 ГБ vision-only)
проходит, а в fp16 (~0.6 ГБ) остаётся место под детектор/доп. головы. Проблема бюджета возникает только при ансамблях из
нескольких крупных моделей.

---

## Часть А2. Ориентиры скорости инференса (порядки величин)

### MLPerf Inference (TensorRT, ResNet-50, сценарий Offline = максимальная пропускная способность)

| GPU | Результат | Точность | Раунд | Источник |
|---|---|---|---|---|
| T4 | 44 977.8 samples/s на системе 8×T4 → **~5 622 img/s на 1×T4** | int8 (вход и веса int8) | MLPerf Inference v0.5 (2019) | лог [ПРОВЕРЕНО: https://raw.githubusercontent.com/mlcommons/inference_results_v0.5/master/closed/NVIDIA/results/T4x8/resnet/Offline/performance/run_1/mlperf_log_summary.txt]; точность int8 [ПРОВЕРЕНО: https://raw.githubusercontent.com/mlcommons/inference_results_v0.5/master/closed/NVIDIA/measurements/T4x8/resnet/Offline/T4x8_tensorrt_Offline.json] |
| L4 | **13 157.8 samples/s** (1×L4) | int8 («weight_transformations: quantization, affine fusion») | MLPerf Inference v3.0 (2023) | лог [ПРОВЕРЕНО: https://raw.githubusercontent.com/mlcommons/inference_results_v3.0/main/closed/NVIDIA/results/L4x1_TRT/resnet50/Offline/performance/run_1/mlperf_log_summary.txt]; int8 [ПРОВЕРЕНО: https://raw.githubusercontent.com/mlcommons/inference_results_v3.0/main/closed/NVIDIA/measurements/L4x1_TRT/resnet50/Offline/L4x1_TRT_Offline.json] |
| A100 SXM 80GB (1 GPU) | **42 379.3 samples/s** | int8 | MLPerf Inference v3.0 (2023) | лог [ПРОВЕРЕНО: https://raw.githubusercontent.com/mlcommons/inference_results_v3.0/main/closed/NVIDIA/results/DGX-A100_A100-SXM-80GBx1_TRT/resnet50/Offline/performance/run_1/mlperf_log_summary.txt]; int8 [ПРОВЕРЕНО: https://raw.githubusercontent.com/mlcommons/inference_results_v3.0/main/closed/NVIDIA/measurements/DGX-A100_A100-SXM-80GBx1_TRT/resnet50/Offline/DGX-A100_A100-SXM-80GBx1_TRT_Offline.json] |
| L4 vs T4 | «L4 delivered up to 3x more performance than T4 at 99.9% of the reference (FP32) accuracy of BERT» (заявление NVIDIA о соотношении поколений; для CV-порядков тоже ориентир ~2–3x) | — | v3.0 | [ПРОВЕРЕНО: https://developer.nvidia.com/blog/setting-new-records-in-mlperf-inference-v3-0-with-full-stack-optimizations-for-ai/] |

Примечание: актуальная страница NVIDIA «Inference Performance for Data Center Deep Learning» сейчас (MLPerf v6.0) содержит только LLM/генеративные
модели, ResNet-50 из неё убран [ПРОВЕРЕНО: https://developer.nvidia.com/deep-learning-performance-training-inference/ai-inference]. Архив web.archive.org из этой среды недоступен.

### PyTorch (eager/AMP), ResNet-50 v1.5 — официальные таблицы NVIDIA DeepLearningExamples
[ПРОВЕРЕНО: https://raw.githubusercontent.com/NVIDIA/DeepLearningExamples/master/PyTorch/Classification/ConvNets/resnet50v1.5/README.md]

| GPU | batch=1 | batch=256 |
|---|---|---|
| V100 16GB | FP32: 96 img/s, **10.37 мс**; AMP: 78 img/s, 12.78 мс | FP32: 1 261 img/s; AMP: **3 382 img/s** (84.14 мс/батч) |
| T4 | FP32: 98 img/s, **10.7 мс**; AMP: 79 img/s, 12.96 мс | FP32: 415 img/s; AMP: **1 198 img/s** (225.95 мс/батч) |

(При batch=1 AMP медленнее FP32 — накладные расходы каста доминируют; это честные цифры PyTorch без компиляции.)

### timm official benchmarks (AMP≈fp16, batch=1024, 224px) — ориентир для RTX-карт
| GPU | resnet50 | vit_base_patch16_224 | Источник |
|---|---|---|---|
| RTX 3090 (PyTorch 2.4.0/cu124) | **3 219 img/s** | **1 571 img/s** | [ПРОВЕРЕНО: https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/results/benchmark-infer-amp-nchw-pt240-cu124-rtx3090.csv] |
| RTX 4090 (PyTorch 2.4.0/cu124) | **4 218 img/s** | **10 113 img/s** (SDPA/fused attention даёт ViT огромный буст) | [ПРОВЕРЕНО: https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/results/benchmark-infer-amp-nchw-pt240-cu124-rtx4090.csv] |

Список всех benchmark-CSV: [ПРОВЕРЕНО: https://github.com/huggingface/pytorch-image-models/tree/main/results]

### Итоговые порядки величин для планирования
- **batch=1, PyTorch eager**: ResNet50 ≈ 5–13 мс/кроп на T4/V100 (см. таблицу NVIDIA выше — 10.4–12.9 мс); на A100/L4/RTX40xx быстрее. ViT-B при batch=1 — того же порядка или медленнее [ПО ПАМЯТИ / НЕ ПРОВЕРЕНО для bs=1].
- **batch=1, TensorRT fp16/int8**: типично ~1–2 мс для ResNet50 на T4-классе [ПО ПАМЯТИ / НЕ ПРОВЕРЕНО — прямого опубликованного замера bs=1 не нашёл; проверенная амортизированная нижняя граница из MLPerf: 1000/5622 ≈ 0.18 мс/изобр. на T4 int8 при полном батче].
- **Пропускная способность при батче**: ResNet50 — от ~1.2k img/s (T4, PyTorch AMP) и ~5.6k img/s (T4, TRT int8) до ~13k (L4, TRT int8) и ~42k (A100, TRT int8). ViT-B — ~1.5–10k img/s на RTX 3090/4090 (AMP), т.е. на «неизвестном GPU организатора» ViT-B в 2–4 раза дороже ResNet50, кроме случаев с fused attention на новых архитектурах.
- Эмбеддинг-извлечение — вычислительно та же классификационная сеть без головы, поэтому эти ориентиры переносимы на re-id инференс по кропам.

---

## Часть А3. Что даёт ONNX Runtime / TensorRT против PyTorch eager (опубликованные сравнения)

| Сравнение | Кратность | Контекст | Источник |
|---|---|---|---|
| Torch-TensorRT vs PyTorch eager | **«up to 6x»** | EfficientNet-B0, A100, FP16, batch size 1; «Torch-TensorRT accelerates the model performance up to 6x» | [ПРОВЕРЕНО: https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/] |
| ONNX Runtime (+fp16-квантизация) vs PyTorch | **«2.88x throughput gain over PyTorch»** | GPU Tesla T4, продовый NLP-трансформер Microsoft; на CPU (Xeon+VNNI, int8) — «27 percent improvement over PyTorch» | [ПРОВЕРЕНО: https://opensource.microsoft.com/blog/2022/04/19/scaling-up-pytorch-inference-serving-billions-of-daily-nlp-inferences-with-onnx-runtime/] |
| ONNX Runtime, официальная позиция | без конкретной кратности: «With ONNXRuntime, you can reduce latency and memory and increase throughput» + графовые оптимизации из коробки | официальный tutorial «Accelerate PyTorch model inference» | [ПРОВЕРЕНО: https://onnxruntime.ai/docs/tutorials/accelerate-pytorch/pytorch.html] |

Практический вывод: главный источник ускорения — не сам формат, а fp16/int8 + слияние графа (kernel fusion). Для CNN на batch=1
экспорт в TensorRT обычно даёт наибольший выигрыш (устранение launch-overhead), MLPerf-цифры выше показывают потолок int8-пути.
Кратность сильно зависит от модели/батча/GPU — обещать заказчику стоит «2–6x к eager fp32», ссылаясь на две первые строки таблицы.

---

## Часть Б4. Лицензии/условия доступа датасетов vehicle re-id

| Датасет | Как получают | Условия (цитаты) | Источник |
|---|---|---|---|
| **VeRi-776** | письмо с ФИО и аффилиацией на xinchenliu at bupt dot cn: «Please email your full name and affiliation to the contact person» | «We ask for your information only to make sure the dataset is used for **non-commercial purposes**»; «We will not give it to any third party or publish it publicly anywhere» | [ПРОВЕРЕНО: https://raw.githubusercontent.com/JDAI-CV/VeRidataset/master/README.md] и [ПРОВЕРЕНО: https://vehiclereid.github.io/VeRi/] |
| **VehicleID (PKU)** | скачать и подписать agreement, отправить с академической почты на pkuml at pku.edu.cn; «Requests from free email addresses (outlook, gmail, qq etc) will be kindly refused» | «The images and the corresponding annotation results can only be used for **ACADEMIC PURPOSES. NO COMERCIAL USE is allowed**» (орфография оригинала); обязательное цитирование CVPR'16 | [ПРОВЕРЕНО: https://pkuml.org/resources/pku-vehicleid.html] |
| **VERI-Wild** | письмо с ФИО и аффилиацией на yanbai at pku dot edu dot cn | «We ask for your information only to make sure the dataset is used for **non-commercial purposes**»; «If you download our dataset, it means you have agreed to our terms of access in the email» | [ПРОВЕРЕНО: https://raw.githubusercontent.com/PKU-IMRE/VERI-Wild/master/README.md] |
| **CityFlow / AI City Challenge** (данные NVIDIA) | онлайн-форма запроса + принятие «DATASET LICENSE AGREEMENT» на каждый трек | П.1(ii): после челленджа данные и модели можно использовать «for **non-commercial, academic purposes only**». П.2(a): «**You may not use the DATASET or any models developed using the DATASET for any commercial or production purpose**». П.2(c): нельзя «copy, sell, rent, sublicense, transfer or distribute the dataset, or share with others». Собственность NVIDIA (п.3) | процесс [ПРОВЕРЕНО: https://www.aicitychallenge.org/2024-data-and-evaluation/]; текст соглашения (v. February 9, 2022) [ПРОВЕРЕНО: http://www.aicitychallenge.org/wp-content/uploads/2022/02/Dataset-License-AIC2022.pdf — PDF прочитан целиком] |
| **VehicleX** (синтетика, ANU) | открытый GitHub-репозиторий (данные/Unity-проект/fbx-модели) | Файла LICENSE в репозитории **нет** (GitHub API /license → 404 [ПРОВЕРЕНО: https://api.github.com/repos/yorkeyao/VehicleX/license]). В README только неформальное: «**You are welcomed to use every single part of the code for your research purpose**» — формальная лицензия не задекларирована, статус для коммерции не определён (нужен запрос авторам) | [ПРОВЕРЕНО: https://raw.githubusercontent.com/yorkeyao/VehicleX/master/README.md] |

**Итог по датасетам:** все реальные vehicle re-id датасеты (VeRi-776, VehicleID, VERI-Wild, CityFlow) — research/non-commercial;
у CityFlow запрет прямо распространён и на **модели**, обученные на данных. VehicleX — де-факто открыт, но без формальной лицензии.

---

## Часть Б5. Лицензии предобученных весов

| Веса | Лицензия | Детали/цитаты | Источник |
|---|---|---|---|
| torchvision (код + веса ImageNet) | BSD-3-Clause (репозиторий pytorch/vision) | Оговорка в доках: «**The pre-trained models provided in this library may have their own licenses or terms and conditions derived from the dataset used for training. It is your responsibility to determine whether you have permission to use the models for your use case.**» | [ПРОВЕРЕНО: https://api.github.com/repos/pytorch/vision/license]; оговорка [ПРОВЕРЕНО: https://docs.pytorch.org/vision/stable/models.html] |
| torchvision ViT SWAG-веса (IMAGENET1K_SWAG_*) | **CC-BY-NC 4.0 — НЕкоммерческая!** | LICENSE facebookresearch/SWAG: «Creative Commons Attribution-NonCommercial 4.0 International Public License»; ссылка на неё стоит прямо в карточке vit_b_16 | [ПРОВЕРЕНО: https://raw.githubusercontent.com/facebookresearch/SWAG/main/LICENSE]; [ПРОВЕРЕНО: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html] |
| timm (веса на HF) | apache-2.0 (тег на карточках) | пример: timm/resnet50.a1_in1k — «License: apache-2.0», «Trained on ImageNet-1k» (у отдельных моделей timm лицензия может отличаться — проверять тег карточки) | [ПРОВЕРЕНО: https://huggingface.co/timm/resnet50.a1_in1k] |
| fast-reid (JDAI-CV) | Apache-2.0 (репозиторий) | Model Zoo содержит готовые веса VeRi / VehicleID / VERI-Wild (SBS(R50-ibn), BoT(R50-ibn)) — обучены на research-only датасетах; отдельной лицензии на веса в MODEL_ZOO.md нет | [ПРОВЕРЕНО: https://api.github.com/repos/JDAI-CV/fast-reid/license]; [ПРОВЕРЕНО: https://raw.githubusercontent.com/JDAI-CV/fast-reid/master/MODEL_ZOO.md] |
| torchreid / OSNet | MIT (репозиторий KaiyangZhou/deep-person-reid) | веса zoo обучены на Market-1501/Duke/MSMT17 (person re-id, research-only) и ImageNet | [ПРОВЕРЕНО: https://api.github.com/repos/KaiyangZhou/deep-person-reid/license] |
| DINOv2 | **Apache-2.0** (репозиторий + HF-карточка) | facebookresearch/dinov2 → «Apache License 2.0»; facebook/dinov2-large → «apache-2.0» | [ПРОВЕРЕНО: https://api.github.com/repos/facebookresearch/dinov2/license]; [ПРОВЕРЕНО: https://huggingface.co/facebook/dinov2-large] |
| DINOv3 | **«DINOv3 License»** (спец-лицензия Meta), доступ на HF gated (нужно принять условия и передать контактные данные) | Грант: «You are granted a non-exclusive, worldwide, non-transferable and royalty-free limited license … to use, reproduce, distribute, copy, create derivative works of, and make modifications to the DINO Materials» — **коммерческое использование разрешено**, research-only ограничения нет. Запреты: «not to use … DINO Materials for any activities subject to ITAR or end uses prohibited by Trade Controls, including those related to military or warfare purposes, nuclear industries or applications, espionage, or the development or use of guns or illegal weapons»; обязателен acknowledgement в публикациях | [ПРОВЕРЕНО: https://ai.meta.com/resources/models-and-libraries/dinov3-license/]; gated-доступ [ПРОВЕРЕНО: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m] |
| CLIP (OpenAI) | **MIT** («MIT License, Copyright (c) 2021 OpenAI») | | [ПРОВЕРЕНО: https://raw.githubusercontent.com/openai/CLIP/main/LICENSE] |
| SigLIP | apache-2.0 (HF-карточка google/siglip-base-patch16-224) | | [ПРОВЕРЕНО: https://huggingface.co/google/siglip-base-patch16-224] |
| SigLIP 2 | apache-2.0 (HF-карточка google/siglip2-base-patch16-224) | | [ПРОВЕРЕНО: https://huggingface.co/google/siglip2-base-patch16-224] |
| PaddlePaddle (PaddleClas и веса) | Apache-2.0 (репозиторий) | | [ПРОВЕРЕНО: https://api.github.com/repos/PaddlePaddle/PaddleClas/license] |
| ImageNet (данные, на которых обучены ImageNet-претрейны) | terms of access | «**Researcher shall use the Database only for non-commercial research and educational purposes**»; «If Researcher is employed by a for-profit, commercial entity, Researcher's employer shall also be bound by these terms and conditions» | [ПРОВЕРЕНО: https://image-net.org/download.php] |

---

## Часть Б6. «Веса, обученные на research-only датасете» и коммерческое использование

Что удалось установить по первоисточникам (без домыслов):

1. **Есть случаи, где запрет на модели прописан явно.** Лицензия данных AI City Challenge / NVIDIA:
   «You may not use the DATASET **or any models developed using the DATASET** for any commercial or production purpose»
   [ПРОВЕРЕНО: http://www.aicitychallenge.org/wp-content/uploads/2022/02/Dataset-License-AIC2022.pdf]. Для подписанта соглашения это
   не серая зона: модель, обученная на CityFlow, в коммерческий продукт нельзя.

2. **У VeRi-776 / VehicleID / VERI-Wild формулировки касаются датасета (изображений и разметки), про веса моделей в открытых
   текстах прямо не сказано.** VeRi/VERI-Wild ограничиваются «used for non-commercial purposes» [ПРОВЕРЕНО: README выше];
   VehicleID — «The images and the corresponding annotation results can only be used for ACADEMIC PURPOSES»
   [ПРОВЕРЕНО: https://pkuml.org/resources/pku-vehicleid.html]. Распространяется ли ограничение на обученные веса как «производную» — в
   доступных публичных текстах не определено. Полный текст подписываемых agreement (PDF по запросу) может содержать доп. условия — его
   надо читать перед подписанием.

3. **Индустриальная практика противоречива и никем официально не разрешена.** ImageNet-претрейны повсеместно используются в
   коммерции, хотя terms ImageNet требуют «non-commercial research and educational purposes» [ПРОВЕРЕНО: https://image-net.org/download.php].
   PyTorch при этом раздаёт такие веса под BSD/Apache, но с явным перекладыванием ответственности: «It is your responsibility to
   determine whether you have permission to use the models for your use case» [ПРОВЕРЕНО: https://docs.pytorch.org/vision/stable/models.html].

4. **Правовая наука фиксирует неопределённость.** Henderson, Li, Jurafsky, Hashimoto, Lemley, Liang, «Foundation Models and Fair Use»
   (arXiv:2303.15715): «fair use is not guaranteed, and additional work may be necessary to keep model development and deployment
   squarely in the realm of fair use» [ПРОВЕРЕНО: https://arxiv.org/abs/2303.15715]. Важно: fair use — это про копирайт (США); в нашем
   случае сильнее контрактный слой — подписанное соглашение с автором датасета связывает подписанта независимо от копирайтных доктрин.

**Вывод: серая зона, нужен юрист.** Однозначного общепринятого ответа «веса свободны от условий датасета» не существует; для CityFlow
запрет прямой, для VeRi/VehicleID/VERI-Wild — контрактный риск + риск для репутации у авторов, собирающих аффилиации именно чтобы
контролировать non-commercial использование.

### Практические рекомендации для коммерческого решения
- **Не брать готовые веса fast-reid model zoo (VeRi/VehicleID/VERI-Wild)** и не файнтюниться на этих датасетах для прода без письменного разрешения авторов/юридического заключения. Код fast-reid (Apache-2.0) использовать можно.
- Чистые базы для эмбеддера: **DINOv2 (Apache-2.0), DINOv3 (коммерция разрешена лицензией Meta, с запретами ITAR/military), CLIP (MIT), SigLIP/SigLIP2 (Apache-2.0), Paddle (Apache-2.0)** — дообучать на собственных/лицензированных/синтетических кропах.
- Осторожно с «подводными камнями» внутри открытых библиотек: SWAG-веса ViT в torchvision — **CC-BY-NC 4.0** (некоммерческие).
- ImageNet-претрейн — общепринят, но формально terms non-commercial; для консервативного заказчика — использовать веса, обученные на данных с явной лицензией, либо зафиксировать риск в юр. заключении.
- VehicleX (синтетика) фактически открыт, но без LICENSE-файла — для прода запросить у авторов явное разрешение.

---

## Быстрые ссылки (все открывались 2026-09-15)
- torchvision model pages: https://docs.pytorch.org/vision/stable/models.html (+generated-страницы моделей)
- timm benchmarks: https://github.com/huggingface/pytorch-image-models/tree/main/results
- MLPerf raw: https://github.com/mlcommons/inference_results_v0.5 , https://github.com/mlcommons/inference_results_v3.0
- NVIDIA DeepLearningExamples ResNet50: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5
- Torch-TensorRT 6x: https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/
- ORT 2.88x: https://opensource.microsoft.com/blog/2022/04/19/scaling-up-pytorch-inference-serving-billions-of-daily-nlp-inferences-with-onnx-runtime/
- AI City license PDF: http://www.aicitychallenge.org/wp-content/uploads/2022/02/Dataset-License-AIC2022.pdf
- DINOv3 license: https://ai.meta.com/resources/models-and-libraries/dinov3-license/
- ImageNet terms: https://image-net.org/download.php
