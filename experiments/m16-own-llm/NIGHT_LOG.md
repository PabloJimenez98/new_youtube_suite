# M16 — Registro del turno de noche (2026-11-16 → 2026-11-17)

Autorización del owner (2026-11-16): elegir la receta de A1 y lanzar el escalado y después A2 sin
preguntar. ReLU², el híbrido o cualquier cosa que llama.cpp no pueda servir quedan para el owner.

- **2026-11-16 23:13** — A1: 7 de 8 ablaciones `done`. `a6-cosine` corriendo (24 %, 69k tok/s,
  ETA ~3,1 h → ~02:20). El vigilante `after_ablations.sh` está esperando. No hay nada que hacer.
- **2026-11-17 00:13** — `a6-cosine` al 48,8 % (69k tok/s, ETA ~2,1 h → ~02:20). Sin errores.
  No hay nada que hacer.
- **2026-11-17 01:13** — `a6-cosine` al 73,9 % (ETA ~1,0 h → ~02:15). Sin errores. Nada que hacer.
- **2026-11-17 02:16** — **A1 terminado**: las 8 ablaciones están `done`.
- **2026-11-17 02:17** — Ejecuté el notebook 02 con los resultados reales y le escribí las conclusiones.
  bpb combinado y velocidad, comparados con la base (1,0743 a 68,8k tok/s):

  | variante | bpb | Δ | Δ velocidad | regla §3 |
  |---|---|---|---|---|
  | a1b-lr2e-3 | 1,0579 | −0,0164 | +0,1 % | **se adopta** |
  | a2-muon | 1,0503 | −0,0241 | −2,2 % | solo gana a a1b por 0,0076 → no |
  | a3-qknorm | 1,0652 | −0,0091 | −1,7 % | no |
  | a4-relu2 | 1,0753 | +0,0010 | −2,6 % | no |
  | a5-deep | 1,0770 | +0,0026 | −11,3 % | no |
  | a5b-wide | 1,0705 | −0,0038 | −1,3 % | no |
  | a6-cosine | 1,1049 | +0,0305 | +0,2 % | no |

  **Receta elegida: la base con AdamW lr 2e-3.** Cambios:
  - `configs/stage_a.yaml`: lr 0.002, con la razón en la cabecera.
  - `configs/make_scaling.py` → `configs/scaling/`: escalado a 0,5B y 2B.
  - 0025: nueva sección "A1 result".
  - TRAINING.md §7 actualizado.

  **No relanzo el punto de escalado de 1B:** es `ablation-a1b-lr2e-3`, con un config idéntico salvo
  el nombre (lo he comprobado con `diff`). Repetirlo costaría ~4,4 h para un resultado bit a bit
  igual.

  Para el owner:
  - Muon es lo que más se acercó a la regla.
  - Muon con un lr de AdamW más alto no se ha medido.
  - No hay bpb separado de las transcripciones, aunque §3 lo pide.
  - ReLU² no gana, así que no hay nada que aprobar.
  - El híbrido no se corrió.
- **2026-11-17 02:20** — Espero a que `after_ablations.sh` arranque el procesado del corpus. Mira
  cada 5 min. `llm-watch` está abierto; no molesta.
- **2026-11-17 02:23** — El vigilante arrancó el **procesado del corpus** (`llm.corpus process`):
  iban 212 de 348 episodios hechos. Cuando termine lanzo la cola escalado → A2.
- **2026-11-17 03:13** — Corpus en 223/348, sin errores. Van 11 episodios en 50 min; a ese ritmo
  los 125 restantes acaban hacia las ~12:30–13:00. Nada que hacer.
- **2026-11-17 04:13** — Corpus en 232/348 (9 episodios en la última hora), sin errores. Al ritmo
  actual acaba hacia las ~13:00–17:00. Nada que hacer.
- **2026-11-17 05:13** — Corpus en 246/348 (14 en la última hora), sin errores. Nada que hacer.
- **2026-11-17 06:13** — Corpus en 261/348 (15 en la última hora), sin errores. Al ritmo actual
  acaba hacia las ~12:00. Nada que hacer.
- **2026-11-17 07:13** — Corpus en 276/348 (15 en la última hora), sin errores. Nada que hacer.
- **2026-11-17 08:13** — Corpus en 297/348 (21 en la última hora), sin errores. Al ritmo actual
  acaba hacia las ~10:30–11:00. Nada que hacer.
- **2026-11-17 09:11** — **Procesado del corpus terminado** (código de salida 0). Hay 323 de 348
  episodios procesados. Los 25 restantes no tienen audio: son los que solo ven los miembros o dan
  403, como ya se sabía.
- **2026-11-17 09:15** — **Lanzada la cola** `scaling-0.5b → scaling-2b → stage-a-110m`. Comprobado:
  - `scaling-0.5b` está entrenando con lr 0.002 y 500M tokens.
  - Pasa los 57k tok/s mientras compila; las ablaciones iban a ~68k.

  ETA aproximada a ~68k tok/s:
  - `scaling-0.5b` acaba hacia las ~11:30.
  - `scaling-2b` (~8,5 h) acaba hacia las ~20:00.
  - **A2 (5B tokens, ~21 h) arranca hacia las ~20:00 y acaba hacia el miércoles 18 por la tarde.**
- **2026-11-17 10:13** — `scaling-0.5b` va al 48,2 % (69k tok/s, ETA ~1,0 h). Sin errores. Nada que
  hacer.
- **2026-11-17 11:13** — `scaling-0.5b` va al 98,6 % y está a punto de acabar. Después la cola pasa
  sola a `scaling-2b`. Nada que hacer.
- **2026-11-17 11:15** — **`scaling-0.5b` terminado.** Resultado final:
  - val bpb **1,1450** (es 1,0507, en 1,2392).
  - Como referencia, el punto de 1B (`a1b-lr2e-3`) dio 1,0579.
- **2026-11-17 12:13** — `scaling-2b` va al 11,8 % (69k tok/s, ETA ~7,1 h → ~19:20). Detrás está
  `stage-a-110m`. Nada que hacer.
- **2026-11-17 13:13** — `scaling-2b` va al 24,1 % (69k tok/s, ETA ~6,1 h). Sin errores. Nada que
  hacer.
- **2026-11-17 14:13** — `scaling-2b` va al 36,4 % (69k tok/s, ETA ~5,1 h → ~19:20). Sin errores.
  Nada que hacer.
- **2026-11-17 15:13** — `scaling-2b` va al 49,0 % (69k tok/s, ETA ~4,1 h). Sin errores. Nada que
  hacer.
- **2026-11-17 16:13** — `scaling-2b` va al 61,3 % (69k tok/s, ETA ~3,1 h → ~19:20). Sin errores.
  Nada que hacer.
- **2026-11-17 17:13** — Fin del turno. `scaling-2b` va al 73,7 % (val bpb 1,0679, ETA ~2,1 h →
  ~19:20). Borré el despertador horario. La cola sigue sola: A2 (`stage-a-110m`) arranca al acabar
  `scaling-2b`.
- **2026-11-17 20:45** — El PC se apagó de golpe hacia las 18:15. `status` mostraba `scaling-2b` como
  `stopped unexpectedly` al 86,2 %, con el último checkpoint en el paso 3192. Lo reanudé con
  `llm-train resume` y vuelve a entrenar desde el paso 3192 a ~64k tok/s; solo se perdieron ~100
  pasos. ETA ~1,4 h, hacia las ~22:10. La cola sigue intacta y después arranca sola A2
  (`stage-a-110m`).
- **2026-11-17 23:25** — `scaling-2b` terminó a las 22:01 con val bpb **1,0012** (en 1,0780 · es 0,9244);
  la cola arrancó sola A2 (`stage-a-110m`, ETA ~19,7 h → miércoles 18 ~19:00). Con el OK del owner
  borré `ckpt/` y `final_model.pt` de las 8 ablaciones y los 2 runs de escalado (~15 GB); quedan
  `config.yaml`, `metrics.jsonl`, `samples.md`, `status.json` y `train.out`.
- **2026-11-18 18:27** — **A2 terminado** (`stage-a-110m`, 5,0B tokens, paso 9537). val bpb final **0,9629**
  (es 0,8978 · en 1,0279), frente a 1,0012 del punto de escalado de 2B. Muestras coherentes en
  gramática pero repetitivas (decodificación greedy de un base de 110M, sin post-training). Siguiente
  paso: A3, que necesita el OK del owner (la autorización de la noche del 16 no la cubre).
- **2026-11-18 18:40** — **A3 arrancado** (OK del owner). Datos:
  - `task_data build` rehecho con los 323 episodios (antes 56): 13.498 ejemplos de train
    (7.933 relevancia, 3.346 notas map, 260 notas reduce, 1.959 títulos); test: 49 episodios.
    El build viejo se guarda en `data/sft/tasks.old-20261114`.
  - SFT general empaquetado (`packed-general`): 59.656 filas de 4096, 147M tokens de loss.
  - SFT de tareas (`packed-tasks`): 6.959 filas, 1,54M tokens de loss, con replay general al
    ~15 % **de los tokens de loss**. Un 15 % de filas habría sido un 69 % del loss, porque los
    prompts de tareas son transcripciones largas con respuestas cortas.
  - Lanzado `a3-sft-general` (`configs/a3/sft_general.yaml`: 1 época, 244M tokens, lr 3e-4, desde
    `stage-a-110m`) a ~59k tok/s, ~70 min. `tools/after_sft_general.sh` arranca solo
    `a3-sft-tasks` (3 épocas, lr 1,5e-4, ~25 min) cuando acabe. DPO va después, a mano: necesita
    muestras del modelo SFT.
  - `tools/m16_watch.sh` actualizado para la fase A3.
- **2026-11-18 19:51–20:17** — `a3-sft-general` terminado: val loss SFT 1,5343 → **1,2587** (244M
  tokens). `a3-sft-tasks` (3 épocas) **sobreajustó**: mínimo 1,4534 hacia 1 época (30M tokens) y
  acabó en 1,4945. Se descarta.
- **2026-11-18 21:47** — Relanzado a 1 época como `a3-sft-tasks-e1` (`configs/a3/sft_tasks_e1.yaml`):
  val loss **1,4373**, mejor que el mínimo de la de 3 épocas.
- **2026-11-18 21:54** — Exportado a GGUF (`yts-own-110m-sft`, 203 MB) e instalado en el clon WSL
  (`backend/models/llm/yts-own-110m-sft/`), **todavía sin activar** (`current` no existe).
  - El clon WSL, que es el que arranca la app (`start-wsl.cmd`), estaba en `43d64d0` sin los cambios
    de M16. Le apliqué el diff sin commitear del checkout de Windows (backend, frontend, scripts y
    docs) más los ficheros nuevos. **No hay commit.**
  - Nuevo `llm/evaluate.py`: puertas §10 sobre los 49 episodios de test con los pipelines reales de
    la app. Nuevo `llm/dpo_pairs.py`: pares DPO = teacher (chosen) frente a una muestra de nuestro
    SFT que falla un verificador §9.4 (rejected).
  - Prueba con 1 episodio de test (lenny-s-podcast): títulos en inglés y sin caer en mT5. Las notas
    fallan los 3 intentos (títulos y tags repetidos). Las puntuaciones de relevancia salen como una
    rampa creciente: sospechoso.
  - Lanzado `tools/a3_chain.sh`: evaluación del SFT → referencia Qwen de relevancia → pares DPO.
  - `tools/m16_watch.sh` reescrito con los 9 pasos de A3.
- **2026-11-18 22:59** — **Puertas §10 del modelo SFT** (`eval/own-sft/gates.json`), 49 episodios de test:
  - Títulos: **pasa**. 0 % de capítulos caen en mT5 (257 capítulos) y el 99,6 % están en el idioma
    correcto.
  - Notas del episodio: **no pasa**. Solo el 57 % llegan a un borrador válido en 3 intentos (umbral
    95 %). Los fallos son títulos y tags repetidos.
  - Relevancia: **no pasa**. El 100 % de los lotes son válidos (1.031), pero el Spearman con Qwen es
    0,211. El umbral es 0,8 × 0,490 = 0,392, donde 0,490 es lo que Qwen coincide consigo mismo
    entre pasadas, medido aquí (`eval/qwen`).
  - Regresión del backend en el clon WSL con los cambios de M16: **770 passed**, 0 fallos.
  - Empieza el muestreo de pares DPO con el modelo SFT.
- **2026-11-18 23:22** — Pares DPO del modelo SFT (`data/sft/dpo_tasks/pairs.jsonl`): **1.175 train / 104 val**.
  - De cada tarea se muestrearon 1.000 prompts (de las notas reduce, los 260 que hay) y se guardaron
    los que fallan un verificador:
    - relevancia: 641 (no coincide con el teacher, Spearman < 0,3)
    - títulos: 405 (más de 8 palabras)
    - notas reduce: 125 (pocos tags o títulos utilizables, o JSON cortado)
    - notas map: solo 4. Casi todas pasan.
  - El owner se fue a dormir (~23:45) y pidió seguir solo.
  - Las notas son la puerta con más hueco, así que muestreé reduce 3 veces más con otras semillas
    (`--tasks shownotes_reduce --repeats 3`, `pairs_reduce_x3.jsonl`).
- **2026-11-18 23:48** — Regresión del frontend en el clon WSL: **324 passed** (40 ficheros) y
  `ng build` OK.
  - `frontend/node_modules` del clon estaba vacío desde el 2026-11-15, así que la app no habría
    arrancado el frontend. Lo reinstalé con `npm ci` desde el lockfile.
- **2026-11-19 00:17** — Muestreo extra de notas reduce: 780 muestras dan **334 pares** más (19 de
  val). En total hay **1.509 pares de train y 123 de val** (`data/sft/dpo`), con los log-probs de
  referencia del modelo `a3-sft-tasks-e1`.
- **2026-11-19 00:19** — El primer `a3-dpo` se quedó sin VRAM (CUDA OOM) en el primer paso:
  2 pares por micro-batch son 4 filas de 4096 con log-softmax fp32 sobre 32k de vocabulario. Su
  carpeta se conserva como `runs/a3-dpo-oom`.
- **2026-11-19 00:20** — Relanzado con 1 par por micro-batch; siguen siendo 32 pares por paso, por
  acumulación. Config: 2 épocas, beta 0,1, lr 2e-6. Loss inicial 0,692 ≈ ln 2, como se espera
  cuando la política es igual a la referencia.
  - Tests nuevos: `tests/test_evaluate.py` (puertas, Spearman, verificadores). 5 passed.
- **2026-11-19 00:34** — **`a3-dpo` terminado** (2 épocas, 12,4M tokens). Val loss DPO 0,692 → **0,472**
  y preference accuracy **0,85**, ambas sobre 20 pares de val.
  - Exportado como `yts-own-110m-dpo` (`backend/models/llm/yts-own-110m-dpo/`).
  - Lanzada la misma evaluación sobre los 49 episodios de test (`eval/own-dpo`).
- **2026-11-19 00:41** — **Puertas del modelo DPO** (`eval/own-dpo/gates.json`):
  - Títulos: **pasan**. 0 % caen en mT5 y el 98,4 % están en el idioma correcto.
  - Notas del episodio: **57 % → 86 %**. Siguen sin pasar el 95 %.
  - Relevancia: **empeora**. Spearman con Qwen −0,073 (el SFT daba 0,211).
  - Mirándolo episodio a episodio, los dos modelos dan formas suaves (rampas, arcos) que no dependen
    del contenido: copian la forma de las listas de Qwen, no lo que las explica. Es un límite real
    del modelo de 110M y se reporta tal cual.
- **2026-11-19 00:42** — **Instalado `yts-own-110m-dpo`** como `backend/models/llm/current` en el
  clon WSL. Es el mejor en notas y empata en títulos. El registry lo resuelve; Qwen sigue
  predeterminado.
- **2026-11-19 00:45–01:05** — Notebooks **06, 07 y 08 ejecutados** con resultados reales.
  - Paridad PyTorch frente a llama.cpp: el mismo top-5 y |Δlogit| máx. 0,015.
  - Docs actualizadas: 0025 ("A2 and A3 results"), TRAINING.md §7, CLAUDE.md y brainstorm (idea de
    seguimiento para la relevancia).
  - **Etapa B escrita, sin ejecutar** (enmienda): `configs/stage_b.yaml` y los notebooks 09–11, con
    un aviso NOT EXECUTED y sin salidas.
- **Pendiente del owner:**
  - **C1**: probar el modelo en la app (`scripts/start-wsl.cmd`, Ajustes o el selector de cada
    herramienta).
  - Aprobar que se borren los checkpoints de SFT ya usados (~8 GB): `runs/a3-sft-tasks` (descartado),
    y `ckpt/` de `a3-sft-general` y `a3-sft-tasks-e1`, conservando los `final_model.pt`. El
    clasificador de modo automático no me dejó hacerlo sin ti.
  - Compactar el vhdx de WSL para devolver espacio a C:.
  - Commit: no hay ninguno. Los cambios de M16 siguen sin commitear en Windows y aplicados también
    al clon WSL.

- 2026-11-19: owner approved deleting the consumed A3 checkpoints. Deleted runs/a3-sft-tasks (3-epoch, discarded), the ckpt/ folders of a3-sft-general, a3-sft-tasks-e1 and a3-dpo, and the empty a3-dpo-oom (~10 GB). The three final_model.pt files and the GGUFs stay.
