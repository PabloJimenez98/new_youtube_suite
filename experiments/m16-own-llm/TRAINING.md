# M16 — Cómo entrenar, pausar y reanudar nuestro LLM

Guía práctica del entrenamiento del LLM propio (M16). El diseño completo y el porqué de cada
decisión están en `docs/decisions/0025-own-llm-m16.md`. Los experimentos y sus resultados están en
los notebooks de esta carpeta.

---

## 1. Lo esencial: parar y seguir

| Quiero... | Desde Windows (doble clic o terminal) | Desde WSL (`experiments/m16-own-llm/`) |
|---|---|---|
| **Pausar** (liberar la GPU) | `scripts\llm-pause.cmd` | `python -m llm.train pause` |
| **Reanudar** donde lo dejé | `scripts\llm-resume.cmd` | `python -m llm.train resume` |
| **Ver cómo va** | `scripts\llm-status.cmd` | `python -m llm.train status` |
| **Verlo todo en vivo** (entrenamiento, GPU, descargas, procesado) | `scripts\llm-watch.cmd` | `watch -n 10 -t bash tools/m16_watch.sh` |
| Empezar un entrenamiento nuevo | `scripts\llm-train.cmd start configs\stage_a.yaml` | `python -m llm.train start configs/stage_a.yaml` |

En WSL, `python` es el del venv de experimentos: `~/projects/new_youtube_suite/experiments/.venv/bin/python`.
Los `.cmd` de Windows ya lo usan.

**Pausar:**
- El entrenamiento termina el micro-lote en curso, como mucho un par de segundos.
- Si estaba a mitad de un paso de optimización, descarta ese paso a medias.
- Guarda un checkpoint y sale. Al salir el proceso se libera **toda** la VRAM.
- `pause` espera a que termine y te dice "GPU is free".

**Reanudar:**
- Carga el último checkpoint y sigue **exactamente** donde se quedó: los mismos pesos, el mismo estado
  del optimizador, la misma posición en los datos y el mismo estado aleatorio.
- El resultado final es idéntico bit a bit al de un entrenamiento sin pausas. Hay un test que lo
  comprueba (`tests/test_resume.py`).
- Si el paso se había descartado a medias, se rehace desde el principio con los mismos datos.

Puedes pausar y reanudar tantas veces como quieras. Una pausa no cuesta nada salvo el tiempo
de recompilación al volver (~20–80 s).

---

## 2. ¿Puedo usar el PC mientras entrena?

| Actividad | ¿Hace falta pausar? |
|---|---|
| Navegar, YouTube, vídeo | **No.** Ocupa poca VRAM y el entrenamiento va un poco más lento. |
| Jugar (Persona 5, etc.) | **Sí.** El juego necesita VRAM y la GPU entera. Pausa, juega y reanuda. |
| Usar la app (procesar un episodio) | **Sí.** El worker usa Whisper, pyannote y Qwen en la GPU. Pausa, procesa y reanuda. |
| Apagar o reiniciar el PC | Mejor pausar antes. Si no, ver §3. |

`start` y `resume` se niegan a arrancar si hay menos de **9,5 GiB de VRAM libre**: es la señal de que
un juego o el worker siguen usando la GPU, y el mensaje lo dice. Cierra lo que la esté usando y
vuelve a intentarlo. `--force` salta la comprobación (no recomendado).

El entrenamiento limita su memoria al 92 % de la VRAM. Así, si algo le quita memoria, falla con un
error claro en vez de ir 20 veces más lento desbordando a la RAM del sistema, que es lo que pasó en
la primera ejecución del benchmark (notebook 00).

---

## 3. Cortes inesperados (luz, reinicio de Windows, WSL cerrado)

- Se guarda un checkpoint **cada 15 minutos**, así que como mucho se pierden 15 minutos.
- Los checkpoints se escriben de forma atómica: primero un fichero temporal y luego se renombra.
  Un corte a mitad de escritura nunca deja un checkpoint roto, y el anterior sigue siendo válido.
- Después de un corte, `status` muestra
  `stopped unexpectedly (crash, reboot or power cut) -> python -m llm.train resume`.
  Basta con reanudar.
- **Desactiva la suspensión de Windows** mientras entrena, en Configuración → Sistema → Energía. El
  entrenamiento no puede impedir que el PC se duerma.

---

## 4. Qué muestra `status`

```
run      : stage-a-110m  (/home/pablo/yts-experiments/m16-llm/runs/stage-a-110m)
state    : running  (pid 12345)
progress : step 1520 · 0.797B / 5.000B tokens (15.9 %)
speed    : 58,900 tok/s · ETA 19.8 h of training
loss     : 3.1842 · val bpb 1.0412
last ckpt: step_00001500.pt · updated 2026-11-20 18:03:11
samples  : /home/pablo/yts-experiments/m16-llm/runs/stage-a-110m/samples.md
```

- **state**: `running`, `pausing...`, `paused`, `done`, `crashed` (con el error) o
  `stopped unexpectedly`.
- **ETA**: horas de entrenamiento que faltan, sin contar las pausas.
- **val bpb**: bits por byte sobre textos que el modelo no ve al entrenar. Cuanto más bajo, mejor. Se
  mide cada 250M tokens.
- **samples.md**: lo que el modelo escribe con unas frases fijas en cada evaluación. Sirve para ver
  cómo aprende a escribir.

---

## 5. Dónde está cada cosa

Todo lo grande vive en WSL (ext4), nunca en el repo:

```
~/yts-experiments/m16-llm/
  data/raw/{es,en}/         textos descargados (FineWeb2-HQ español, FineWeb-Edu inglés)
  data/shards/{es,en}/      textos ya tokenizados (lo que lee el entrenamiento)
  tokenizer/tokenizer.json  nuestro tokenizador
  runs/<nombre>/            un entrenamiento (ver abajo)
  runs/latest               apunta al último entrenamiento lanzado
  logs/                     logs de descarga y tokenización
~/yts-data/podcasts/m16-corpus/   audio y transcripciones del corpus de tareas
```

Dentro de un entrenamiento (`runs/<nombre>/`):

| Fichero | Qué es |
|---|---|
| `config.yaml` | La configuración del entrenamiento |
| `ckpt/step_*.pt` | Checkpoints completos (se guardan los 2 últimos) |
| `ckpt/permanent/model_XXXpct.pt` | Pesos guardados cada 10 % del entrenamiento, para análisis |
| `final_model.pt` | Los pesos finales |
| `metrics.jsonl` | Pérdida, learning rate, velocidad y evaluaciones (lo leen los notebooks) |
| `samples.md` | Textos de muestra en cada evaluación |
| `status.json` | Lo que muestra `status` |
| `train.out` | La salida del proceso. Aquí está el error si algo falla. |
| `PAUSE` | Existe mientras hay una pausa pedida |

El entrenamiento corre en segundo plano, así que cerrar la terminal no lo para. Para verlo en
primer plano, añade `--foreground` a `start` o `resume`. Entonces **Ctrl+C pausa**, igual que
`pause`.

---

## 5b. Espacio en disco y limpieza automática

Lo intermedio se borra solo en cuanto el paso siguiente lo ha consumido (decisión tuya del
2026-11-14):

| Qué | Se borra cuando... | Para conservarlo |
|---|---|---|
| Texto descargado (`data/raw/`) | cada fichero, en cuanto está tokenizado | `prepare_data tokenize --keep-raw` |
| Muestra del tokenizador | en cuanto el tokenizador está entrenado | `prepare_data train-tokenizer --keep-sample` |
| Audio del corpus | en cuanto el episodio está transcrito y procesado | `corpus process --keep-audio` |
| Checkpoints de las variantes A1 | sus configs guardan solo el último y ninguna copia permanente | — |

Lo que se conserva: shards, tokenizador, transcripciones, salidas del profesor, checkpoints de las
etapas A y B, y los GGUF. El pico estimado es de ~150 GB y quedan ~110 GB de forma estable.

**Importante:** el disco de WSL es un fichero (`%LOCALAPPDATA%\wsl\{...}\ext4.vhdx`) que vive en
**C:**. Crece al escribir y **no encoge solo** al borrar. Para devolver el espacio a C:, haz una de
estas dos cosas desde PowerShell, con WSL apagado:

```powershell
wsl --shutdown
wsl --manage Ubuntu-24.04 --set-sparse true   # a partir de ahora el espacio liberado vuelve a C: solo
```

Antes de apagar WSL, pausa el entrenamiento (`llm-pause`) y espera a que terminen las descargas o
el procesado.

---

## 5c. Procesado del corpus (transcripción + profesor Qwen)

`python -m llm.corpus process --wait` transcribe cada episodio con Whisper y ejecuta las tres
herramientas de la app con Qwen: capítulos con títulos, notas del episodio y relevancia. Cada
llamada a Qwen queda grabada en `processed/<id>/calls.jsonl`; son los ejemplos del SFT de tareas.

- Tarda ~3,3 s por minuto de audio, unas **22 h** para las ~414 h del corpus. Usa la GPU.
- **Parar:** `touch ~/yts-data/podcasts/m16-corpus/PAUSE` (para tras el episodio en curso) o Ctrl+C
  / `kill` (una vez: tras el episodio; dos veces: inmediatamente). Reanudar es volver a lanzar el
  mismo comando, después de borrar el fichero `PAUSE` si lo usaste. Los episodios terminados se
  saltan.
- No lo ejecutes a la vez que un entrenamiento: los dos usan la GPU.
- Log: `~/yts-experiments/m16-llm/logs/corpus_process.log`.

---

## 6. Cambiar cosas entre una pausa y la reanudación

Mientras está pausado, se puede editar `runs/<nombre>/config.yaml`, pero **solo** estos campos:

`total_tokens`, `eval_every_tokens`, `eval_batches`, `autosave_minutes`, `log_every_steps`,
`sample_prompts`, `sample_tokens`, `min_free_vram_gib`, `keep_last`, `permanent_every_fraction`.

- Si cambias cualquier otro campo (modelo, datos, learning rate...), `resume` se niega y te dice
  qué campo cambió. El checkpoint ya no correspondería a esa configuración.
- **Alargar un entrenamiento:** sube `total_tokens`. Hazlo antes de que empiece la fase final, el
  último 20 % (`decay_fraction`), en la que el learning rate baja a 0. El calendario WSD está
  pensado para eso: la fase intermedia es plana, y alargarla no obliga a empezar de cero.
- Un entrenamiento `done` también se puede alargar subiendo `total_tokens`, aunque el learning rate
  ya habrá bajado a 0. Para la etapa B, decide el presupuesto antes de llegar al 80 %.

---

## 7. Pasos de M16 y estado actual

| Paso | Qué | Comando o notebook | Estado |
|---|---|---|---|
| A0 | Descarga del texto de preentrenamiento | `HF_HUB_DISABLE_XET=1 python -m llm.prepare_data fetch --source es --workers 1` (§9) | hecho: inglés 46 GB, español 37,8 GB. El español extra para la etapa B (bajado el 2026-11-15) se borró al salir B del alcance |
| A0 | Muestra y tokenizador | `prepare_data sample`, después `prepare_data train-tokenizer` (notebook 01) | hecho: 7 % más eficiente que Qwen en español |
| A0 | Tokenización a shards | `prepare_data tokenize` | hecho: 8,24B es + 10,56B en (sobra para la etapa A) |
| A0 | Corpus de tareas: selección y descarga | `python -m llm.corpus select` (374 h, hecho), después `tools/download_guard.sh` (§9) | hecho: 284 de 309 (los otros 25 son solo para miembros o dan 403) |
| A0 | Transcripción y salidas del profesor (Qwen) | `python -m llm.corpus process --wait` (§5c; notebook 03 analiza el resultado) | 212 hechos; los 111 que faltan arrancan solos al acabar A1 (`tools/after_ablations.sh`) |
| A1 | Comparación de variantes a ~110M | `llm-train start configs\ablations\*.yaml` como cola (notebook 02) | **hecho** (2026-11-17 02:17). Gana AdamW lr 2e-3 (−0,0164 bpb). Muon se queda a 0,0076 de él y no se adopta. Detalle en notebook 02 y 0025 |
| A1 | Puntos de escalado (0,5B y 2B; el de 1B es `a1b-lr2e-3`) | `llm-train start configs\scaling\scaling-0.5b.yaml configs\scaling\scaling-2b.yaml configs\stage_a.yaml` | **hecho** (2026-11-17): bpb 1,1450 (0,5B), 1,0579 (1B), 1,0012 (2B) |
| A2 | Preentrenamiento de la etapa A (~110M, 5B tokens, ~1 día) | `llm-train start configs\stage_a.yaml` | **hecho** (2026-11-18 18:27): val bpb 0,9629 (es 0,898 · en 1,028). Receta: AdamW lr 2e-3 |
| A3 | Post-entrenamiento, GGUF y selector en la app → **C1**: lo pruebas en la app | `configs/a3/*.yaml`, `python -m llm.dpo_pairs`, `python -m llm.evaluate` (notebooks 05–08) | **hecho** (2026-11-19): SFT general → SFT de tareas (1 época) → DPO. Instalado `yts-own-110m-dpo` en `backend/models/llm/current`. Puertas: títulos pasan; notas 86 % (falla, umbral 95 %); relevancia falla (Spearman −0,07). **Falta C1** |
| B0 | Decidir tamaño y presupuesto de la etapa B → **C2**: lo apruebas tú | — | **fuera de alcance** (enmienda 0025, 2026-11-15) |
| B1 | Preentrenamiento de la etapa B (~330M, 20–30B tokens, ~2–3 semanas) | `llm-train start configs\stage_b.yaml` | **fuera de alcance**: config y notebooks escritos pero no ejecutados |
| B2 | Post-entrenamiento y evaluación → **C3**: revisión a ciegas | notebooks 05–07 | **fuera de alcance** |

Qwen sigue siendo el modelo por defecto de la app. Sin la etapa B no hay revisión a ciegas (C3), así
que nuestro modelo se puede elegir (en Ajustes o al lanzar cada herramienta) pero no pasa a ser el predeterminado.

---

## 8. Problemas frecuentes

| Síntoma | Qué hacer |
|---|---|
| `Only X GiB of VRAM free` | Algo usa la GPU (un juego, el worker de la app, otro entrenamiento). Ciérralo y reintenta. |
| `config.yaml no longer matches the checkpoint` | Cambiaste un campo no editable (§6). Deshaz el cambio. |
| `already running` | Ya está entrenando. Mira `status`. |
| `state: crashed` | El error está en `status` y en `train.out`. Normalmente basta con `resume` cuando se arregla la causa. |
| `stopped unexpectedly` | Hubo un corte o un reinicio. `resume`. |
| La velocidad (tok/s) cae mucho | Algo más está usando la GPU. Si persiste, pausa y reanuda. |
| `pause` tarda más de un minuto | Está escribiendo un checkpoint grande. Espera y mira `status`. |

---

## 9. Red: descargas sin dejarte sin internet

El 2026-11-14 las descargas en paralelo saturaron la NAT IPv4 de tu operador (CGNAT). Durante un
rato **todo el PC** se quedó sin conexiones IPv4 nuevas, aunque el IPv6 y el ping seguían
funcionando. Desde entonces:

- **WSL usa red en modo espejo** (`C:\Users\Pablo\.wslconfig`). Para volver al modo anterior, borra
  el fichero y ejecuta `wsl --shutdown`.
- **Texto de Hugging Face:** una sola conexión.
  `HF_HUB_DISABLE_XET=1 python -m llm.prepare_data fetch --source es --workers 1`
  Mejor de noche. Unos 3,5 MB/s, así que los ~57 GB que faltan llevan unas 4–5 h.
- **Audio de YouTube:** `tools/download_guard.sh` descarga por IPv4 con una sola conexión y se
  **detiene solo** si el IPv4 falla 3 minutos seguidos. Log: `logs/download_guard.log`.
- **Si notas internet lento:** para las descargas con
  `pkill -f 'prepare_data fetc[h]'; pkill -f 'yt-dl[p]'`. La NAT del operador se vacía sola en unos
  minutos.
- **Tests del backend mientras el IPv4 esté inestable:** ejecútalos con `HF_HUB_OFFLINE=1`, porque
  algunos tests cargan modelos de HF y se cuelgan sin red.
