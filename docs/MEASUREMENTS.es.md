# Mediciones: Ryzen 7 5700X + RTX 3090 + RTX 5060 Ti (Linux)

Velocidades de esta rama (`custom-plus`) con UD-IQ4_XS y UD-Q4_K_XL de unsloth, con una gráfica y con dos. Las
configuraciones están en [examples/r7-5700x](../examples/r7-5700x/). English version: [MEASUREMENTS.md](MEASUREMENTS.md).

## El equipo

| | |
|---|---|
| CPU | AMD Ryzen 7 5700X: 8 núcleos / 16 hilos, Zen 3, AVX2 (sin AVX-512) |
| RAM | 128 GB DDR4-3200 CL22 (4 x 32 GB), ~36 GB/s de lectura (STREAM) |
| Gráfica principal | NVIDIA RTX 3090 24 GB, PCIe 4.0 x8, límite de potencia 250 W |
| Segunda gráfica (nivel de expertos) | NVIDIA RTX 5060 Ti 16 GB, PCIe 4.0 x8, límite de potencia 150 W |
| Software | Linux (kernel 7.2), driver NVIDIA 615.71, CUDA 13.4, motor compilado con `CMAKE_CUDA_ARCHITECTURES=86;120` |

Con las gráficas a su potencia de serie (350 W y 180 W) las velocidades fueron las mismas (UD-Q4_K_XL: 63,5 -> 63,8
tok/s con la 3090, 80,9 -> 80,3 con las dos): al motor lo limitan la CPU, la RAM y el PCIe, no la potencia.

## Los modelos

Qwen3.8-Flash-Next (48 capas x 512 expertos), ficheros GGUF de unsloth. Los expertos viven en RAM y el motor guarda
en VRAM los más usados (una caché adaptativa); la CPU calcula el resto.

| | UD-IQ4_XS | UD-Q4_K_XL |
|---|---|---|
| Ficheros GGUF | 3 | 4 |
| Pesos de los expertos | 55,4 GiB | 71,7 GiB |
| Formatos de los expertos (gate/up · down) | IQ3_S (47 capas), IQ4_XS (1) · IQ4_NL (43), Q8_0 (5) | Q4_K (47), Q5_K (1) · Q5_1 (43), Q8_0 (5) |
| RAM que ocupa el motor | ~58 GiB | ~81 GB |
| Kernels de CPU de los expertos | limitados por cálculo: escalan con los núcleos | limitados por ancho de banda: la DDR4 se satura con ~4 núcleos |

En los dos: 200.192 tokens de contexto, caché KV int8, la capa de borrador MTP con vocabulario español
(`data/draft_vocab_es.bin`) y búsqueda en el prompt, verificación greedy (la especulación nunca cambia la salida).
Estas cifras se midieron con el subconjunto español de 94.962 ids que se publicaba antes; el de 47.196 ids que se
publica ahora midió lo mismo con UD-IQ4_XS (74,8 % frente a 75,1 % de borradores en español aceptados, 90,4 frente a
89,5 tok/s) con la mitad de cabeza de borrador.

## Ajustes

| | UD-IQ4_XS 1 GPU | UD-IQ4_XS 2 GPU | UD-Q4_K_XL 1 GPU | UD-Q4_K_XL 2 GPU |
|---|---:|---:|---:|---:|
| `--pool-workers` | 7 | 7 | 4 | 4 |
| `--adapt-swaps` | 192 | 192 | 96 | 96 |
| `--adapt-decay` | 0,92 | 0,7 | 0,7 | 0,7 |
| `STRATA_IQ256_GATHER` | 1 | 1 | - | - |
| `--pcie-frac` (kernel) | 0 | 0 | 0,3 | 0 |
| Segunda GPU | - | `--second-gpu 0`, reserva 512 MiB, prefetch 6 MB, `--head-split 0.45` | - | igual que IQ4_XS |
| `--kv-grow` | sí | sí | sí | sí |
| `STRATA_POOL_FUSED` | 1 | 1 | 1 | 1 |

## Cómo se ha medido

- El motor directamente por su protocolo `--serve`, sin caché de prompt (cada prompt se lee entero), greedy, con
  especulación (`--spec 4 --spec-lookup 16`).
- Seis textos fijos, casi todos en español: `es_chat` (pregunta de 71 tokens, 400 de respuesta), `es_think` (133
  tokens, 700 de respuesta, con razonamiento), `es_doc` (un documento de 18.076 tokens y 400 de respuesta), `code`
  (62 tokens, 600 de respuesta), `edit` (un script de 3.340 tokens devuelto editado, ~2.600), `proto5k` (un prompt
  de 5.296 tokens, 256 de respuesta). Los textos largos son notas privadas y no se publican.
- Velocidad de generación = media geométrica de los seis; cada configuración es la media de dos rondas (UD-IQ4_XS
  1 GPU 93,0 / 92,8, 2 GPU 116,8 / 115,7; UD-Q4_K_XL 71,0 / 71,4 y 85,4 / 84,3).
- Sesiones de agente: 18 turnos a través del servidor de Niko1221/Strata, con la conversación creciendo hasta ~31K
  tokens (ficheros de código pegados, documentación, cambios de tema), 600 tokens por respuesta.

## Resultados

Velocidad de generación (tokens/s):

| Texto | UD-IQ4_XS 1 GPU | UD-IQ4_XS 2 GPU | UD-Q4_K_XL 1 GPU | UD-Q4_K_XL 2 GPU |
|---|---:|---:|---:|---:|
| es_chat | 89,9 | 116,7 | 70,2 | 88,8 |
| es_think | 97,5 | 122,9 | 83,7 | 86,4 |
| es_doc (tras 18K) | 79,5 | 101,8 | 59,3 | 77,9 |
| code | 99,1 | 129,4 | 74,8 | 89,9 |
| edit | 132,8 | 183,2 | 97,0 | 129,2 |
| proto5k (tras 5K) | 70,0 | 71,4 | 51,7 | 53,8 |
| **Media geométrica** | **92,9** | **116,3** | **71,2** | **84,9** |
| (antes de `--kv-grow` y el pool fusionado) | 83,6 | 109,4 | 64,2 | 80,5 |

Lectura del prompt (prefill, tokens/s):

| Prompt | UD-IQ4_XS 1 GPU | UD-IQ4_XS 2 GPU | UD-Q4_K_XL 1 GPU | UD-Q4_K_XL 2 GPU |
|---|---:|---:|---:|---:|
| 18.076 tokens | 1.803 | 1.836 | 1.730 | 1.758 |
| 5.296 tokens | 1.180 | 1.550 | 899 | 1.113 |
| 3.340 tokens | 851 | 1.086 | 660 | 794 |

- La segunda gráfica suma **+25 %** a UD-IQ4_XS y **+19 %** a UD-Q4_K_XL: guarda ~14,5 GB más de expertos y calcula su
  parte de cada capa mientras la CPU calcula la suya. Apenas cambia el prefill de 18K, limitado por el enlace x8 de la
  3090.
- Aciertos de expertos en VRAM: 83,6 % / 90,6 % (UD-IQ4_XS, 1 / 2 GPU), 80,7 % / 85,1 % (UD-Q4_K_XL). Aceptación de
  los borradores ~89 %.
- Sesiones de agente (una gráfica, una ronda cada una): UD-IQ4_XS 86,2 tok/s, UD-Q4_K_XL 64,5 tok/s (antes: 78,1 y
  59,9).
- Aquí UD-IQ4_XS es más rápido que UD-Q4_K_XL (+30 % con una gráfica, +37 % con dos): cada experto ocupa menos, así
  que caben más en VRAM, y sus kernels de CPU, limitados por cálculo, aprovechan los 7 workers. UD-Q4_K_XL es el
  cuantizado más grande de los dos.

## Prompts largos

Prompts de 32K a 200K tokens distintos (prosa, luego la documentación y el código de este repositorio), 128 tokens de respuesta,
una ronda cada uno.

| Prompt | Lectura (prefill tok/s) UD-IQ4_XS 1 / 2 GPU | UD-Q4_K_XL 1 / 2 GPU | Generación después: UD-IQ4_XS 1 / 2 GPU | UD-Q4_K_XL 1 / 2 GPU |
|---|---:|---:|---:|---:|
| 32.022 tokens | 13,4 s (2.390) / 13,3 s (2.400) | 13,6 s (2.363) / 13,8 s (2.320) | 73,5 / 94,7 | 58,6 / 83,0 |
| 64.022 tokens | 26,4 s (2.427) / 25,6 s (2.504) | 26,6 s (2.405) / 26,5 s (2.418) | 93,6 / 99,9 | 77,0 / 90,0 |
| 120.019 tokens | 50,4 s (2.383) / 48,1 s (2.494) | 51,0 s (2.355) / 50,0 s (2.399) | 66,2 / 98,1 | 48,4 / 69,8 |
| 200.019 tokens | 88,5 s (2.261) / 83,6 s (2.392) | 90,7 s (2.205) / 87,9 s (2.276) | 60,0 / 97,3 | 41,6 / 74,0 |

La K/V crece con el prompt: con la 3090 sola, UD-IQ4_XS llega a las 200.192 celdas cediendo 452 de sus 7.160 ranuras
de expertos (6.121 siguen con expertos, como sin `--kv-grow`).

## Con 64 GB de RAM

El mismo PC con el motor limitado a 60 GiB (lo que deja libre un PC de 64 GB) con un cgroup de systemd (`MemoryMax=60G`;
la memoria del motor y la caché de disco de los ficheros que lee contaban en él). Los 71,7 GiB de expertos de
UD-Q4_K_XL no caben: va con un pack con `experts.bin` (`iq_pack.py --experts-bin`) y `--mmap-experts`, leyendo del
NVMe (Crucial P3 Plus), y ahí `--kv-grow` queda apagado. Antes de cada ronda se vaciaba la caché de los ficheros del
modelo. Rondas: UD-IQ4_XS 2 y 2, UD-Q4_K_XL 1 y 1.

| Generación, tokens/s | UD-IQ4_XS 1 GPU | UD-IQ4_XS 2 GPU | UD-Q4_K_XL 1 GPU | UD-Q4_K_XL 2 GPU |
|---|---:|---:|---:|---:|
| Chat en español | 89,8 | 112,4 | 25,2 | 37,3 |
| Razonamiento | 97,3 | 120,2 | 30,5 | 39,7 |
| Tras un documento de 18K | 79,0 | 104,4 | 19,9 | 32,9 |
| Código | 99,2 | 127,7 | 28,3 | 43,1 |
| Edición | 132,8 | 183,3 | 44,2 | 71,3 |
| Tras un prompt de 5K | 69,6 | 72,8 | 15,7 | 26,1 |
| **Media geométrica** | **92,7** | **115,7** | **25,9** | **39,7** |
| **Media geométrica** (128 GB) | 92,9 | 116,3 | 71,2 | 84,9 |

| Prefill, tokens/s | UD-IQ4_XS 1 GPU | UD-IQ4_XS 2 GPU | UD-Q4_K_XL 1 GPU | UD-Q4_K_XL 2 GPU |
|---|---:|---:|---:|---:|
| 18.076 tokens | 1.813 | 1.833 | 438 | 491 |
| 5.296 tokens | 1.180 | 1.550 | 162 | 213 |
| 3.340 tokens | 851 | 1.084 | 147 | 157 |

Prompts largos con 64 GB (una ronda cada uno): tiempo de lectura del prompt · generación justo después. Los de
UD-Q4_K_XL son del motor anterior a la K/V elástica y el pool fusionado (no se repitieron: el prompt de 200K tardó
43,8 min).

| Prompt | UD-IQ4_XS 1 GPU | UD-IQ4_XS 2 GPU | UD-Q4_K_XL 1 GPU | UD-Q4_K_XL 2 GPU |
|---|---:|---:|---:|---:|
| 32.022 tokens | 13,4 s · 75,1 | 13,4 s · 101,4 | 42,7 s · 17,8 | 44,1 s · 33,1 |
| 64.022 tokens | 26,3 s · 95,0 | 25,6 s · 97,7 | 2,7 min · 22,9 | 1,7 min · 50,0 |
| 120.019 tokens | 50,3 s · 64,9 | 48,1 s · 105,6 | 4,6 min · 2,7 | 72,1 s · 50,2 |
| 200.019 tokens | 88,3 s · 59,4 | 83,3 s · 98,2 | 43,8 min · 6,0 | 2,0 min · 8,2 |

Sesiones de agente con 64 GB (una ronda cada una, `--prompt-cache 4`): UD-IQ4_XS 85,6 tok/s (128 GB: 86,2),
UD-Q4_K_XL 30,7 (128 GB: 64,5). Con los 16 puntos de control que trae el motor por defecto (~112 MiB de RAM cada uno),
la sesión de UD-IQ4_XS pasó el límite de 60 GiB en el turno 12.

## Lo que han aportado los cambios sobre `custom` de eddoursul

UD-IQ4_XS con una gráfica, media geométrica de los seis textos:

| Paso | tok/s |
|---|---:|
| `custom` de eddoursul + #2 + su continuación + los arreglos de esta rama (cinco rondas) | 77,3 (76,8-77,8) |
| + el kernel IQ4_XS AVX-2 multi-token (Niko1221/Strata) | 78,0 (+1,3 %) |
| + el gather AVX2 de la tabla de IQ3_S (`STRATA_IQ256_GATHER=1`) | 80,9 (+4,9 %) |
| + `--adapt-decay 0.92` | 83,3 (+7,7 %) |
| + los añadidos posteriores de PR del original (#863, #851, #606/#838) | 83,6 (+8,2 %) |
| + la K/V elástica (`--kv-grow`, 0.1.40 del original) | 90,9 |
| + el pool de CPU fusionado (`STRATA_POOL_FUSED=1`) | 92,9 |

Los dos últimos se midieron uno frente a otro y frente al paso anterior en la misma sesión, dos rondas cada uno (82,5
-> 90,9 -> 92,9: +10,1 %, +12,5 %); los pasos anteriores, los días antes.

En las sesiones de agente los mismos pasos dieron 75,0 -> 76,0 -> 77,8 (+3,7 %). El gather da exactamente los mismos
números (la herramienta de paridad sale idéntica con y sin él); la aceptación y el prefill no cambiaron.

`--adapt-decay` en UD-IQ4_XS (una gráfica, con el gather; dos rondas cada uno, antes de la K/V elástica y el pool fusionado): 0,7 76,8 · 0,85 82,1 · 0,92 83,6 · 0,95 81,0 · 0,97 81,3 · 0,99 70,8
(el 1,0 se rechaza desde el PR #591). Con la segunda gráfica 0,92 anula la ganancia del gather y
0,95 da -2 %, y en UD-Q4_K_XL no aporta en las sesiones de agente, así que esas configuraciones se quedan en 0,7.

### Añadidos posteriores (de pull requests abiertos del motor original)

- **Bloques q8_1 siempre finitos** (#606 y PR #838 del original, en los cinco cuantizadores de este fork): una
  activación enorme podía convertir en infinito la escala o la suma fp16 de un bloque, luego en NaN, y el modelo se
  quedaba repitiendo un token. Los mismos bits en todo bloque normal, sin coste de velocidad.
- **El gather AVX2 reorganizado** (PR #863 de Hardin22): UD-IQ4_XS con la 3090 83,1 -> 83,6 tok/s (cinco rondas), con las dos
  108,0 -> 109,4 (cuatro rondas). Su camino de un solo token para IQ3_S sigue apagado en AMD, donde el dot de ggml aún es
  algo más rápido para un token (0,322 frente a 0,334 ms por experto en el 5700X).
- **Un cuantizador AVX2 de activaciones Q8_K** (PR #851 de Hardin22): los mismos bytes que el de ggml; aquí no se nota.
- **`--adapt-decay` fuera de (0, 1) rechazado** (PR #591). UD-Q4_K_XL no cambia con ninguno de los cuatro.

### De la 0.1.40 del original y de otros forks

- **La K/V elástica** (2fbe321 + 4da8a57 del original, `--kv-grow`): 82,5 -> 90,9 en UD-IQ4_XS con la 3090.
  Comprobado: con una caché fija de 3.000 ranuras y sin adaptación, la K/V creciendo desde 4.096 celdas de 2.048 en
  2.048 con VRAM libre da los tokens de los seis textos bit a bit; también nueve peticiones de dos conversaciones
  intercaladas por el servidor, con la caché de prompts guardándolas y restaurándolas. Apagada con `--mmap-experts`
  (allí fallaba tras el primer crecimiento; el original pide todos los expertos en RAM).
- **El pool de CPU fusionado** (`STRATA_POOL_FUSED=1`, a partir de 11b1f25 de Hardin22/Strata-DualGPU): de +2,3 %
  (2 GPU) a +3,7 % (UD-Q4_K_XL, 1 GPU), las mismas operaciones.
- **Las subidas de la tabla de residencia esperan a su copia** (#1001 del original): sin cambio en la salida.
- Probado sin ganancia: el decodificado escalar de IQ3_S del PR #930 del original (0,403 -> 0,278 ms por experto con
  un token en un hilo, pero 83,6 -> 83,3 en el motor: sus 8 hilos los limita la RAM), `--no-second-gpu-adapt`
  (108,4 -> 108,5) y las cachés K/V de 4 bits (no se usan: pierden algo de precisión).
- El contexto reservado, antes de `--kv-grow` (UD-IQ4_XS, 3090, dos rondas cada uno): 200K 83,3 · 128K 86,7 ·
  64K 89,9 · 32K 95,3 tok/s.

Otros ajustes probados en este equipo sin ganancia: más workers que 7 (UD-IQ4_XS) o 4 (UD-Q4_K_XL), `--spec` 3/5/6,
`--spec-min-p` 0,3/0,7, `--spec-lookup` 8/24, `--adapt-every 2`, `--adapt-swaps 192` en UD-Q4_K_XL,
`--pcie-frac` por encima de 0,3 y las gráficas a su potencia de serie.

En esta CPU con SMT, el arreglo de Linux de esta rama (un worker por núcleo físico) llevó UD-Q4_K_XL con la 3090 de
50,7 tok/s (15 workers en 16 procesadores lógicos) a 58,1 (7 workers en 7 núcleos), antes del resto del afinado.
