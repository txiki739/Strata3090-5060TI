# Optimización de la 5950X, 2026-10-03

Solicitud: maximizar el rendimiento de Qwen3.8 Flash-Next IQ4_XS y UD-Q4_K_XL sin perder calidad del modelo.

## Resultado

UD-IQ4_XS en la 5950X (RTX 3090 a 300 W, 62,7 GiB): **68,8 -> 99,3 tok/s** de media de decode en las seis
pruebas del banco (`iq4xs-baseline-new` -> `iq4xs-prod-final2`, binario y config de producción), mismo prefill
(2.268 tok/s a 18K, limitado por el PCIe 4.0 x8). Config: la de antes con `--pool-workers 14` y
`--adapt-swaps 192` (`examples/r9-5950x/ud-iq4_xs-3090.json`); posible gracias al arreglo del tier adaptativo
(c99b7b3). Contexto largo: 32K/64K/120K tokens a 2.596/2.628/2.582 tok/s de prefill (120K en 46,5 s) y 72/90/73
tok/s de decode después (`iq4xs-ctx-long`). UD-Q4_K_XL quedó fuera de la 5950X: por mmap funciona (9265a58) pero a
~30 tok/s, al no caber sus 71,7 GiB de expertos en RAM.

## Restricciones

- Conservar pesos GGUF, cuantización, KV int8, contexto máximo 200192 y muestreo.
- No podar expertos, no requantizar, no reducir capas ni contexto.
- Medir configuración base y candidatos en la 5950X (62,69 GiB RAM, RTX 3090 24 GiB a 300 W).
- Ejecutar benchmarks secuenciales en unidades de usuario limitadas por memoria y tiempo; no repetir la arena Q4_K_XL de 71,73 GiB.
- Mantener la 5700X operativa; desarrollar en este worktree aislado.

## Diseño

1. IQ4_XS: medir workers y afinidad por CCD, prefill, feed-max y especulación exacta. Comparar pruebas completas y contexto largo para finalistas; mantener solo ganancias reproducibles.
2. Q4_K_XL: corregir FileExpertSource para utilizar los offsets/tamaños nativos que ya usa ArenaExpertSource. Empaquetar los expertos sin alterar bytes con iq_pack.py --experts-bin. Mapear ese archivo y permitir al kernel gestionar páginas; no cargar una arena residente completa.
3. Verificar lectura exacta mediante regresión C++ con capas heterogéneas, tamaños erróneos y límites. Comprobar texto/logits para comparar fuentes mmap/residente sobre un modelo que cabe (IQ4_XS) y/o Q4_K_XL en la 5700X si su GPU queda disponible sin interrumpir al usuario.
4. Integrar configs/lanzadores solo tras carga, generación y mediciones satisfactorias; guardar antes/después y límites observados.

Alternativas: la arena residente sigue siendo preferible para IQ4_XS si gana; para Q4_K_XL excede RAM. Requantizar no cumple la petición. Modificar la caché GPU para eliminar copias CPU requeriría una reestructuración mucho mayor: se evaluará solo si mmap no basta.

## Estado

- [x] Contexto, hardware, causa OOM y referencia COMPARISON.md revisados.
- [x] Base IQ4_XS nueva (`iq4xs-opt-arena-r2`: 68,43 tok/s, 6/6 hashes = `iq4xs-baseline-new`) y barrido de CPU (abajo).
- [x] Regresión mmap nativo: 33 fallos antes del arreglo, PASS después (la 5700X y la 5950X).
- [x] Identidad mmap = arena en IQ4_XS: con `--expert-cache=4000 --adapt-every=0` ambas ramas abren 5757 huecos y dan 6/6 hashes, aceptaciones y aciertos idénticos (`iq4xs-id2-*`). Ojo: con packs nativos `--expert-cache N` es un presupuesto que recorta la VRAM libre; con 6122 la arena abrió 6123 huecos y el mmap 6173, y el texto difería solo por eso (`iq4xs-id-*`).
- [x] Pack Q4_K_XL exacto (`packs/ud-q4_k_xl-mmap`: experts.bin 77.017.907.200 B; dense/index/native_experts/tokenizer idénticos al pack original). Arranca por mmap en la 5950X sin OOM con `MemoryMax=56G`: 23,41 tok/s de media de decode.
- [x] Q4_K_XL por mmap medido (en caliente 28,5-30,5 tok/s; los 64 bloques de staging resultaron MÁS lentos, 23,8-26,4, y se descartaron). Integrado y probado de extremo a extremo; después el usuario decidió quitar Q4_K_XL de la 5950X (menú, configs y 236 GB de disco).
- [x] Bloqueo del tier adaptativo encontrado y corregido. Con >= 6 pool workers la caché VRAM se congelaba a mitad de sesión (edit 69 tok/s al 35 % de aciertos). Causa: desde 75fbfb0 los pumps llaman a `retire()` y tiraban el índice de lo llegado; si una petición acababa con un lote en vuelo y la siguiente usaba el prompt path, sus 96 movimientos no se admitían nunca y en modo ráfaga (una GPU) el tier no volvía a actualizar. Arreglo: `landed_` como miembro. Regresión: pw12 edit 137,8 tok/s al 79 %, media 97,2 tok/s (pw4 68,3). Traza opcional `STRATA_TIER_TRACE=1`.
- [x] Barrido de workers con el arreglo. La 5950X: 4 -> 68,7, 8 -> 89,7, 12 -> 97,2, **14 -> 98,2 tok/s** (repetido 98,3); repartir entre CCD solo ayuda con 8. La 5700X (un CCD): 4 sigue siendo lo mejor (3090 63,5; dual 80,9), el arreglo no cambia su velocidad.
- [x] Segunda tanda en la 5950X (14 workers): `--pcie-frac` empeora en todos los modos (dma 0,55 -> 60,4; kernel 0,30 -> 90,2), se queda en 0; `--prefill 16384` baja el prefill de 18K de 2.271 a 1.717 tok/s, se queda en 32768.
- [x] Producción: binarios con el arreglo en la 5700X y la 5950X (`build/strata.bak-pretierfix-20261003` el anterior), la 5950X `--pool-workers 14` (98,36 tok/s con binario y config de producción, opción 8 del menú probada por la API). Menú `modelos` con la cifra nueva.
- [x] Agentes: sync con Strata de la 5950X (`strata_opts`), `EXTRA` para los vLLM que solo lanza el launcher web, retirados los modelos que ya no se sirven; aplicado y verificado en las 8 máquinas.
- [x] Especulación y tier con 14 workers (`iq4xs-sa-*`, base 97,5): `--spec` 3/5/6 -> 94,7/96,8/93,4; `--spec-min-p` 0,3/0,7 -> 93,8/97,2; `--spec-lookup` 8/24 -> 96,4/97,8 (ruido); `--adapt-every 2` -> 93,9; `--adapt-swaps` 48 -> 92,9, **192 -> 99,3 y 99,0** (confirmado frente a bases 97,5-98,4), 320 -> 98,7. Se adopta `--adapt-swaps 192`: 99,31 tok/s con la config de producción (`iq4xs-prod-final2`).
- [x] Commits: 9265a58 (mmap nativo), c99b7b3 (tier adaptativo) y este informe con los resultados.
- Hardware: la 3090 de la 5950X va a PCIe 4.0 x8 porque la Aquantia AQC113 ocupa PCIEX16_2 (PRIME X570-PRO reparte x8/x8). Estimado a x16: prefill +18-21 % a 18K, +38-54 % a 3-5K.
