# MMQ del camino de prompt: ya va por defecto, 2026-10-14

La ruta MMQ de los expertos en el prompt (los kernels MMQ de llama.cpp para los K-quants — Q4_K, Q5_K, Q6_K de los
ficheros de Unsloth —, en `src/prefill/moe_mmq.cu`) está **activa por defecto**; `STRATA_PREFILL_MMQ=0` la apaga.
Medido hoy para dejarlo documentado por escrito:

| Prueba | con MMQ (por defecto) | con `STRATA_PREFILL_MMQ=0` | Δ |
|---|---:|---:|---:|
| prompt 18.076 tokens | **1.704,6** | 1.637,5 | +4,1 % |
| prompt 5.296 tokens | **901,4** | 813,2 | +10,8 % |
| prompt 3.340 tokens | **663,4** | 625,5 | +6,1 % |
| decode geomean | 73,01 | 72,52 | igual |

Conclusión: se queda por defecto (ya lo estaba); apagarlo cuesta entre un 4 y un 11 % de prompt.
