# `--ple-io ram` en la config de UD-IQ4_XS con la 3090 sola, 2026-10-14

Petición: rascar el prompt de la config de una sola GPU sin tocar la calidad, y de paso comprobar los pines de
expertos en VRAM (`--vram-pin-gib`) que quedaban pendientes para esta config.

## Resultado

La tabla PLE leída de RAM (`--ple-io ram`) **se queda**: prompt de 18.076 tokens 1.855,1 -> **2.335,7 tok/s
(+26 %)**, 3.340 tokens 862,1 -> **943,6 (+9,5 %)**, 5.296 tokens y los cortos igual, decode dentro del ruido
(geomean de las seis pruebas 93,70 -> 93,83) y **los seis textos bit a bit** (mismos hashes de la respuesta y mismas
aceptaciones de la especulación).

| Prueba | prompt antes | prompt con `--ple-io ram` | decode antes -> después |
|---|---:|---:|---:|
| es_doc (18.076 tokens) | 1.855,1 | **2.335,7** | 80,30 -> 80,38 |
| edit (3.340 tokens) | 862,1 | **943,6** | 132,39 -> 132,33 |
| proto5k (5.296 tokens) | 1.181,9 | 1.181,2 | 69,94 -> 70,14 |
| es_chat / es_think / code | 69,0 / 105,0 / 76,7 | 68,9 / 104,9 / 76,7 | iguales |
| **decode geomean** | 93,70 | 93,83 | — |

Config: `examples/r7-5700x/ud-iq4_xs-3090.json` (con `--ple-io ram` añadido en esta fecha).

## Pines descartados

- `--vram-pin-gib 12`: el motor **muere al cargar** (el pin se come el préstamo de la K/V elástica; el log lo deja en
  45 chunks sin VRAM). No arranca.
- `--vram-pin-gib 8`: arranca, pero el geomean de decode cae a **86,35 (-8 %)** (88-99 tok/s por prueba) con el prompt
  igual de rápido. No compensa: se quedan apagados en esta config.
