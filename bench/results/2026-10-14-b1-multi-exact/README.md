# El layout `ncols > 1` de upstream, medido en sm_86, 2026-10-14

Contexto: el comentario de `g_multi_exact` (`src/kernels/cuda/native_mmvq.cu`) pedía el layout de varias columnas
"hasta que se mida en GPU ociosa (regla: por defecto solo lo medido)". Esta es esa medición.

## Resultado

Se queda el layout **exacto** (una fila por bloque), que además es bit a bit igual a las llamadas de una sola columna.

| | 3090 sola | 3090 + 5060 Ti (reparto por capas) |
|---|---:|---:|
| exacto (por defecto) | **70,40** tok/s geomean | 89,82 |
| upstream (2 filas por bloque) | 69,25 (-1,6 %) | 91,86 (+2,3 %) |

- Seis pruebas (chat, pensar, documento de 18K, código, edición de 3,3K/2.600, prompt de 5,3K), decode tok/s por
  prueba y geomean sobre ellas; una pasada por brazo con las tarjetas libres, cambiando de layout por la API de la
  biblioteca (no forma parte del árbol).
- Las diferencias por prueba van en ambos sentidos (+4 % para upstream en el documento de 18K, -11 % en la prueba de
  pensar), así que el geomean es una indicación de una pasada, no una cifra fina. El prompt de 18K sale igual en los
  dos brazos (1.766,6 / 1.766,9 tok/s), como se espera: el camino bajo prueba es el de varias columnas (decode y
  ventanas de verificación).
- Los bits difieren por diseño: 5 de 6 textos con hash distinto en la sola, 6 de 6 en el par (allí la comparación de
  velocidad queda contaminada por trayectorias distintas).
- Comprobación independiente: el commit `945fee23` de la 0.1.44 de upstream eligió una fila por bloque para sm_86
  (su variante de una fila mantiene los bits de la de dos), misma dirección desde el otro lado.
- Queda registrado en el PR #22 de este equipo al `custom` de eddoursul (solo comentarios, sin cambio de
  comportamiento).
