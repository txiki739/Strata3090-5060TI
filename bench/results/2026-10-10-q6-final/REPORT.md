# Huihui UD-Q6_K_XL frente a Q4 y Q8 (dual, RTX 3090 + RTX 5060 Ti)

Expertos fijos elegidos para el Q6: **6 GiB por GPU**.

## Velocidad (bench, media geométrica de 6 prompts, tok/s)

| Ejecución | Media | es_chat | es_think | es_doc | code | edit | proto5k | Prefill 18K / 5,3K / 3,3K |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Q6-pin0 | **54.6** | 54.5 | 59.1 | 47.7 | 56.8 | 74.0 | 41.0 | 858 / 595 / 281 |
| Q6-pin6 | **55.6** | 58.2 | 56.6 | 52.4 | 55.5 | 62.4 | 49.6 | 1444 / 728 / 519 |
| Q6-pin11 | **55.1** | 57.4 | 54.3 | 57.2 | 52.2 | 52.7 | 57.1 | 1469 / 726 / 520 |
| Q6-pin11-b | **54.3** | 59.6 | 52.0 | 57.4 | 53.1 | 53.0 | 51.2 | 1469 / 604 / 402 |
| Q6-pin6-b | **56.2** | 60.2 | 55.8 | 55.4 | 55.9 | 62.5 | 48.6 | 1445 / 728 / 515 |
| Q6-pin0-b | **53.8** | 60.6 | 56.6 | 46.0 | 55.1 | 71.5 | 38.9 | 1414 / 666 / 511 |
| Q4-s | **81.9** | 63.5 | 85.9 | 82.5 | 97.0 | 133.9 | 51.7 | 922 / 1116 / 810 |
| Q8-s | **42.8** | 42.9 | 44.5 | 44.5 | 41.2 | 41.4 | 42.2 | 1340 / 610 / 425 |

## Prompts largos (32K-200K tokens, 128 de respuesta)

| Ejecución | 32k: lectura tok/s · s · generación tok/s | 64k: lectura tok/s · s · generación tok/s | 120k: lectura tok/s · s · generación tok/s | 200k: lectura tok/s · s · generación tok/s |
|---|---|---|---|---|
| L6 | 2205 · 14.5 · 51.8 | 2297 · 27.9 · 66.9 | 2225 · 53.9 · 41.2 | 1676 · 119.3 · 52.1 |
| L8 | 2147 · 14.9 · 40.0 | 2256 · 28.4 · 46.6 | 2177 · 55.1 · 33.5 | 2057 · 97.2 · 40.4 |
| L4 | 2339 · 13.7 · 76.1 | 2428 · 26.4 · 92.3 | 2410 · 49.8 · 74.8 | 2284 · 87.6 · 71.0 |
| R6 | 2173 · 14.7 · 52.0 | 2292 · 27.9 · 63.3 | 2223 · 54.0 · 42.3 | 2109 · 94.8 · 51.6 |
| R8 | 2148 · 14.9 · 38.9 | 2251 · 28.4 · 44.7 | 2172 · 55.3 · 33.1 | 2054 · 97.4 · 38.5 |
(L4 = Huihui Q4, L8 = Huihui Q8 de la tanda anterior; L6 = Huihui Q6.  R6/R8 = Q6 y Q8 repetidos con contadores del sistema (2026-10-10-q6-200k): los 200K del L6 (119 s) no se reproducen, el Q6 da 94,8 s, entre el Q4 y el Q8 como el resto)

## Sesión de agente (18 pasos): Huihui Q6

57.72 tok/s de media, 10427 tokens generados, contexto final 31699 (Q4: 84,6 tok/s; Q8: 44,6)

## Afinado del Q6 (expertos fijos 6+6)

T0/T0b = base (PCIe 0,25, spec-min-p 0,8, precarga 6 MB); P15/P35 = PCIe 0,15/0,35; M7/M9 = spec-min-p 0,7/0,9; F12 = precarga 12 MB; W5 = 5 hilos de CPU; H35/H55 = cabeza 35/55 % en la 5060 Ti; C1/C2 = la combinación ganadora.

| Ejecución | Media tok/s | es_chat | es_think | es_doc | code | edit | proto5k |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0 | **55.6** | 61.0 | 54.1 | 54.9 | 54.3 | 63.9 | 46.8 |
| P15 | **55.9** | 57.5 | 63.0 | 54.1 | 54.6 | 61.6 | 46.3 |
| P35 | **54.6** | 57.2 | 56.6 | 54.6 | 55.9 | 60.5 | 44.4 |
| M7 | **54.8** | 56.9 | 57.5 | 52.1 | 52.5 | 64.1 | 46.9 |
| M9 | **56.8** | 57.5 | 64.0 | 54.6 | 55.6 | 62.7 | 48.0 |
| F12 | **55.3** | 55.5 | 59.1 | 52.3 | 54.2 | 64.0 | 48.2 |
| W5 | **55.3** | 59.5 | 62.9 | 52.4 | 52.1 | 62.6 | 44.9 |
| H35 | **56.0** | 58.0 | 56.5 | 55.8 | 55.0 | 63.8 | 48.0 |
| H55 | **55.2** | 55.2 | 64.3 | 52.2 | 52.4 | 63.4 | 45.8 |
| T0b | **56.2** | 58.9 | 62.8 | 52.2 | 55.9 | 62.5 | 46.7 |

Configuración final del Q6: PCIe 0.25, spec-min-p 0.8, precarga 6 MB, hilos 4, cabeza 0.45, fijos 6 + 6 GiB.

## Calidad: evaluación forzada de 19.208 tokens frente al Huihui Q8 (B8)

ΔNLL en nats por token (negativo = predice mejor el texto que el Q8); KL y top-1 frente al Q8. B8b es el Q8 repetido (ruido).

| Modelo | Tipo de texto | Perplejidad | ΔNLL | KL | Top-1 igual |
|---|---|---:|---:|---:|---:|
| Huihui Q8 (repetido) | código | 3.049 | -0.0374 ± 0.0052 | 0.0458 | 93.9 % |
| Huihui Q8 (repetido) | en-técnico | 5.432 | +0.0079 ± 0.0037 | 0.0202 | 93.8 % |
| Huihui Q8 (repetido) | es-enciclopedia | 5.930 | +0.0191 ± 0.0036 | 0.0144 | 95.7 % |
| Huihui Q8 (repetido) | es-reservado | 5.514 | +0.0065 ± 0.0023 | 0.0118 | 95.5 % |
| Huihui Q8 (repetido) | TOTAL | 4.933 | +0.0003 ± 0.0018 | 0.0212 | 94.8 % |
| Huihui Q6 (abliteración del Q8) | código | 3.081 | -0.0271 ± 0.0039 | 0.0238 | 95.6 % |
| Huihui Q6 (abliteración del Q8) | en-técnico | 5.406 | +0.0031 ± 0.0033 | 0.0165 | 94.5 % |
| Huihui Q6 (abliteración del Q8) | es-enciclopedia | 6.020 | +0.0342 ± 0.0040 | 0.0181 | 95.2 % |
| Huihui Q6 (abliteración del Q8) | es-reservado | 5.546 | +0.0122 ± 0.0026 | 0.0147 | 95.0 % |
| Huihui Q6 (abliteración del Q8) | TOTAL | 4.965 | +0.0067 ± 0.0017 | 0.0177 | 95.1 % |
| Huihui Q4 | código | 3.015 | -0.0487 ± 0.0079 | 0.1158 | 91.9 % |
| Huihui Q4 | en-técnico | 5.329 | -0.0113 ± 0.0059 | 0.0502 | 90.5 % |
| Huihui Q4 | es-enciclopedia | 5.733 | -0.0146 ± 0.0064 | 0.0556 | 91.6 % |
| Huihui Q4 | es-reservado | 5.493 | +0.0026 ± 0.0045 | 0.0478 | 91.1 % |
| Huihui Q4 | TOTAL | 4.860 | -0.0146 ± 0.0030 | 0.0641 | 91.2 % |
| unsloth Q4 (original) | código | 3.073 | -0.0296 ± 0.0054 | 0.0458 | 93.4 % |
| unsloth Q4 (original) | en-técnico | 5.403 | +0.0025 ± 0.0054 | 0.0368 | 91.4 % |
| unsloth Q4 (original) | es-enciclopedia | 5.735 | -0.0143 ± 0.0059 | 0.0435 | 92.5 % |
| unsloth Q4 (original) | es-reservado | 5.439 | -0.0073 ± 0.0039 | 0.0382 | 92.2 % |
| unsloth Q4 (original) | TOTAL | 4.876 | -0.0113 ± 0.0025 | 0.0406 | 92.3 % |

## 50 problemas difíciles (respuestas calculadas por código), con razonamiento

| Modelo | Matemáticas y lógica | Programación | Tokens de razonamiento (total) | Tiempo (total) |
|---|---:|---:|---:|---:|
| Huihui Q4 | 35/40 | 8/10 | 116,189 | 21 min |
| Huihui Q6 | 33/40 | 7/10 | 123,543 | 36 min |
| Huihui Q8 | 32/40 | 8/10 | 116,969 | 49 min |

Problemas en los que no todos aciertan:

| Problema | Esperado | Huihui Q4 | Huihui Q6 | Huihui Q8 |
|---|---|---|---|---|
| P04 ¿De cuántas formas se puede cubrir por completo un tablero d… | 571 | ✅ | ❌  | ❌  |
| P07 ¿Cuánto vale la suma de todos los números primos menores que… | 277050 | ❌  | ❌  | ❌  |
| P09 ¿Cuántos días hay entre el 15 de marzo de 1987 y el 9 de oct… | 14454 | ✅ | ✅ | ❌ 14453 |
| P13 ¿Cuántos números de cuatro cifras (de 1000 a 9999) tienen su… | 1120 | ❌  | ❌  | ❌  |
| P14 ¿Cuántos subconjuntos del conjunto {1, 2, ..., 15} tienen su… | 401 | ❌  | ❌  | ❌  |
| P25 ¿De cuántas formas distintas se puede formar 1 euro con mone… | 4562 | ❌  | ❌  | ❌  |
| P27 41 personas en círculo, numeradas del 1 al 41, cuentan de 3 … | 31 | ✅ | ❌  | ✅ |
| P31 ¿Cuántos enteros positivos menores que 1000 son iguales a la… | 5 | ✅ | ✅ | ❌  |
| P34 ¿Cuál es el menor número primo mayor que 1.000.000?… | 1000003 | ❌  | ❌  | ❌  |
| C1 Escribe una función Python `coincide(s, p)` que diga si la c… | tests | ❌ | ✅ | ❌ |
| C4 Escribe una función Python `mochila(pesos, valores, capacida… | tests | ✅ | ❌ | ✅ |
| C5 Escribe una función Python `evaluar(expr)` que evalúe una ex… | tests | ❌ | ❌ | ❌ |
| C6 Escribe una función Python `palindromo_mas_largo(s)` que dev… | tests | ✅ | ❌ | ✅ |
