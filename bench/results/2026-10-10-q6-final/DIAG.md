# Diagnóstico de los picos de NLL del Huihui Q6

Frente al Huihui Q8 (B8), 19.208 tokens. Picos = tokens con ΔNLL > 2 nats.

| Variante | ΔNLL medio | Mediana | Picos > 2 | Picos < −2 | Aporte de los picos |
|---|---:|---:|---:|---:|---:|
| Huihui Q4 (referencia de abliteración completa) | -0.0145 | -0.00006 | 53 | 82 | -0.0057 |
| unsloth Q4 original | -0.0113 | -0.00008 | 25 | 51 | -0.0035 |
| Q6 normal (tanda anterior) | +0.0406 | +0.00023 | 153 | 83 | +0.0113 |
| Q6 normal, repetido | +0.0385 | +0.00029 | 158 | 74 | +0.0141 |
| Q6 sin expertos fijos (0 GiB) | +0.0302 | +0.00024 | 125 | 70 | +0.0090 |
| Q6 sin PCIe (--pcie-frac 0) | +0.0263 | +0.00007 | 134 | 81 | +0.0087 |
| Q6 sin la 5060 Ti | +0.0445 | +0.00033 | 172 | 58 | +0.0179 |
| Q6 con el pool sin fusionar | — | | | | |
| Q6 con ventanas cortas (--spec 2, sin búsqueda) | +0.0395 | +0.00030 | 163 | 82 | +0.0124 |
| Q6 con la abliteración a medias (bytes del Q8) | +0.0066 | +0.00000 | 14 | 15 | +0.0001 |
