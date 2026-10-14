# `env_to_user` en las configs de esta máquina, 2026-10-14

Qué es: el servidor de la línea 0.1.44 mueve el bloque `<env>` que los clientes de agente (opencode) ponen en el
turno de sistema al **primer mensaje de usuario**. El modelo ve lo mismo, solo cambia dónde va; y como el bloque
`<env>` es lo único que cambia entre sesiones de carpetas o días distintos, la sesión nueva pasa a reutilizar la
caché de prompts en vez de releer el sistema entero.

Medición con la forma exacta de una petición de opencode (sistema idéntico de ~12,7K tokens, cuatro "sesiones" con
carpeta y fecha distintas; la caché ya estaba caliente por sesiones previas):

| | primera petición | con la caché caliente |
|---|---:|---:|
| sin `env_to_user` | 6,2-6,7 s | — |
| con `env_to_user` | 6,6-7,0 s (la caché todavía se está formando) | **0,5-0,6 s** — 12.671 de 12.720 tokens desde caché |

Adoptado: `"env_to_user": true` en las cuatro configs de `examples/r7-5700x/`. Solo afecta a las peticiones que
llevan un bloque `<env>`; el resto no cambia.
