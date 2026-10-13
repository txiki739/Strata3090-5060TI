# Notas para analizar la batería profunda (deep_eval.py)

- Bloque A: 4 peticiones piden cosas contradictorias («exactamente N viñetas / pasos y nada más» junto con título,
  posdata o comillas envolventes): A14, A19, A20, A38. Se excluyen del recuento estricto para TODOS los modelos
  (quedan 56). No se cambió el script durante la tanda para que todos los modelos reciban las mismas peticiones.
- Bloque F: 26 preguntas (no 30): el generador descarta las que tendrían empate o se repetirían.
