# DDD Pipeline — Limpieza y estructuración de tensores

Pipeline automatizado para limpiar y estructurar el **Driver Drowsiness Dataset (DDD)**
(`Drowsy/` vs `Non Drowsy/`) en tensores `.npz` listos para entrenar una red neuronal.

## Características

- **Validación**: decodifica cada PNG, fuerza RGB y descarta los que no tengan el tamaño nativo (`227x227`).
- **Duplicados**: detección por hash SHA-256 (se pueden saltar con `--skip-hash`).
- **Reporte de limpieza**: `cleaning_report.csv` con el estado de cada imagen.
- **Split agrupado por sujeto** (70/15/15 por defecto, `seed=42`): evita *data leakage*, ya que el mismo
  sujeto (rostro) no aparece en dos splits distintos.
- **Tensores**: `train.npz`, `val.npz`, `test.npz` con `X` uint8 `[N, 227, 227, 3]` e `y` `[N]`.
- **Estadísticas**: `normalization.json` (media/desvío por canal, calculados solo en train) y
  `class_weights.json`.
- **Bajo consumo de RAM**: los tensores se escriben a través de un buffer `.npy` mapeado en memoria.

## Estructura

```
DDD-Pipeline/
├─ src/
│  └─ prepare_ddd.py       # script principal
├─ .vscode/                # configuración de VS Code (launch + settings)
├─ DDD-Pipeline.sln        # solución de Visual Studio
├─ DDD-Pipeline.pyproj     # proyecto Python de Visual Studio
├─ requirements.txt
└─ README.md
```

## Instalación

```powershell
python -m pip install -r requirements.txt
```

## Uso

```powershell
python src/prepare_ddd.py `
  --data-dir "D:/IA/ProyectoRedesNeuronales/DriversDrowsinessDataset/Driver Drowsiness Dataset (DDD)" `
  --out-dir  "D:/IA/ProyectoRedesNeuronales/processed"
```

### Opciones

| Argumento      | Default | Descripción                                        |
|----------------|---------|----------------------------------------------------|
| `--data-dir`   | —       | Carpeta raíz con `Drowsy/` y `Non Drowsy/`.        |
| `--out-dir`    | —       | Carpeta de salida (tensores + reportes).           |
| `--val`        | 0.15    | Fracción de validación (por sujeto).               |
| `--test`       | 0.15    | Fracción de test (por sujeto).                     |
| `--seed`       | 42      | Semilla aleatoria.                                 |
| `--size`       | 227     | Tamaño nativo esperado de la imagen.               |
| `--skip-hash`  | off     | Omite la detección de duplicados (más rápido).     |
| `--limit`      | 0       | Prueba rápida: limita imágenes por clase.          |

## Salidas (`--out-dir`)

| Archivo                | Contenido                                                    |
|------------------------|-------------------------------------------------------------|
| `train.npz`/`val.npz`/`test.npz` | `X` uint8 `[N,227,227,3]` + `y` uint8 `[N]`.     |
| `normalization.json`   | `mean`/`std` por canal (train, escala 0-1).                 |
| `class_weights.json`   | Pesos de clase balanceados + conteos.                       |
| `splits.csv`           | `path, subject, frame, label, split`.                       |
| `cleaning_report.csv`  | Estado y motivo por imagen (`kept`/`dropped`).              |
| `metadata.json`        | Resumen completo de parámetros y resultados.                |

## Uso en código

```python
import numpy as np

data = np.load("processed/train.npz")
X, y = data["X"], data["y"]          # uint8, [N, 227, 227, 3], [N]
X = X.astype("float32") / 255.0
# normalizar con mean/std de normalization.json
```

## Abrir en Visual Studio

Abre `DDD-Pipeline.sln` y ejecuta `src/prepare_ddd.py`. En VS Code, la configuración de
`F5` ya está en `.vscode/launch.json`.
