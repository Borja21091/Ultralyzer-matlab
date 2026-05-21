# Ultralyzer

A pipeline to analyze retinal UltraWidefield images using MVC GUI structure and background processing.

## Geometry Backends

Ultralyzer uses a geometry adapter for geodesic-distance and spherical-area measurements.

- This build is configured for the MATLAB geometry backend.
- Set `ULTRALYZER_GEOMETRY_BACKENDS=matlab` before launching the app.
- If the configured geometry backend is unavailable or incomplete, Ultralyzer still runs, but some geometry-dependent metrics are skipped.

Examples:

```bash
# macOS / Linux
export ULTRALYZER_GEOMETRY_BACKENDS=matlab
python src/ultralyzer/main.py
```

```bat
REM Windows (Command Prompt / Anaconda Prompt)
set ULTRALYZER_GEOMETRY_BACKENDS=matlab
python src\ultralyzer\main.py
```
