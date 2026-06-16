# Installation and Setup

This guide documents the setup path for Ultralyzer.

## System requirements

- Python 3.12 or newer
- Windows, macOS, or Linux
- A virtual environment manager, such as [Miniconda](https://www.anaconda.com/download/success?reg=skipped), is strongly recommended
- MATLAB available for geometry-dependent distance and area measurements

Ultralyzer can still open without complete geometry support. When that happens, segmentation, review, editing, and export workflows remain available, but some geometry-dependent metrics may be skipped.

The install scripts detect the installed MATLAB release and install the matching `matlabengine` package automatically. If MATLAB is installed in a non-standard location, set `ULTRALYZER_MATLABROOT` to the MATLAB root before running the installer.

If the installer reports that your current Python version is incompatible with the detected MATLAB release, recreate the environment with a Python version supported by that MATLAB release before enabling the MATLAB backend.

:::{important}

- Spherical geometry measurements are supported by a [MATLAB p-file](https://uk.mathworks.com/help/matlab/matlab_prog/building-a-content-obscured-format-with-p-code.html) called `distanceOnSP.p`. Make sure to place this file in `/Ultralyzer/src/ultralyzer/.bin/` after installation to enable geometry-dependent metrics.
- `distanceOnSP.p` is kindly provided by [Optos PLC](https://www.optos.com/contact-us/). If you do not have access to this file, refer to the Optos contact page for assistance.
:::

## Quick start

::::{tab-set}

:::{tab-item} Windows

Start by opening `Anaconda Prompt`. Then run the following commands:

1. Clone the repository.

   ```bash
   git clone https://github.com/Borja21091/Ultralyzer-matlab.git
   cd Ultralyzer-matlab
   ```

2. Create and activate an environment.

   ```bash
   conda create -n ultralyzer_env python=3.12
   conda activate ultralyzer_env
   ```

3. Install dependencies.

   ```bash
   install.bat
   ```

   This installs the general Python dependencies first and then installs the `matlabengine` version that matches the detected MATLAB release.

4. Select MATLAB as the backend.

   ```bash
   set ULTRALYZER_GEOMETRY_BACKENDS=matlab
   ```

5. Run the GUI.

   ```bash
   python src/ultralyzer/main.py
   ```

:::

:::{tab-item} macOS and Linux

Start by opening a terminal and run the following commands:

1. Clone the repository.

   ```bash
   git clone https://github.com/Borja21091/Ultralyzer-matlab.git
   cd Ultralyzer-matlab
   ```

2. Create and activate an environment.

   ```bash
   conda create -n ultralyzer_env python=3.12
   conda activate ultralyzer_env
   ```

3. Install dependencies.

   ```bash
   chmod +x install.sh
   bash install.sh
   ```

   This installs the general Python dependencies first and then installs the `matlabengine` version that matches the detected MATLAB release.

4. Select MATLAB as the backend.

   ```bash
   export ULTRALYZER_GEOMETRY_BACKENDS=matlab
   ```

5. Run the GUI.

   ```bash
   python src/ultralyzer/main.py
   ```

:::

::::

## Geometry readiness and skipped metrics

Ultralyzer checks geometry readiness from the GUI and summarizes it with a traffic-light indicator in the **Current image** card:

- `Green`: geometry-dependent metrics are fully available.
- `Yellow`: only part of the geometry workflow is available, so some metrics may be skipped.
- `Red`: geometry-dependent metrics are unavailable and will be skipped.

This status does not stop the rest of the application from loading. You can still review images, segment structures, edit masks, and export results.
