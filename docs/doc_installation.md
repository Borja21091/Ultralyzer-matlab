# Installation

## System Requirements

Ultralyzer is compatible with Linux, Windows, and MacOS systems that support Python 3.12 (tested) or higher. The software leverages hardware acceleration for image processing and GUI rendering, so a system with a dedicated GPU is recommended for optimal performance, although not strictly necessary.

We recommend using a virtual environment (e.g., Conda) to manage dependencies and avoid conflicts with other Python packages on your system. You can start by installing [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install) or [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install).

## Geometry Backend Configuration

Ultralyzer uses a geometry adapter for geodesic-distance and spherical-area metrics.

- This build is configured for the MATLAB geometry backend.
- Start the application with `ULTRALYZER_GEOMETRY_BACKENDS=matlab`.
- If the selected geometry backend is unavailable or incomplete, the application still opens and non-geometry workflows continue to work, but some geometry-dependent metrics are skipped.

## Installation Steps

<details>

<summary>Windows</summary>

1. **Clone the Repository**:

   Open Anaconda Prompt and run:

   ```bash
   git clone https://github.com/Borja21091/Ultralyzer-matlab.git
   cd Ultralyzer-matlab
   ```

2. **Set Up a Virtual Environment** (optional but recommended):

   ```bash
   conda create -n ultralyzer_env python=3.12
   conda activate ultralyzer_env
   ```

3. **Install Dependencies**:

   Run the following command from the command line:

   ```bash
   install.bat
   ```

4. **Select the Geometry Backend**:

   Set the MATLAB geometry backend:

   ```bash
   set ULTRALYZER_GEOMETRY_BACKENDS=matlab
   ```

5. **Run the Application**:

   ```bash
   python src/ultralyzer/main.py
   ```

   If MATLAB geometry is not available, Ultralyzer still starts, but some metrics may be skipped.

</details>

<br>

<details>

<summary>Linux & MacOS</summary>

1. **Clone the Repository**:

   Open a terminal and run:

   ```bash
   git clone https://github.com/Borja21091/Ultralyzer-matlab.git
   cd Ultralyzer-matlab
   ```

2. **Set Up a Virtual Environment** (optional but recommended):

   ```bash
   conda create -n ultralyzer_env python=3.12
   conda activate ultralyzer_env
   ```

3. **Install Dependencies**:

   Run the following command in the terminal:

    ```bash
    chmod +x install.sh
    bash install.sh
    ```

4. **Select the Geometry Backend**:

   Set the MATLAB geometry backend:

   ```bash
   export ULTRALYZER_GEOMETRY_BACKENDS=matlab
   ```

5. **Run the Application**:

   ```bash
   python src/ultralyzer/main.py
   ```

   If MATLAB geometry is not available, Ultralyzer still starts, but some metrics may be skipped.

</details>
