# Ollama Benchmark

Ollama Benchmark is a terminal-based performance and benchmarking utility for local Large Language Models (LLMs) running on **Ollama**. It delivers detailed diagnostics regarding local model **disk footprints**, **GPU VRAM allocation**, **KV Cache scaling**, and **generation/prefill velocities** under heavy concurrent stress workloads.

The application provides hardware insights to help you optimize hardware capacity, allocate VRAM buffers, and evaluate parallel multi-user request performance.

## Features

*   **Hardware-Level VRAM Profiling**: Queries your NVIDIA graphics processor directly via `nvidia-smi` to profile physical memory usage rather than relying solely on API-reported values.
*   **Transient Spike Tracking (`PeakVRAMTracker`)**: Spawns a background thread polling every 50ms during active inference requests to capture transient memory spikes which are missed by standard interval-based polling.
*   **Velocity & Performance Diagnostics**:
    *   *Prompt Prefill Velocity*: Prompt processing rate in tokens/second (Prefill Phase).
    *   *Token Generation Velocity*: Generation rate in tokens/second (Decode Phase).
    *   *Wall-Clock Durations*: Process response timings for single-stream, saturated context, and parallel queries.
    *   *Concurrency Slowdown factor*: Calculates the performance degradation ratio under simultaneous multi-user loads.
*   **Comprehensive 5-Stage Profiling Suite**:
    1.  **Baseline**: Clears VRAM by dynamically unloading loaded models to measure initial system noise.
    2.  **Idle Weight Load**: Loads the model weights into GPU memory to establish inactive weight residency.
    3.  **Active Query**: Runs a single 50-token query to measure prompt evaluation prefill and decode VRAM spikes.
    4.  **Saturated Context**: Loads the model with customized context windows (e.g. 4096, 8192, 16384 tokens) and processes a saturated context prompt to profile prefill memory thresholds.
    5.  **Concurrency Stress**: Fires parallel requests concurrently (using thread pooling) to calculate parallel stream KV Cache expansion rate.
*   **Automatic History Export**: Dynamically creates an `output/` directory and logs clean ASCII-formatted plain-text execution reports prefixed with `YYYYMMDD_HHMMSS_<model_name>.txt` for audit history. All interactive menu screens are stripped from files, saving only the exact benchmark session.

### Quick Setup

1. First, you need to clone the repository:
    ```bash
    git clone https://github.com/ysfemreAlbyrk/ollama-benchmark.git
    cd ollama-benchmark
    ```

2. Install dependencies:
    ```bash
    uv sync

    # or

    python -m venv .venv
    ```

3. Activate the virtual environment:
    ```bash
    .venv\Scripts\activate   # Windows
    source .venv/bin/activate # Linux and macOS
    ```

4. Install dependencies:
    ```bash
    # If you used uv sync, you dont need to do anything.

    # If you used python -m venv .venv
    pip install -r requirements.txt
    ```


## How to Use

```bash
uv run .\benchmark.py

# or

python benchmark.py
```

### Run with Flags
Run commands directly with explicit flags if you are automating tests, scripting benchmarks, or bypass menus.

#### Basic run with defaults:
```bash
uv run .\benchmark.py run qwen3.5:0.8b
```

#### Run with customized concurrency limit, context window, and custom output path:

```bash
uv run .\benchmark.py run qwen3.5:0.8b --concurrency 4 --context-size 8192 --gpu 0 --output custom_report.txt
```

#### List Downloaded Local Models:

```bash
uv run .\benchmark.py list
```
---

This project is licensed under the MIT License.
