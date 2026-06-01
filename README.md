# Ollama Benchmark

Ollama Benchmark is a terminal-based performance and benchmarking utility for local Large Language Models (LLMs) running on **Ollama**. It delivers detailed diagnostics regarding local model **disk footprints**, **GPU VRAM allocation**, **KV Cache scaling**, and **generation/prefill velocities** under heavy concurrent stress workloads.

The application provides hardware insights to help you optimize hardware capacity, allocate VRAM buffers, and evaluate parallel multi-user request performance.

## Features

*   **Hardware-Level VRAM Profiling**: Queries `nvidia-smi` directly for precise physical GPU memory metrics.
*   **Speed & Performance Diagnostics**: Calculates prompt prefill speed (tokens/s), token generation speed (tokens/s), wall-clock durations, and parallel load slowdown ratios.
*   **5-Stage Profiling**: Evaluates VRAM at Baseline (empty), Idle weight load, Active query spike, Saturated context thresholds, and Concurrency stress load.
*   **Automatic History Export**: Saves clean, time-stamped text logs to the `output/` folder, omitting CLI menu frames.

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
