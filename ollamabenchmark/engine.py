import time
import threading
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.status import Status
from .gpu_monitor import GPUMonitor
from .ollama_client import OllamaClient
from .config import console as default_console

class PeakVRAMTracker:
    """
    Context manager that spawns a background thread to poll nvidia-smi 
    at a high frequency, capturing the absolute peak VRAM usage during a workload.
    """
    def __init__(self, monitor: GPUMonitor, interval: float = 0.05):
        self.monitor = monitor
        self.interval = interval
        self.peak_vram = 0
        self.active = False
        self._thread = None

    def __enter__(self):
        self.peak_vram = self.monitor.get_used_vram()
        self.active = True
        self._thread = threading.Thread(target=self._poll_loop)
        self._thread.daemon = True
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.active = False
        if self._thread:
            self._thread.join()

    def _poll_loop(self):
        while self.active:
            vram = self.monitor.get_used_vram()
            if vram > self.peak_vram:
                self.peak_vram = vram
            time.sleep(self.interval)


class BenchmarkEngine:
    def __init__(self, client: OllamaClient, gpu_monitor: GPUMonitor, console: Optional[Console] = None):
        self.client = client
        self.gpu_monitor = gpu_monitor
        self.console = console or default_console

    def format_bytes(self, num_bytes: float) -> str:
        """Utility to format bytes to human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if num_bytes < 1024.0:
                return f"{num_bytes:.2f} {unit}"
            num_bytes /= 1024.0
        return f"{num_bytes:.2f} PB"

    def format_mib(self, mib: float) -> str:
        """Utility to format MiB to human-readable GB/MB string."""
        if mib >= 1024:
            return f"{mib / 1024:.2f} GB"
        return f"{int(mib)} MB"

    def run_benchmark(self, model_name: str, concurrency: int = 3, context_size: int = 4096) -> Dict[str, Any]:
        """
        Runs the full suite of benchmarks on the given model name.
        """
        results = {
            "model_name": model_name,
            "disk_size_bytes": 0,
            "parameter_size": "N/A",
            "quantization": "N/A",
            "baseline_vram_mib": 0,
            "idle_vram_mib": 0,
            "idle_ollama_vram_mib": 0,
            "simple_query_peak_mib": 0,
            "full_context_peak_mib": 0,
            "concurrent_peak_mib": 0,
            "concurrent_per_query_cache_mib": 0,
            "gpu_name": "N/A",
            "gpu_total_mib": 0,

            # Speed & Performance Metrics
            "simple_prompt_tokens": 0,
            "simple_prompt_time_ms": 0.0,
            "simple_prompt_rate": 0.0,
            "simple_gen_tokens": 0,
            "simple_gen_time_ms": 0.0,
            "simple_gen_rate": 0.0,
            "simple_total_time_ms": 0.0,
            "simple_load_time_ms": 0.0,

            "full_prompt_tokens": 0,
            "full_prompt_time_ms": 0.0,
            "full_prompt_rate": 0.0,
            "full_gen_tokens": 0,
            "full_gen_time_ms": 0.0,
            "full_gen_rate": 0.0,
            "full_total_time_ms": 0.0,

            "concurrent_avg_gen_rate": 0.0,
            "concurrent_total_time_ms": 0.0,
        }

        # 0. Get GPU Name & Total VRAM
        gpu_info = self.gpu_monitor.get_gpu_info()
        if gpu_info:
            results["gpu_name"] = gpu_info["name"]
            results["gpu_total_mib"] = gpu_info["total_vram"]

        # 1. Fetch Model Info from local tags
        models = self.client.list_models()
        model_meta = next((m for m in models if m["name"] == model_name or m["model"] == model_name), None)
        if not model_meta:
            raise ValueError(f"Model '{model_name}' is not downloaded in Ollama. Please run 'ollama pull {model_name}' first.")
        
        results["disk_size_bytes"] = model_meta.get("size", 0)
        details = model_meta.get("details", {})
        results["parameter_size"] = details.get("parameter_size", "N/A")
        results["quantization"] = details.get("quantization_level", "N/A")

        # 2. Unload models and establish clean Baseline
        with Status("[bold cyan]Cleaning VRAM baseline...", console=self.console) as status:
            unloaded = self.client.unload_all_models()
            if unloaded > 0:
                self.console.log(f"Unloaded {unloaded} active model(s) to prepare baseline.")
            else:
                self.console.log("VRAM already clear (no models loaded).")
            
            # Wait for VRAM deallocation to settle
            time.sleep(2.0)
            baseline = self.gpu_monitor.get_used_vram()
            results["baseline_vram_mib"] = baseline
            self.console.log(f"Baseline System VRAM: [bold yellow]{self.format_mib(baseline)}[/bold yellow]")

        # 3. Idle Load Benchmark
        with Status(f"[bold cyan]Loading {model_name} in VRAM (Idle state)...", console=self.console) as status:
            # Load the model with default options, keep alive for 5 minutes
            self.client.generate(model_name, prompt="", keep_alive=300)
            time.sleep(2.0)  # Settle down
            
            idle_now = self.gpu_monitor.get_used_vram()
            results["idle_vram_mib"] = max(0, idle_now - baseline)
            
            # Query Ollama's reported VRAM
            running = self.client.get_running_models()
            model_ps = next((m for m in running if m["name"] == model_name or m["model"] == model_name), None)
            if model_ps:
                results["idle_ollama_vram_mib"] = int(model_ps.get("size_vram", 0) / (1024 * 1024))
            
            self.console.log(f"Model loaded. VRAM Added (Idle): [bold yellow]{self.format_mib(results['idle_vram_mib'])}[/bold yellow] (Ollama reported: {self.format_mib(results['idle_ollama_vram_mib'])})")

        # 4. Simple Query Benchmark
        with Status("[bold cyan]Running simple query test (single prompt)...", console=self.console) as status:
            prompt = "Write a 5-sentence paragraph explaining why the sky is blue."
            
            with PeakVRAMTracker(self.gpu_monitor) as tracker:
                resp = self.client.generate(model_name, prompt, options={"num_predict": 20}, keep_alive=300)
                
            results["simple_query_peak_mib"] = max(0, tracker.peak_vram - baseline)
            
            # Extract speed metrics
            results["simple_prompt_tokens"] = resp.get("prompt_eval_count", 0)
            results["simple_prompt_time_ms"] = resp.get("prompt_eval_duration", 0) / 1_000_000
            results["simple_prompt_rate"] = (results["simple_prompt_tokens"] / (results["simple_prompt_time_ms"] / 1000.0)) if results["simple_prompt_time_ms"] > 0 else 0.0
            
            results["simple_gen_tokens"] = resp.get("eval_count", 0)
            results["simple_gen_time_ms"] = resp.get("eval_duration", 0) / 1_000_000
            results["simple_gen_rate"] = (results["simple_gen_tokens"] / (results["simple_gen_time_ms"] / 1000.0)) if results["simple_gen_time_ms"] > 0 else 0.0
            
            results["simple_total_time_ms"] = resp.get("total_duration", 0) / 1_000_000
            results["simple_load_time_ms"] = resp.get("load_duration", 0) / 1_000_000

            self.console.log(f"Simple query peak VRAM Added: [bold yellow]{self.format_mib(results['simple_query_peak_mib'])}[/bold yellow] (Spike: {self.format_mib(results['simple_query_peak_mib'] - results['idle_vram_mib'])})")
            self.console.log(f"Prompt Eval Rate: [bold green]{results['simple_prompt_rate']:.2f} tok/s[/bold green] | Generation Rate: [bold green]{results['simple_gen_rate']:.2f} tok/s[/bold green]")

        # 5. Full Context Benchmark
        with Status(f"[bold cyan]Running full context saturation test ({context_size} tokens)...", console=self.console) as status:
            # Standard estimation: 1 token = 4 characters. Let's create a large prompt to fill context.
            # We construct repeating paragraphs to match (context_size * 4) characters
            paragraph = "Ollama is a lightweight, extensible framework for building and running language models on local machines. "
            repeat_count = int((context_size * 4) / len(paragraph)) + 1
            long_prompt = (paragraph * repeat_count)[:context_size * 4]
            long_prompt += "\nSummarize the paragraph above in exactly one word."
            
            # Configure context size and limit prediction to make prefill test ultra-fast
            options = {"num_ctx": context_size, "num_predict": 5}
            
            # Since loading a new large context might re-allocate the KV cache, we track VRAM
            resp = {}
            with PeakVRAMTracker(self.gpu_monitor) as tracker:
                try:
                    resp = self.client.generate(model_name, long_prompt, options=options, keep_alive=300)
                except Exception as e:
                    self.console.log(f"[bold red]Full context query error (might have run out of VRAM/Memory): {e}[/bold red]")
                    
            results["full_context_peak_mib"] = max(0, tracker.peak_vram - baseline)
            
            # Extract speed metrics
            if resp:
                results["full_prompt_tokens"] = resp.get("prompt_eval_count", 0)
                results["full_prompt_time_ms"] = resp.get("prompt_eval_duration", 0) / 1_000_000
                results["full_prompt_rate"] = (results["full_prompt_tokens"] / (results["full_prompt_time_ms"] / 1000.0)) if results["full_prompt_time_ms"] > 0 else 0.0
                
                results["full_gen_tokens"] = resp.get("eval_count", 0)
                results["full_gen_time_ms"] = resp.get("eval_duration", 0) / 1_000_000
                results["full_gen_rate"] = (results["full_gen_tokens"] / (results["full_gen_time_ms"] / 1000.0)) if results["full_gen_time_ms"] > 0 else 0.0
                results["full_total_time_ms"] = resp.get("total_duration", 0) / 1_000_000

            self.console.log(f"Full context peak VRAM Added: [bold yellow]{self.format_mib(results['full_context_peak_mib'])}[/bold yellow] (Spike from Idle: {self.format_mib(results['full_context_peak_mib'] - results['idle_vram_mib'])})")
            if resp:
                self.console.log(f"Context Prefill Rate: [bold green]{results['full_prompt_rate']:.2f} tok/s[/bold green] (Tokens: {results['full_prompt_tokens']})")

        # 6. Concurrency Benchmark
        with Status(f"[bold cyan]Running concurrency test ({concurrency} parallel queries)...", console=self.console) as status:
            prompt = "What are the three laws of thermodynamics?"
            
            resps = []
            start_time = time.time()
            with PeakVRAMTracker(self.gpu_monitor) as tracker:
                try:
                    # Fire concurrent queries with small token generation limits to complete instantly
                    resps = self.client.generate_concurrent(model_name, prompt, count=concurrency, options={"num_predict": 20})
                except Exception as e:
                    self.console.log(f"[bold red]Concurrency query error: {e}[/bold red]")
                    
            results["concurrent_peak_mib"] = max(0, tracker.peak_vram - baseline)
            results["concurrent_total_time_ms"] = (time.time() - start_time) * 1000.0
            
            # Calculate generation rates for concurrent queries
            rates = []
            for r in resps:
                if isinstance(r, dict) and "error" not in r:
                    g_tokens = r.get("eval_count", 0)
                    g_time_ms = r.get("eval_duration", 0) / 1_000_000
                    if g_time_ms > 0:
                        rates.append(g_tokens / (g_time_ms / 1000.0))
            
            if rates:
                results["concurrent_avg_gen_rate"] = sum(rates) / len(rates)
            else:
                results["concurrent_avg_gen_rate"] = 0.0

            # Calculate cache overhead per concurrent query
            base_active = results["simple_query_peak_mib"]
            total_active = results["concurrent_peak_mib"]
            
            if concurrency > 1 and total_active > base_active:
                results["concurrent_per_query_cache_mib"] = max(0, (total_active - base_active) / (concurrency - 1))
            else:
                results["concurrent_per_query_cache_mib"] = 0
                
            self.console.log(f"Concurrent peak VRAM Added: [bold yellow]{self.format_mib(results['concurrent_peak_mib'])}[/bold yellow]")
            if rates:
                self.console.log(f"Avg Concurrency Generation Rate: [bold green]{results['concurrent_avg_gen_rate']:.2f} tok/s[/bold green] (per stream)")
            self.console.log(f"VRAM Cache overhead per extra concurrent query: [bold green]~{results['concurrent_per_query_cache_mib']:.1f} MB[/bold green]")

        # 7. Unload models to restore original state
        with Status("[bold cyan]Benchmarking complete. Cleaning up VRAM...", console=self.console) as status:
            self.client.unload_all_models()
            time.sleep(1.5)

        return results
