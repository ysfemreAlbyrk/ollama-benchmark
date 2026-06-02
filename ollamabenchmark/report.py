import os
from datetime import datetime
from rich.table import Table
from rich.panel import Panel
from .config import console
from .ollama_client import OllamaClient
from .engine import BenchmarkEngine

def print_banner():
    banner = """
    [bold cyan]  
  ██████╗ ██╗     ██╗      █████╗ ███╗   ███╗ █████╗     ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗
 ██╔═══██╗██║     ██║     ██╔══██╗████╗ ████║██╔══██╗    ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║
 ██║   ██║██║     ██║     ███████║██╔████╔██║███████║    ██████╔╝█████╗  ██╔██╗ ██║██║     ███████║
 ██║   ██║██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║    ██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║
 ╚██████╔╝███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║    ██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║
  ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝[/bold cyan]
    [bold white]========================= OLLAMA VRAM & MEMORY PROFILING CLI =========================[/bold white]
    """
    console.print(banner)


def list_models_cmd(client: OllamaClient):
    """Lists local Ollama models with their parameters and sizes."""
    try:
        models = client.list_models()
        if not models:
            console.print("[yellow]No local models found. Please download one using 'ollama pull <model>'[/yellow]")
            return
            
        table = Table(title="Available Local Ollama Models", show_header=True, header_style="bold magenta")
        table.add_column("Model Name", style="cyan", min_width=25)
        table.add_column("Parameter Size", justify="right", style="green")
        table.add_column("Quantization", justify="center", style="yellow")
        table.add_column("Disk Size", justify="right", style="blue")
        
        for m in models:
            details = m.get("details", {})
            size_bytes = m.get("size", 0)
            size_gb = size_bytes / (1024 * 1024 * 1024)
            
            table.add_row(
                m.get("name", "N/A"),
                details.get("parameter_size", "N/A"),
                details.get("quantization_level", "N/A"),
                f"{size_gb:.2f} GB"
            )
            
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Error querying models: {e}[/bold red]")


def draw_vram_chart(results: dict, engine: BenchmarkEngine):
    """Draws an elegant, styled ASCII bar chart showing VRAM escalation across states."""
    console.print("\n[bold magenta]📊 VRAM Escalation Profiler (VRAM added above baseline)[/bold magenta]")
    
    stages = [
        ("Idle Model Load", results["idle_vram_mib"]),
        ("Simple Query Active", results["simple_query_peak_mib"]),
        ("Saturated Context", results["full_context_peak_mib"]),
        ("Concurrent Queries", results["concurrent_peak_mib"])
    ]
    
    max_val = max(1, max(val for _, val in stages))
    bar_width = 40
    
    for label, val in stages:
        fill_count = int((val / max_val) * bar_width)
        bar = "█" * fill_count + "░" * (bar_width - fill_count)
        vram_str = engine.format_mib(val)
        console.print(f"  [cyan]{label:<22}[/cyan] | [bold green]{bar}[/bold green] | [bold yellow]{vram_str:>8}[/bold yellow]")


def print_benchmark_report(results: dict, engine: BenchmarkEngine, model_name: str, concurrency: int, context_size: int):
    """Builds and outputs the detailed VRAM footprint table and the token speed table."""
    
    # 1. Gorgeous summary table
    table = Table(title=f"📊 Benchmark Report for {model_name}", show_header=True, header_style="bold magenta", border_style="cyan")
    table.add_column("Measurement Metric", style="bold cyan")
    table.add_column("VRAM / Disk Allocation", justify="right", style="bold yellow")
    table.add_column("Description & Deep-dive Insights", style="dim white")
    
    # Format disk space
    disk_str = engine.format_bytes(results["disk_size_bytes"])
    table.add_row("Model File on Disk", disk_str, "Static space occupied by weights in storage.")
    table.add_row("System VRAM Baseline", engine.format_mib(results["baseline_vram_mib"]), "VRAM used by OS and background processes before load.")
    
    # Format idle
    idle_str = engine.format_mib(results["idle_vram_mib"])
    table.add_row(
        "Model VRAM (Idle Load)", 
        idle_str, 
        f"VRAM occupied when loaded to GPU. Ollama reported weight size: {engine.format_mib(results['idle_ollama_vram_mib'])}."
    )
    
    # Simple query
    simple_str = engine.format_mib(results["simple_query_peak_mib"])
    spike_simple = results["simple_query_peak_mib"] - results["idle_vram_mib"]
    table.add_row(
        "VRAM (Active Query)", 
        simple_str, 
        f"Peak VRAM during a single active query execution. (KV Cache / Runtime spike: +{spike_simple} MB)."
    )
    
    # Full context
    full_str = engine.format_mib(results["full_context_peak_mib"])
    spike_full = results["full_context_peak_mib"] - results["idle_vram_mib"]
    table.add_row(
        "VRAM (Full Context)", 
        full_str, 
        f"Peak VRAM with saturated context ({context_size} tokens). (KV Cache spike: +{spike_full} MB)."
    )
    
    # Concurrency
    concurrent_str = engine.format_mib(results["concurrent_peak_mib"])
    table.add_row(
        "VRAM (Concurrent Stress)", 
        concurrent_str, 
        f"Peak VRAM with {concurrency} parallel requests running concurrently."
    )
    
    # Cache per query
    table.add_row(
        "KV Cache per parallel slot", 
        f"~{results['concurrent_per_query_cache_mib']:.1f} MB", 
        "Approximate memory overhead allocated for each additional concurrent user stream."
    )
    
    console.print("\n")
    console.print(table)
    
    # 1.5 Speed and Performance Metrics Table
    speed_table = Table(title=f"⚡ Generation Speed & Performance for {model_name}", show_header=True, header_style="bold magenta", border_style="cyan")
    speed_table.add_column("Benchmark Phase", style="bold cyan")
    speed_table.add_column("Prompt Prefill Speed", justify="right", style="bold green")
    speed_table.add_column("Token Generation Speed", justify="right", style="bold green")
    speed_table.add_column("Total Duration", justify="right", style="bold yellow")
    speed_table.add_column("Performance Details", style="dim white")
    
    # Simple Query Row
    speed_table.add_row(
        "Simple Query (Single)",
        f"{results['simple_prompt_rate']:.2f} tok/s",
        f"{results['simple_gen_rate']:.2f} tok/s",
        f"{results['simple_total_time_ms'] / 1000.0:.2f} s",
        f"Prefill: {results['simple_prompt_tokens']} tokens in {results['simple_prompt_time_ms']:.1f}ms. Generated: {results['simple_gen_tokens']} tokens. Load: {results['simple_load_time_ms']:.1f}ms."
    )
    
    # Full Context Row
    speed_table.add_row(
        f"Saturated Context ({context_size} tok)",
        f"{results['full_prompt_rate']:.2f} tok/s",
        f"{results['full_gen_rate']:.2f} tok/s",
        f"{results['full_total_time_ms'] / 1000.0:.2f} s",
        f"Prefill: {results['full_prompt_tokens']} tokens in {results['full_prompt_time_ms']/1000.0:.2f}s."
    )
    
    # Concurrency Row
    slowdown_pct = 0.0
    if results['simple_gen_rate'] > 0:
        slowdown_pct = (1.0 - (results['concurrent_avg_gen_rate'] / results['simple_gen_rate'])) * 100.0
        
    speed_table.add_row(
        f"Concurrency Stress ({concurrency} Parallel)",
        "-",
        f"{results['concurrent_avg_gen_rate']:.2f} tok/s",
        f"{results['concurrent_total_time_ms'] / 1000.0:.2f} s",
        f"Average generation speed per stream. Slowdown: [bold yellow]{slowdown_pct:.1f}%[/bold yellow] under parallel load."
    )
    
    console.print("\n")
    console.print(speed_table)


def print_observations(results: dict, engine: BenchmarkEngine, concurrency: int, context_size: int):
    """Computes VRAM safety margins and theoretical concurrency capacity slot suggestions."""
    console.print("\n[bold cyan]💡 Key Observations & Recommendations[/bold cyan]")
    
    total_gpu = results["gpu_total_mib"]
    max_load = results["full_context_peak_mib"] + results["baseline_vram_mib"]
    margin = total_gpu - max_load
    
    if total_gpu > 0:
        console.print(f"  • [white]Total GPU VRAM Capacity:[/white] [bold yellow]{engine.format_mib(total_gpu)}[/bold yellow]")
        console.print(f"  • [white]Max System VRAM Under Full Load:[/white] [bold yellow]{engine.format_mib(max_load)}[/bold yellow] ({max_load/total_gpu*100:.1f}% of capacity)")
        
        if margin < 500:
            console.print("  • [bold red]⚠️ CRITICAL VRAM RISK:[/bold red] You have very low VRAM headroom under full load. Multiple concurrent queries or larger contexts could trigger CPU offloading, dropping performance significantly.")
        elif margin < 1500:
            console.print("  • [bold yellow]⚠️ MODERATE VRAM MARGIN:[/bold yellow] Keep context window or concurrency stress restricted to avoid running out of VRAM.")
        else:
            console.print("  • [bold green]✅ HEALTHY VRAM HEADROOM:[/bold green] You have comfortable headroom to run larger context sizes or increase parallel query counts.")
            
    # Cache insight
    cache_per_user = results['concurrent_per_query_cache_mib']
    if cache_per_user > 0:
        slots_left = int(margin / cache_per_user) if cache_per_user > 0 and margin > 0 else 0
        console.print(f"  • [white]Calculated Concurrency Capacity:[/white] Based on your KV Cache rate (~{cache_per_user:.1f} MB/slot), your GPU can theoretically support [bold green]~{slots_left + concurrency}[/bold green] concurrent parallel queries before hitting hardware memory thresholds.")
        
    console.print("\n[bold green]Benchmark completed successfully.[/bold green]\n")


def export_report(model_name: str, output_path: str = None):
    """Safely writes the console recording buffer to the output/ directory."""
    if not output_path:
        os.makedirs("output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_clean = model_name.replace(":", "_").replace("/", "_")
        output_path = os.path.join("output", f"{timestamp}_{model_clean}.txt")
        
    try:
        console.save_text(output_path, clear=False)
        console.print(f"[bold green]💾 Detailed text report exported to: {output_path}[/bold green]\n")
    except Exception as e:
        console.print(f"[bold red]⚠️ Failed to export report file: {e}[/bold red]\n")
