import argparse
import sys
from rich.panel import Panel
from .config import console
from .ollama_client import OllamaClient
from .gpu_monitor import GPUMonitor
from .engine import BenchmarkEngine
from .report import (
    print_banner,
    list_models_cmd,
    print_benchmark_report,
    draw_vram_chart,
    print_observations,
    export_report
)

def run_benchmark_cmd(args):
    """Executes the benchmark suite from command line flags."""
    client = OllamaClient(args.url)
    gpu_monitor = GPUMonitor(args.gpu)
    
    if not gpu_monitor.is_available():
        console.print("[bold red]❌ GPU Monitoring Error: nvidia-smi utility not found in system PATH. Cannot profile hardware VRAM.[/bold red]")
        sys.exit(1)
        
    print_banner()
    
    console.print(Panel(
        f"[bold white]Target Model:[/bold white] [cyan]{args.model}[/cyan]\n"
        f"[bold white]GPU Engine:[/bold white] [green]{gpu_monitor.get_gpu_info()['name'] if gpu_monitor.get_gpu_info() else 'NVIDIA GPU'}[/green] (ID: {args.gpu})\n"
        f"[bold white]Context Window Limit:[/bold white] [yellow]{args.context_size} tokens[/yellow]\n"
        f"[bold white]Concurrent Stress requests:[/bold white] [magenta]{args.concurrency}[/magenta]\n"
        f"[bold white]Endpoint URL:[/bold white] {args.url}",
        title="Benchmark Configuration", border_style="cyan"
    ))
    
    engine = BenchmarkEngine(client, gpu_monitor, console)
    
    try:
        results = engine.run_benchmark(
            model_name=args.model,
            concurrency=args.concurrency,
            context_size=args.context_size
        )
        
        # Render gorgeous outputs and charts
        print_benchmark_report(results, engine, args.model, args.concurrency, args.context_size)
        draw_vram_chart(results, engine)
        print_observations(results, engine, args.concurrency, args.context_size)
        export_report(args.model, args.output)
        
    except Exception as e:
        console.print(f"[bold red]Benchmark execution aborted: {e}[/bold red]")
        sys.exit(1)


def main():
    """Main CLI command route parsing options."""
    if len(sys.argv) == 1:
        from .tui import interactive_menu
        interactive_menu()
        return
        
    parser = argparse.ArgumentParser(description="Ollama VRAM and Memory Profiling Benchmark CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Subcommand 'list'
    subparsers.add_parser("list", help="List all available downloaded Ollama models with metadata")
    
    # Subcommand 'run'
    run_parser = subparsers.add_parser("run", help="Profile memory footprint of a specific Ollama model")
    run_parser.add_argument("model", type=str, help="Name of the model to benchmark (e.g. qwen3.5:0.8b)")
    run_parser.add_argument("--concurrency", type=int, default=3, help="Number of parallel queries for concurrency test (default: 3)")
    run_parser.add_argument("--context-size", type=int, default=4096, help="Target context window size for saturation (default: 4096)")
    run_parser.add_argument("--url", type=str, default="http://localhost:11434", help="Base URL of local Ollama instance (default: http://localhost:11434)")
    run_parser.add_argument("--gpu", type=int, default=0, help="NVIDIA GPU Index to query (default: 0)")
    run_parser.add_argument("--output", "-o", type=str, help="Path to save the plain text report file (e.g. report.txt)")
    
    args = parser.parse_args()
    
    if args.command == "list":
        client = OllamaClient(args.url if hasattr(args, "url") else "http://localhost:11434")
        list_models_cmd(client)
    elif args.command == "run":
        run_benchmark_cmd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
