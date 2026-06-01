import argparse
import sys
import os
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from ollama_client import OllamaClient
from gpu_monitor import GPUMonitor
from engine import BenchmarkEngine

console = Console(record=True)

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


def run_benchmark_cmd(args):
    """Executes the benchmark suite."""
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
        
        # 1. Print gorgeous summary table
        table = Table(title=f"📊 Benchmark Report for {args.model}", show_header=True, header_style="bold magenta", border_style="cyan")
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
            f"Peak VRAM with saturated context ({args.context_size} tokens). (KV Cache spike: +{spike_full} MB)."
        )
        
        # Concurrency
        concurrent_str = engine.format_mib(results["concurrent_peak_mib"])
        table.add_row(
            "VRAM (Concurrent Stress)", 
            concurrent_str, 
            f"Peak VRAM with {args.concurrency} parallel requests running concurrently."
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
        speed_table = Table(title=f"⚡ Generation Speed & Performance for {args.model}", show_header=True, header_style="bold magenta", border_style="cyan")
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
            f"Saturated Context ({args.context_size} tok)",
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
            f"Concurrency Stress ({args.concurrency} Parallel)",
            "-",
            f"{results['concurrent_avg_gen_rate']:.2f} tok/s",
            f"{results['concurrent_total_time_ms'] / 1000.0:.2f} s",
            f"Average generation speed per stream. Slowdown: [bold yellow]{slowdown_pct:.1f}%[/bold yellow] under parallel load."
        )
        
        console.print("\n")
        console.print(speed_table)
        
        # 2. Draw bar chart
        draw_vram_chart(results, engine)
        
        # 3. Print recommendations and takeaways
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
            console.print(f"  • [white]Calculated Concurrency Capacity:[/white] Based on your KV Cache rate (~{cache_per_user:.1f} MB/slot), your GPU can theoretically support [bold green]~{slots_left + args.concurrency}[/bold green] concurrent parallel queries before hitting hardware memory thresholds.")
            
        console.print("\n[bold green]Benchmark completed successfully.[/bold green]\n")
        
        # Exporting to file
        output_file = args.output if hasattr(args, 'output') else None
        if not output_file:
            os.makedirs("output", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_clean = args.model.replace(":", "_").replace("/", "_")
            output_file = os.path.join("output", f"{timestamp}_{model_clean}.txt")
            
        try:
            console.save_text(output_file, clear=False)
            console.print(f"[bold green]💾 Detailed text report exported to: {output_file}[/bold green]\n")
        except Exception as e:
            console.print(f"[bold red]⚠️ Failed to export report file: {e}[/bold red]\n")
        
    except Exception as e:
        console.print(f"[bold red]Benchmark execution aborted: {e}[/bold red]")
        sys.exit(1)


def get_key():
    try:
        import msvcrt
        ch = msvcrt.getch()
        if ch in (b'\x00', b'\xe0'):
            ch2 = msvcrt.getch()
            if ch2 == b'H':  # Up Arrow
                return "up"
            elif ch2 == b'P':  # Down Arrow
                return "down"
        elif ch == b'\r':  # Enter
            return "enter"
        elif ch == b'\x1b':  # Esc
            return "esc"
    except (ImportError, AttributeError):
        pass
    return None


def show_menu(title: str, options: list, default_index: int = 0) -> int:
    """
    Renders an interactive navigable menu using arrow keys and Enter.
    """
    current_index = default_index
    
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print_banner()
        console.print(f"[bold cyan]{title}[/bold cyan]\n")
        
        for idx, option in enumerate(options):
            if idx == current_index:
                console.print(f"  [bold green]➔ [underline]{option}[/underline][/bold green]")
            else:
                console.print(f"    {option}")
                
        console.print("\n[dim]Use Up/Down Arrow keys to navigate, press Enter to select.[/dim]")
        
        key = get_key()
        if key == "up":
            current_index = (current_index - 1) % len(options)
        elif key == "down":
            current_index = (current_index + 1) % len(options)
        elif key == "enter":
            return current_index
        elif key == "esc":
            return -1
        elif key is None:
            # Fallback for non-interactive or non-windows environments
            try:
                choice = console.input(f"\nSelect option (1-{len(options)}): ").strip()
                val = int(choice) - 1
                if 0 <= val < len(options):
                    return val
            except (ValueError, KeyboardInterrupt, EOFError):
                return -1


def interactive_menu():
    client = OllamaClient("http://localhost:11434")
    
    while True:
        options = [
            "📊 Run Benchmark on a Model",
            "📋 List Downloaded Ollama Models",
            "❌ Exit"
        ]
        
        choice_idx = show_menu("🏠 MAIN MENU", options)
        
        if choice_idx == 0:
            # Run benchmark interactive flow
            try:
                models = client.list_models()
                if not models:
                    os.system("cls" if os.name == "nt" else "clear")
                    print_banner()
                    console.print("[yellow]No local models found. Please download one using 'ollama pull <model>'[/yellow]\n")
                    console.input("Press Enter to continue...")
                    continue
                
                # Build model selection menu
                model_options = []
                for m in models:
                    size_bytes = m.get("size", 0)
                    size_gb = size_bytes / (1024 * 1024 * 1024)
                    model_options.append(f"{m['name']:<40} ({size_gb:.2f} GB)")
                model_options.append("⬅️ Back to Main Menu")
                
                model_idx = show_menu("Select a model to benchmark:", model_options)
                
                if model_idx == -1 or model_idx == len(models):
                    continue
                    
                selected_model = models[model_idx]["name"]
                
                # Prompt for custom inputs with nice defaults
                os.system("cls" if os.name == "nt" else "clear")
                
                # Wipe previous interactive menu rendering history from the recording buffer
                if hasattr(console, "_record_buffer"):
                    console._record_buffer.clear()
                    
                print_banner()
                console.print(f"[bold cyan]Selected Model:[/bold cyan] [green]{selected_model}[/green]\n")
                
                concurrency_input = console.input("[bold white]Concurrency count (parallel queries) [Default: 3]: [/bold white]").strip()
                concurrency = int(concurrency_input) if concurrency_input else 3
                
                ctx_input = console.input("[bold white]Context size (tokens) [Default: 4096]: [/bold white]").strip()
                context_size = int(ctx_input) if ctx_input else 4096
                
                gpu_input = console.input("[bold white]NVIDIA GPU ID to query [Default: 0]: [/bold white]").strip()
                gpu_id = int(gpu_input) if gpu_input else 0
                
                # Mock args to call run_benchmark_cmd
                class MockArgs:
                    def __init__(self, model, concurrency, context_size, gpu, url="http://localhost:11434", output=None):
                        self.model = model
                        self.concurrency = concurrency
                        self.context_size = context_size
                        self.gpu = gpu
                        self.url = url
                        self.output = output
                        
                mock_args = MockArgs(selected_model, concurrency, context_size, gpu_id)
                
                # Clear and execute benchmark
                os.system("cls" if os.name == "nt" else "clear")
                run_benchmark_cmd(mock_args)
                console.input("Press Enter to return to main menu...")
                
            except Exception as e:
                console.print(f"[bold red]Interactive benchmark failed: {e}[/bold red]\n")
                console.input("Press Enter to continue...")
                
        elif choice_idx == 1:
            os.system("cls" if os.name == "nt" else "clear")
            print_banner()
            list_models_cmd(client)
            console.input("\nPress Enter to return to main menu...")
            
        elif choice_idx == 2 or choice_idx == -1:
            console.print("[cyan]Exiting. Good bye![/cyan]")
            break


def main():
    if len(sys.argv) == 1:
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
