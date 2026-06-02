import os
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

def get_key():
    """Reads keyboard keystrokes on Windows for navigable TUI."""
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
    """Renders an interactive navigable menu using arrow keys and Enter with zero flicker."""
    from rich.live import Live
    from rich.console import Group
    from rich.text import Text
    
    current_index = default_index
    
    def generate_content():
        banner = """[bold cyan]  
  ██████╗ ██╗     ██╗      █████╗ ███╗   ███╗ █████╗     ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗
 ██╔═══██╗██║     ██║     ██╔══██╗████╗ ████║██╔══██╗    ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║
 ██║   ██║██║     ██║     ███████║██╔████╔██║███████║    ██████╔╝█████╗  ██╔██╗ ██║██║     ███████║
 ██║   ██║██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║    ██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║
 ╚██████╔╝███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║    ██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║
  ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝[/bold cyan]
[bold white]========================= OLLAMA VRAM & MEMORY PROFILING CLI =========================[/bold white]"""
        
        banner_text = Text.from_markup(banner)
        menu_text = Text()
        menu_text.append(f"\n🏠 {title}\n\n", style="bold cyan")
        
        for idx, option in enumerate(options):
            if idx == current_index:
                menu_text.append(f"  ➔ {option}\n", style="bold green underline")
            else:
                menu_text.append(f"    {option}\n")
                
        menu_text.append("\nUse Up/Down Arrow keys to navigate, press Enter to select.", style="dim")
        
        return Group(banner_text, menu_text)

    # Use Rich's Live alternate screen buffer for a flawless, 100% flicker-free full terminal view
    with Live(generate_content(), console=console, auto_refresh=False, screen=True) as live:
        while True:
            live.update(generate_content())
            live.refresh()
            
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
                # Fallback for non-interactive terminal pipelines
                live.stop()
                try:
                    choice = console.input(f"\nSelect option (1-{len(options)}): ").strip()
                    val = int(choice) - 1
                    if 0 <= val < len(options):
                        return val
                except (ValueError, KeyboardInterrupt, EOFError):
                    return -1


def interactive_menu():
    """Main menu loop for the interactive terminal user interface."""
    client = OllamaClient("http://localhost:11434")
    
    while True:
        options = [
            "📊 Run Benchmark on a Model",
            "📋 List Downloaded Ollama Models",
            "❌ Exit"
        ]
        
        choice_idx = show_menu("🏠 MAIN MENU", options)
        
        if choice_idx == 0:
            try:
                models = client.list_models()
                if not models:
                    os.system("cls" if os.name == "nt" else "clear")
                    print_banner()
                    console.print("[yellow]No local models found. Please download one using 'ollama pull <model>'[/yellow]\n")
                    console.input("Press Enter to continue...")
                    continue
                
                # Build model selection options list
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
                
                # Input custom values
                os.system("cls" if os.name == "nt" else "clear")
                
                # Clear previous keyboard menus drawing history from console record buffer
                if hasattr(console, "_record_buffer"):
                    console._record_buffer.clear()
                    
                # print_banner()
                console.print(f"[bold cyan]Selected Model:[/bold cyan] [green]{selected_model}[/green]\n")
                
                concurrency_input = console.input("[bold white]Concurrency count (parallel queries) [Default: 3]: [/bold white]").strip()
                concurrency = int(concurrency_input) if concurrency_input else 3
                
                ctx_input = console.input("[bold white]Context size (tokens) [Default: 4096]: [/bold white]").strip()
                context_size = int(ctx_input) if ctx_input else 4096
                
                gpu_input = console.input("[bold white]NVIDIA GPU ID to query [Default: 0]: [/bold white]").strip()
                gpu_id = int(gpu_input) if gpu_input else 0
                
                # Execute benchmark engine
                os.system("cls" if os.name == "nt" else "clear")
                gpu_monitor = GPUMonitor(gpu_id)
                
                if not gpu_monitor.is_available():
                    console.print("[bold red]❌ GPU Monitoring Error: nvidia-smi utility not found in PATH. Cannot profile VRAM.[/bold red]")
                    console.input("Press Enter to continue...")
                    continue
                    
                print_banner()
                console.print(Panel(
                    f"[bold white]Target Model:[/bold white] [cyan]{selected_model}[/cyan]\n"
                    f"[bold white]GPU Engine:[/bold white] [green]{gpu_monitor.get_gpu_info()['name'] if gpu_monitor.get_gpu_info() else 'NVIDIA GPU'}[/green] (ID: {gpu_id})\n"
                    f"[bold white]Context Window Limit:[/bold white] [yellow]{context_size} tokens[/yellow]\n"
                    f"[bold white]Concurrent Stress requests:[/bold white] [magenta]{concurrency}[/magenta]\n"
                    f"[bold white]Endpoint URL:[/bold white] http://localhost:11434",
                    title="Benchmark Configuration", border_style="cyan"
                ))
                
                engine = BenchmarkEngine(client, gpu_monitor, console)
                results = engine.run_benchmark(
                    model_name=selected_model,
                    concurrency=concurrency,
                    context_size=context_size
                )
                
                # Render rich formatted reports
                print_benchmark_report(results, engine, selected_model, concurrency, context_size)
                draw_vram_chart(results, engine)
                print_observations(results, engine, concurrency, context_size)
                export_report(selected_model)
                
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
