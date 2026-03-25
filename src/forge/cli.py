"""forge CLI — Auto-optimize and deploy LLM models on local Apple Silicon."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from forge import __version__


@click.group()
@click.version_option(__version__, prog_name="forge")
def main():
    """forge — Auto-optimize LLM models for local Apple Silicon."""
    pass


@main.command()
@click.argument("model_id")
@click.option("--trust-remote-code", is_flag=True, help="Trust remote code from HuggingFace")
def analyze(model_id: str, trust_remote_code: bool):
    """Analyze a model and estimate memory requirements.

    MODEL_ID: HuggingFace model ID (e.g. meta-llama/Llama-3-8B)
    """
    from rich.console import Console  # type: ignore[import-untyped]

    console = Console()

    with console.status("[bold blue]Detecting hardware..."):
        from forge.analyzer import hardware_profiler

        hw = hardware_profiler.detect()

    with console.status("[bold blue]Inspecting model..."):
        from forge.analyzer import model_inspector

        try:
            model = model_inspector.inspect(model_id, trust_remote_code=trust_remote_code)
        except Exception as e:
            console.print(f"[red]Error inspecting model: {e}")
            sys.exit(1)

    with console.status("[bold blue]Calculating memory budget..."):
        from forge.analyzer import memory_calculator

        budget = memory_calculator.calculate(model, hw)

    with console.status("[bold blue]Selecting optimization strategy..."):
        from forge.optimizer import strategy_selector

        strategy = strategy_selector.select(model, hw, budget)

    # Print reports
    console.print()
    console.print(hardware_profiler.format_report(hw))
    console.print()
    console.print(model_inspector.format_report(model))
    console.print()
    console.print(memory_calculator.format_report(budget))
    console.print()
    console.print(strategy_selector.format_report(strategy))


@main.command()
@click.argument("model_id")
@click.option("--bits", type=int, default=None, help="Force quantization bits (2-8)")
@click.option("--method", type=click.Choice(["mlx_native", "hqq"]), default=None)
@click.option("--output", "-o", type=click.Path(), default=None, help="Output directory")
@click.option("--speculative", is_flag=True, help="Enable speculative decoding")
@click.option("--profile", is_flag=True, default=True, help="Run profiling after optimization")
@click.option("--trust-remote-code", is_flag=True)
def optimize(
    model_id: str,
    bits: int | None,
    method: str | None,
    output: str | None,
    speculative: bool,
    profile: bool,
    trust_remote_code: bool,
):
    """Download, convert, quantize, and optimize a model.

    MODEL_ID: HuggingFace model ID (e.g. meta-llama/Llama-3-8B)
    """
    from rich.console import Console

    console = Console()

    # 1. Analyze
    with console.status("[bold blue]Analyzing..."):
        from forge.analyzer import hardware_profiler, memory_calculator, model_inspector

        hw = hardware_profiler.detect()
        model = model_inspector.inspect(model_id, trust_remote_code=trust_remote_code)
        budget = memory_calculator.calculate(model, hw)

    # 2. Strategy
    with console.status("[bold blue]Selecting strategy..."):
        from forge.optimizer import strategy_selector

        force_quant = f"int{bits}" if bits else None
        strategy = strategy_selector.select(
            model, hw, budget,
            enable_speculative=speculative,
            force_quant=force_quant,
        )

    if not budget.can_run and not force_quant:
        console.print("[red]Model cannot fit in available memory.")
        console.print(memory_calculator.format_report(budget))
        sys.exit(1)

    console.print(strategy_selector.format_report(strategy))
    console.print()

    # Override method if specified
    if method:
        strategy.quant_method = method

    # 3. Build (convert + quantize)
    safe_name = model_id.replace("/", "--")
    quant_bits = bits or int(strategy.quantization.replace("int", "").replace("fp", ""))
    output_dir = Path(output) if output else Path(f"./optimized/{safe_name}-q{quant_bits}")

    with console.status(f"[bold green]Converting & quantizing ({strategy.quant_method} {strategy.quantization})..."):
        from forge.optimizer.quantizer import quantize

        result = quantize(
            model_id=model_id,
            output_dir=output_dir,
            method=strategy.quant_method,
            bits=quant_bits,
        )

    if not result.success:
        console.print(f"[red]Quantization failed: {result.error}")
        sys.exit(1)

    console.print(f"[green]Model saved to: {result.output_path} ({result.size_gb:.1f} GB)")

    # 4. Save config
    from forge.pipeline.deployer import save_config

    config_path = output_dir / "forge_config.yaml"
    save_config(
        {
            "model_id": model_id,
            "quantization": strategy.quantization,
            "method": strategy.quant_method,
            "runtime": strategy.runtime,
            "context_length": strategy.context_length,
            "estimated_tps": round(strategy.estimated_tps, 1),
            "estimated_memory_gb": round(strategy.estimated_memory_gb, 1),
        },
        model_path=output_dir,
        config_path=config_path,
    )

    # 5. Profile
    if profile:
        console.print()
        with console.status("[bold yellow]Running quick benchmark..."):
            from forge.optimizer.profiler import quick_bench

            metrics = quick_bench(output_dir, max_tokens=50, runtime=strategy.runtime)

        if metrics.error:
            console.print(f"[yellow]Profile warning: {metrics.error}")
        else:
            console.print(f"  TTFT:      {metrics.ttft_seconds:.2f}s")
            console.print(f"  Speed:     {metrics.tps:.1f} tok/s")
            console.print(f"  Tokens:    {metrics.tokens_generated}")

    console.print()
    console.print(f"[bold green]Optimization complete!")
    console.print(f"  Run: forge deploy {output_dir}")


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--host", default="127.0.0.1")
@click.option("--port", type=int, default=8080)
@click.option("--runtime", type=click.Choice(["mlx-lm", "ollama"]), default=None)
def deploy(model_path: str, host: str, port: int, runtime: str | None):
    """Start serving an optimized model.

    MODEL_PATH: Path to optimized model directory
    """
    from rich.console import Console

    console = Console()

    model_dir = Path(model_path)

    # Try to read forge config
    config_file = model_dir / "forge_config.yaml"
    selected_runtime = runtime or "mlx-lm"
    if config_file.exists():
        import yaml

        with open(config_file) as f:
            config = yaml.safe_load(f)
            if not runtime:
                selected_runtime = config.get("runtime", "mlx-lm")

    from forge.pipeline.deployer import DeployConfig, format_deploy_info

    deploy_config = DeployConfig(
        model_path=str(model_dir),
        runtime=selected_runtime,
        host=host,
        port=port,
    )

    console.print(format_deploy_info(deploy_config))
    console.print()

    if selected_runtime == "mlx-lm":
        from forge.pipeline.deployer import serve_mlx

        console.print(f"[bold green]Starting mlx-lm server on {host}:{port}...")
        console.print("Press Ctrl+C to stop.\n")
        try:
            proc = serve_mlx(model_dir, host=host, port=port)
            proc.wait()
        except KeyboardInterrupt:
            console.print("\n[yellow]Server stopped.")
    elif selected_runtime == "ollama":
        console.print("[yellow]Ollama deployment: use 'ollama run <model>' directly.")
    else:
        console.print(f"[red]Unsupported runtime: {selected_runtime}")


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--max-tokens", type=int, default=100)
@click.option("--runtime", default="mlx-lm")
@click.option("--save", type=click.Path(), default=None, help="Save results to JSON")
def bench(model_path: str, max_tokens: int, runtime: str, save: str | None):
    """Run benchmarks on an optimized model.

    MODEL_PATH: Path to optimized model directory
    """
    from rich.console import Console

    console = Console()

    with console.status("[bold blue]Running benchmarks..."):
        from forge.pipeline.benchmarker import format_report, run_benchmark, save_results

        result = run_benchmark(
            model_path=model_path,
            max_tokens=max_tokens,
            runtime=runtime,
        )

    console.print(format_report(result))

    if save:
        save_path = save_results(result, Path(save))
        console.print(f"\n[green]Results saved to: {save_path}")


@main.command(name="list")
@click.option("--dir", "optimized_dir", default="./optimized", help="Optimized models directory")
def list_models(optimized_dir: str):
    """List locally optimized models."""
    from rich.console import Console

    console = Console()
    base = Path(optimized_dir)

    if not base.exists():
        console.print(f"[yellow]No optimized models found in {base}")
        return

    models = [d for d in base.iterdir() if d.is_dir()]
    if not models:
        console.print(f"[yellow]No optimized models found in {base}")
        return

    console.print(f"Optimized Models ({base})")
    console.print("=" * 50)

    for model_dir in sorted(models):
        config_file = model_dir / "forge_config.yaml"
        info = model_dir.name

        if config_file.exists():
            import yaml

            with open(config_file) as f:
                config = yaml.safe_load(f) or {}
            quant = config.get("quantization", "?")
            tps = config.get("estimated_tps", "?")
            mem = config.get("estimated_memory_gb", "?")
            info = f"{model_dir.name}  [quant={quant}, ~{tps} tok/s, ~{mem}GB]"

        # Calculate directory size
        size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        size_gb = size / (1024**3)
        console.print(f"  {info}  ({size_gb:.1f} GB on disk)")


if __name__ == "__main__":
    main()
