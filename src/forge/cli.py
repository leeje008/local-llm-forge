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

    # ANE assessment
    from forge.engine.ane_hybrid import assess_ane_compatibility, format_ane_report

    ane_profile = assess_ane_compatibility(
        model.total_params_b, model.architecture, hw.total_memory_gb, hw.ane_tops,
    )
    console.print()
    console.print(format_ane_report(ane_profile))

    # Routing analysis
    from forge.router.feasibility import format_route_report, route

    decision = route(model, hw, budget)
    console.print()
    console.print(format_route_report(decision))


@main.command()
@click.argument("model_id")
@click.option("--trust-remote-code", is_flag=True)
def route(model_id: str, trust_remote_code: bool):
    """Analyze model feasibility and show all execution paths.

    MODEL_ID: HuggingFace model ID (e.g. meta-llama/Llama-3.1-70B)
    """
    from rich.console import Console

    console = Console()

    with console.status("[bold blue]Analyzing model feasibility..."):
        from forge.analyzer import hardware_profiler, memory_calculator, model_inspector

        hw = hardware_profiler.detect()
        try:
            model = model_inspector.inspect(model_id, trust_remote_code=trust_remote_code)
        except Exception as e:
            console.print(f"[red]Error: {e}")
            sys.exit(1)
        budget = memory_calculator.calculate(model, hw)

    from forge.router.feasibility import format_route_report
    from forge.router.feasibility import route as do_route

    decision = do_route(model, hw, budget)

    console.print()
    console.print(f"[bold]{model_id}[/bold] — {model.total_params_b:.1f}B params, {model.model_type.upper()}")
    console.print()
    console.print(format_route_report(decision))


@main.command()
@click.argument("model_id")
@click.option("--bits", type=int, default=None, help="Force quantization bits (2-8)")
@click.option("--method", type=click.Choice(["mlx_native", "hqq"]), default=None)
@click.option("--output", "-o", type=click.Path(), default=None, help="Output directory")
@click.option("--speculative", is_flag=True, help="Enable speculative decoding")
@click.option("--mixed-precision", "mixed_prec", is_flag=True, help="Per-layer mixed-precision quantization")
@click.option("--profile", is_flag=True, default=True, help="Run profiling after optimization")
@click.option("--trust-remote-code", is_flag=True)
def optimize(
    model_id: str,
    bits: int | None,
    method: str | None,
    output: str | None,
    speculative: bool,
    mixed_prec: bool,
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
        console.print("[yellow]Model cannot fit in available memory with standard quantization.")
        console.print()

        # Show routing alternatives instead of hard exit
        from forge.router.feasibility import format_route_report
        from forge.router.feasibility import route as do_route

        decision = do_route(model, hw, budget)
        console.print(format_route_report(decision))
        console.print()
        console.print("[yellow]Use one of the suggested commands above, or force with --bits flag.")
        sys.exit(1)

    console.print(strategy_selector.format_report(strategy))
    console.print()

    # Override method if specified
    if method:
        strategy.quant_method = method

    # 3. Build (convert + quantize)
    safe_name = model_id.replace("/", "--")
    quant_bits = bits or int(strategy.quantization.replace("int", "").replace("fp", ""))

    if mixed_prec:
        # Mixed-precision path
        output_dir = Path(output) if output else Path(f"./optimized/{safe_name}-mixed")
        with console.status("[bold green]Analyzing layer sensitivity for mixed-precision..."):
            from forge.optimizer.mixed_precision import (
                analyze_and_allocate,
                apply_mixed_quantization,
                format_plan_report,
            )
            plan = analyze_and_allocate(model_id, target_avg_bits=float(quant_bits))

        console.print(format_plan_report(plan))
        console.print()

        with console.status("[bold green]Applying mixed-precision quantization..."):
            ok, msg = apply_mixed_quantization(model_id, plan, output_dir)

        if not ok:
            console.print(f"[red]Mixed-precision failed: {msg}")
            console.print("[yellow]Falling back to uniform quantization...")
            # Fall through to standard path
            mixed_prec = False

    if not mixed_prec:
        output_dir = Path(output) if output else Path(f"./optimized/{safe_name}-q{quant_bits}")

    if mixed_prec and ok:
        # Mixed-precision already applied above; create a QuantResult-like object
        from forge.optimizer.quantizer import QuantResult, _dir_size_gb
        result = QuantResult(
            output_path=output_dir, quant=f"mixed_avg{plan.avg_bits:.0f}",
            method="mixed_precision", size_gb=_dir_size_gb(output_dir), success=True,
        )
    else:
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
@click.argument("prompt")
@click.option("--max-tokens", type=int, default=256)
@click.option("--draft", type=click.Path(), default=None, help="Draft model for speculative decoding")
@click.option("--auto-draft", is_flag=True, help="Auto-select draft model")
@click.option("--redrafter", is_flag=True, help="Use ReDrafter for speculative decoding")
@click.option("--temperature", type=float, default=0.7)
@click.option("--kv-bits", type=click.Choice(["4", "8"]), default=None, help="KV cache quantization bits")
@click.option("--max-kv-size", type=int, default=None, help="Sliding window KV cache limit")
@click.option("--stream/--no-stream", default=True)
def run(model_path: str, prompt: str, max_tokens: int, draft: str | None,
        auto_draft: bool, redrafter: bool, temperature: float, kv_bits: str | None,
        max_kv_size: int | None, stream: bool):
    """Generate text using the MLX engine with optional speculative decoding.

    MODEL_PATH: Path to optimized model
    PROMPT: Text prompt
    """
    from rich.console import Console

    console = Console()

    draft_path = draft
    if redrafter and not draft_path:
        # ReDrafter selection (priority over auto-draft)
        config_file = Path(model_path) / "forge_config.yaml"
        arch = model_id = ""
        if config_file.exists():
            import yaml
            with open(config_file) as f:
                cfg = yaml.safe_load(f) or {}
            model_id = cfg.get("model_id", "")
            arch = model_id.lower()

        from forge.engine.redrafter import format_redrafter_info, select_redrafter
        rd_info = select_redrafter(arch, model_id)
        draft_path = rd_info.model_id
        console.print(format_redrafter_info(rd_info))

    elif auto_draft and not draft_path:
        config_file = Path(model_path) / "forge_config.yaml"
        arch = "unknown"
        if config_file.exists():
            import yaml
            with open(config_file) as f:
                cfg = yaml.safe_load(f) or {}
            model_id = cfg.get("model_id", "")
            arch = model_id.lower()

        from forge.engine.speculative import select_draft_model
        draft_info = select_draft_model(arch)
        if draft_info:
            draft_path = draft_info.model_id
            console.print(f"[blue]Auto-selected draft: {draft_info.model_id} ({draft_info.source})")

    from forge.engine.mlx_engine import EngineConfig, MLXEngine

    config = EngineConfig(
        model_path=model_path,
        max_tokens=max_tokens,
        temperature=temperature,
        draft_model_path=draft_path,
        kv_bits=int(kv_bits) if kv_bits else None,
        max_kv_size=max_kv_size,
    )

    with console.status("[bold blue]Loading model..."):
        engine = MLXEngine(config)
        engine.load()

    if stream:
        console.print()
        for chunk in engine.stream(prompt, max_tokens=max_tokens, temperature=temperature):
            console.print(chunk, end="")
        console.print()
    else:
        result = engine.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        console.print()
        console.print(result.text)
        console.print()
        console.print(f"[dim]{result.tokens_generated} tokens, {result.tps:.1f} tok/s, "
                      f"TTFT {result.ttft_seconds:.2f}s"
                      f"{', speculative' if result.speculative_used else ''}[/dim]")


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
        from forge.pipeline.deployer import create_ollama_model

        # Read model_id from forge config
        model_id = config.get("model_id", "") if config_file.exists() else ""
        if model_id:
            with console.status("[bold green]Registering with Ollama..."):
                ctx = config.get("context_length", 4096) if config_file.exists() else 4096
                ok, msg = create_ollama_model(model_id, context_length=ctx)
            if ok:
                console.print(f"[green]{msg}")
            else:
                console.print(f"[red]{msg}")
        else:
            console.print("[yellow]No model_id in config. Use: ollama run hf.co/<model>")
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


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--prompt", "-p", required=True, help="System prompt to cache")
@click.option("--output", "-o", type=click.Path(), default=None, help="Cache output path")
def cache(model_path: str, prompt: str, output: str | None):
    """Pre-compute KV cache for a system prompt (zero-latency reuse).

    MODEL_PATH: Path to optimized model directory
    """
    from rich.console import Console

    console = Console()

    with console.status("[bold blue]Computing prompt cache..."):
        from forge.engine.prompt_cache import cache_prompt

        cache_out = Path(output) if output else None
        result = cache_prompt(model_path, prompt, cache_out)

    if result.success:
        console.print(f"[green]Prompt cached: {result.cache_path} ({result.size_mb:.1f} MB)")
    else:
        console.print(f"[red]Cache failed: {result.error}")


@main.command(name="cache-list")
@click.argument("model_path", type=click.Path(exists=True))
def cache_list(model_path: str):
    """List cached prompts for a model."""
    from rich.console import Console

    console = Console()

    from forge.engine.prompt_cache import list_caches

    caches = list_caches(model_path)
    if not caches:
        console.print("[yellow]No cached prompts found.")
        return

    console.print("Cached Prompts")
    console.print("=" * 50)
    for c in caches:
        console.print(f"  {c['name']}  ({c['size_mb']} MB)")


@main.command(name="expert-prune")
@click.argument("model_id")
@click.option("--ratio", type=float, default=0.25, help="Fraction of experts to prune (0.25=25%)")
@click.option("--method", type=click.Choice(["remove", "merge"]), default="remove")
@click.option("--num-samples", type=int, default=20, help="Calibration prompts")
@click.option("--save-plan", type=click.Path(), default=None, help="Save pruning plan JSON")
def expert_prune(model_id: str, ratio: float, method: str, num_samples: int, save_plan: str | None):
    """Analyze MoE expert importance and create pruning plan.

    MODEL_ID: HuggingFace MoE model ID (e.g. mistralai/Mixtral-8x7B-Instruct-v0.1)
    """
    from rich.console import Console

    console = Console()

    # Verify MoE model
    with console.status("[bold blue]Inspecting model..."):
        from forge.analyzer.model_inspector import inspect

        model = inspect(model_id, trust_remote_code=True)

    if model.model_type != "moe":
        console.print(f"[red]{model_id} is not a MoE model (type={model.model_type}). Expert pruning requires MoE.")
        sys.exit(1)

    console.print(f"[blue]MoE model: {model.num_experts} experts, {model.num_active_experts} active/token")

    with console.status(f"[bold blue]Capturing expert activations ({num_samples} samples)..."):
        from forge.optimizer.expert_pruner import (
            analyze_expert_importance,
            create_pruning_plan,
            format_pruning_plan,
            save_pruning_plan,
        )

        rankings = analyze_expert_importance(model_id, num_samples=num_samples)

    if not rankings:
        console.print("[red]Failed to capture expert activations. Model may not have accessible gate modules.")
        sys.exit(1)

    plan = create_pruning_plan(
        model_id=model_id,
        rankings=rankings,
        prune_ratio=ratio,
        method=method,
        num_experts=model.num_experts or 8,
    )

    console.print()
    console.print(format_pruning_plan(plan))

    if save_plan:
        save_pruning_plan(plan, Path(save_plan))
        console.print(f"\n[green]Pruning plan saved to: {save_plan}")


@main.command(name="eval")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--suite", type=click.Choice(["quick", "standard", "reasoning", "code", "full"]),
              default="quick")
@click.option("--limit", type=int, default=None, help="Limit samples per task")
@click.option("--save", type=click.Path(), default=None)
def eval_model(model_path: str, suite: str, limit: int | None, save: str | None):
    """Evaluate model quality using standard benchmarks.

    MODEL_PATH: Path to optimized model
    """
    from rich.console import Console

    console = Console()

    with console.status(f"[bold blue]Running {suite} evaluation..."):
        from forge.pipeline.eval_pipeline import format_eval_report, run_eval

        report = run_eval(model_path, suite=suite, limit=limit)

    console.print(format_eval_report(report))

    if save:
        import json
        from dataclasses import asdict

        Path(save).write_text(json.dumps(asdict(report), indent=2))
        console.print(f"\n[green]Saved to {save}")


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--tokens", type=int, default=200, help="Tokens to generate")
@click.option("--prompt", default="Explain the concept of recursion in computer science with examples.")
def profile(model_path: str, tokens: int, prompt: str):
    """Profile token-level latency distribution.

    MODEL_PATH: Path to optimized model
    """
    from rich.console import Console

    console = Console()

    with console.status("[bold blue]Profiling latency..."):
        from forge.analysis.latency_profiler import format_latency_report, profile_generation

        result = profile_generation(model_path, prompt=prompt, max_tokens=tokens)

    console.print(format_latency_report(result))


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--target-bits", type=float, default=4.0, help="Target average bits")
def sensitivity(model_path: str, target_bits: float):
    """Analyze per-layer quantization sensitivity for mixed-precision.

    MODEL_PATH: Path to model (FP16 or quantized)
    """
    from rich.console import Console

    console = Console()

    with console.status("[bold blue]Analyzing layer sensitivity..."):
        from forge.analysis.sensitivity import analyze_weight_sensitivity, format_sensitivity_report

        report = analyze_weight_sensitivity(model_path, target_avg_bits=target_bits)

    console.print(format_sensitivity_report(report))


if __name__ == "__main__":
    main()
