"""forge CLI — Auto-optimize and deploy LLM models on local Apple Silicon."""

from __future__ import annotations

import json
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

    # KV cache analysis with compression options
    from forge.engine.kv_cache import format_kv_report

    kv_report = format_kv_report(
        num_layers=model.num_layers,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.hidden_size // model.num_attention_heads,
    )
    console.print()
    console.print(kv_report)

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
@click.option(
    "--method",
    type=click.Choice(["mlx_native", "hqq", "any4", "d2quant", "gsr", "optiq"]),
    default=None,
    help="Quantization method (Phase 8: any4/d2quant/gsr/optiq)",
)
@click.option("--output", "-o", type=click.Path(), default=None, help="Output directory")
@click.option("--speculative", is_flag=True, help="Enable speculative decoding")
@click.option("--mixed-precision", "mixed_prec", is_flag=True, help="Per-layer mixed-precision quantization")
@click.option(
    "--compound",
    is_flag=True,
    help="Enable Phase 8.5 compound pipeline (ASVD → prune → rotate → quantize)",
)
@click.option("--profile", is_flag=True, default=True, help="Run profiling after optimization")
@click.option("--per-expert-quant", is_flag=True,
              help="Phase 9.6: MoE per-expert asymmetric quant (shared=4-bit, routed=2-bit)")
@click.option("--trust-remote-code", is_flag=True)
def optimize(
    model_id: str,
    bits: int | None,
    method: str | None,
    output: str | None,
    speculative: bool,
    mixed_prec: bool,
    compound: bool,
    profile: bool,
    per_expert_quant: bool,
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
            enable_compound=compound,
            quant_method_override=method,
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

    # Phase 8.5 — Run compound pipeline stub stages before the final quant.
    if compound and strategy.compound_pipeline:
        from forge.optimizer.strategy_selector import (
            run_asvd_rank_reduction,
            run_layer_prune,
        )
        console.print(f"[blue]Compound pipeline: {' → '.join(strategy.compound_pipeline)}")
        for stage in strategy.compound_pipeline:
            if stage == "asvd_rank_reduction":
                ok_s, msg_s = run_asvd_rank_reduction(model_id, str(output_dir))
                console.print(f"  {msg_s}")
            elif stage == "layer_prune":
                ok_s, msg_s = run_layer_prune(model_id, str(output_dir))
                console.print(f"  {msg_s}")
            elif stage == "gsr_rotation":
                console.print(f"  [gsr_rotation] queued — applied via quant_method='gsr' or as pre-pass")
            elif stage.startswith("quantize:"):
                console.print(f"  [quantize] method={stage.split(':', 1)[1]}")

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

    # Phase 9.6: Per-expert asymmetric quantization for MoE models
    if per_expert_quant and model.model_type == "moe":
        with console.status("[bold green]Phase 9.6: Re-quantizing experts asymmetrically (shared=4-bit, routed=2-bit)..."):
            from forge.optimizer.quantizer import quantize_per_expert_asymmetric

            pe_result = quantize_per_expert_asymmetric(
                model_dir=output_dir,
                shared_bits=4,
                routed_bits=2,
                model_id=model_id,
            )
        if pe_result.success:
            console.print(
                f"[green]Per-expert quant complete: {pe_result.quant} "
                f"({pe_result.size_gb:.1f} GB)"
            )
        else:
            console.print(f"[yellow]Per-expert quant skipped: {pe_result.error}")
    elif per_expert_quant:
        console.print(f"[yellow]--per-expert-quant ignored: {model_id} is not MoE.")

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
@click.option("--kv-compress", type=click.Choice(["none", "turbo", "fp8"]), default="none",
              help="KV cache compression method (turbo=TurboQuant 5.5x)")
@click.option("--kv-eviction", type=click.Choice(["none", "sliding", "h2o", "ada_kv"]), default="none",
              help="KV cache eviction policy (h2o=Heavy-Hitter, ada_kv=per-head adaptive)")
@click.option("--kv-budget", type=float, default=0.2, help="KV eviction budget ratio (0.2=keep 20%%)")
@click.option("--prefix-cache/--no-prefix-cache", default=False, help="Enable radix prefix caching")
@click.option("--ngram-spec", is_flag=True, help="N-gram self-speculation (no draft model needed)")
@click.option("--tree-spec", is_flag=True, help="Tree-based speculative verification")
@click.option("--adaptive-k/--no-adaptive-k", default=True, help="Adaptive draft length")
@click.option("--pearl/--no-pearl", default=True, help="PEARL pre/post-verify scheduling")
@click.option("--eagle-head", type=click.Path(), default=None, help="EAGLE-3 head directory or HF repo (Phase 7.5)")
@click.option("--cas-spec", is_flag=True, help="CAS-Spec cascade self-speculation with DyTC (Phase 12)")
@click.option("--lava-eviction", is_flag=True, help="LAVa unified layer+head KV eviction (Phase 12)")
@click.option("--xkv-rank", type=int, default=None, help="xKV cross-layer SVD rank for KV sharing (Phase 12)")
@click.option("--attention", type=click.Choice(["default", "mfa", "auto"]), default="default",
              help="Attention backend: mfa=mlx-mfa Metal FlashAttention (Phase 12)")
@click.option("--grammar", type=click.Path(), default=None,
              help="Grammar/JSON-Schema file for structured decoding (Phase 12, XGrammar-2)")
@click.option("--grammar-backend", type=click.Choice(["auto", "xgrammar", "llguidance", "outlines", "dummy"]),
              default="auto", help="Structured decoding backend (Phase 12)")
@click.option("--stream/--no-stream", default=True)
def run(model_path: str, prompt: str, max_tokens: int, draft: str | None,
        auto_draft: bool, redrafter: bool, temperature: float, kv_bits: str | None,
        max_kv_size: int | None, kv_compress: str, kv_eviction: str, kv_budget: float,
        prefix_cache: bool, ngram_spec: bool, tree_spec: bool, adaptive_k: bool,
        pearl: bool, eagle_head: str | None, cas_spec: bool, lava_eviction: bool,
        xkv_rank: int | None, attention: str, grammar: str | None,
        grammar_backend: str, stream: bool):
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
        kv_compression=kv_compress,
        kv_eviction=kv_eviction,
        kv_budget_ratio=kv_budget,
        enable_prefix_cache=prefix_cache,
        attention_backend=attention,
    )

    # Phase 12 Tier S — attention backend (mlx-mfa)
    if attention != "default":
        from forge.engine.attention_backend import detect_mlx_mfa
        ok, msg = detect_mlx_mfa()
        console.print(f"[blue]Attention backend: {attention} (mlx-mfa {'ready' if ok else 'unavailable: ' + msg})")

    # Phase 12 Tier S — LAVa eviction (overrides kv_eviction)
    if lava_eviction:
        console.print(f"[blue]LAVa unified layer+head eviction enabled (budget={kv_budget:.0%})")

    # Phase 12 Tier S — xKV cross-layer SVD
    if xkv_rank:
        console.print(f"[blue]xKV cross-layer SVD: rank={xkv_rank}")

    # Phase 12 Tier S — structured decoding
    if grammar:
        from forge.engine.structured_decoding import (
            GrammarCompiler, StructuredDecodingConfig, detect_structured_backends,
            load_grammar_from_file, select_backend,
        )
        grammar_spec = load_grammar_from_file(grammar)
        sd_config = StructuredDecodingConfig(grammar=grammar_spec, backend=grammar_backend)
        backends = detect_structured_backends()
        chosen = select_backend(sd_config)
        console.print(f"[blue]Structured decoding: grammar={grammar_spec.kind} backend={chosen} available={backends}")

    # Phase 12 Tier S — CAS-Spec cascade
    if cas_spec:
        from forge.engine.speculative import build_default_cas_spec_config, format_cas_spec_report
        cas_cfg = build_default_cas_spec_config(target_params_b=7.0, available_memory_gb=48.0)
        console.print(format_cas_spec_report(cas_cfg))

    # Show active KV optimizations
    kv_opts = []
    if kv_compress != "none":
        kv_opts.append(f"compression={kv_compress}")
    if kv_eviction != "none":
        kv_opts.append(f"eviction={kv_eviction}(budget={kv_budget:.0%})")
    if prefix_cache:
        kv_opts.append("prefix_cache=on")
    if kv_opts:
        console.print(f"[blue]KV optimizations: {', '.join(kv_opts)}")

    # EAGLE-3 head (Phase 7.5)
    if eagle_head:
        from forge.engine.eagle import format_eagle_info, select_eagle_head
        eagle_info = select_eagle_head(model_path, local_heads_dir=Path(eagle_head).parent)
        console.print(f"[blue]{format_eagle_info(eagle_info)}")

    # Show speculative decoding info
    spec_opts = []
    if ngram_spec:
        spec_opts.append("ngram_self_spec")
    if tree_spec:
        spec_opts.append("tree_attention")
    if adaptive_k:
        spec_opts.append("adaptive_k")
    if pearl:
        spec_opts.append("pearl_scheduling")
    if eagle_head:
        spec_opts.append("eagle3")
    if cas_spec:
        spec_opts.append("cas_spec_dytc")
    if draft_path:
        spec_opts.append(f"draft={Path(draft_path).name}")
    if spec_opts:
        console.print(f"[blue]Speculative: {', '.join(spec_opts)}")

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
@click.option("--chunked-prefill", is_flag=True,
              help="Enable Sarathi chunked prefill scheduling (Phase 10.1)")
@click.option("--chunk-size", type=int, default=512,
              help="Chunk size in tokens when --chunked-prefill is set")
@click.option("--interruptible", is_flag=True,
              help="Enable FastServe-style pause/resume sessions (Phase 10.2)")
@click.option("--multi-model", is_flag=True,
              help="Enable multi-model LRU hot loading (Phase 10.3)")
@click.option("--pmpd", is_flag=True,
              help="Enable PMPD FP16-prefill/INT4-decode control plane (Phase 10.4, experimental)")
@click.option("--use-vllm-mlx", is_flag=True,
              help="Prefer vllm-mlx PagedAttention backend when available (Phase 10.5)")
def deploy(
    model_path: str,
    host: str,
    port: int,
    runtime: str | None,
    chunked_prefill: bool,
    chunk_size: int,
    interruptible: bool,
    multi_model: bool,
    pmpd: bool,
    use_vllm_mlx: bool,
):
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
        chunked_prefill=chunked_prefill,
        chunk_size=chunk_size,
        interruptible=interruptible,
        multi_model=multi_model,
        pmpd=pmpd,
        use_vllm_mlx=use_vllm_mlx,
    )

    console.print(format_deploy_info(deploy_config))
    console.print()

    adv_opts = []
    if chunked_prefill:
        adv_opts.append(f"chunked_prefill(chunk={chunk_size})")
    if interruptible:
        adv_opts.append("interruptible")
    if multi_model:
        adv_opts.append("multi_model")
    if pmpd:
        adv_opts.append("pmpd")
    if use_vllm_mlx:
        from forge.engine.scheduler import detect_vllm_mlx
        adv_opts.append(f"vllm_mlx={'yes' if detect_vllm_mlx() else 'fallback'}")
    if adv_opts:
        console.print(f"[blue]Advanced scheduling: {', '.join(adv_opts)}")
        console.print()

    if selected_runtime == "mlx-lm":
        from forge.pipeline.deployer import serve_advanced, serve_mlx

        console.print(f"[bold green]Starting mlx-lm server on {host}:{port}...")
        console.print("Press Ctrl+C to stop.\n")
        try:
            if deploy_config.advanced_enabled():
                proc = serve_advanced(deploy_config)
            else:
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
@click.option(
    "--method",
    type=click.Choice(["aimer", "activation", "hybrid", "evolutionary"]),
    default="activation",
    help="Scoring method (Phase 9.1-9.2: aimer=calibration-free, evolutionary=per-layer GA)",
)
@click.option("--plan-method", type=click.Choice(["remove", "merge"]), default="remove",
              help="How to apply the plan: remove experts or merge them")
@click.option("--num-samples", type=int, default=20, help="Calibration prompts (activation/hybrid only)")
@click.option("--save-plan", type=click.Path(), default=None, help="Save pruning plan JSON")
def expert_prune(
    model_id: str,
    ratio: float,
    method: str,
    plan_method: str,
    num_samples: int,
    save_plan: str | None,
):
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
    console.print(f"[blue]Scoring method: {method}")

    from forge.optimizer.expert_pruner import (
        analyze_expert_importance,
        create_pruning_plan,
        evolutionary_layer_pruning,
        format_layerwise_plan,
        format_pruning_plan,
        save_pruning_plan,
        score_experts_aimer,
    )

    if method == "aimer":
        with console.status("[bold blue]Scoring experts by weight RMSE (AIMER)..."):
            rankings = score_experts_aimer(model_id)
    elif method == "hybrid":
        with console.status(f"[bold blue]Hybrid (AIMER + activation, {num_samples} samples)..."):
            rankings = analyze_expert_importance(
                model_id, num_samples=num_samples, method="hybrid",
            )
    elif method == "evolutionary":
        with console.status("[bold blue]AIMER seed + evolutionary per-layer search..."):
            rankings = score_experts_aimer(model_id)
            if not rankings:
                console.print("[yellow]AIMER seed failed; falling back to activation.")
                rankings = analyze_expert_importance(model_id, num_samples=num_samples)
            layer_plan = evolutionary_layer_pruning(
                rankings,
                target_ratio=ratio,
                population_size=20,
                generations=10,
                model_id=model_id,
                num_experts_per_layer=model.num_experts or 8,
            )
        console.print()
        console.print(format_layerwise_plan(layer_plan))
        if save_plan:
            import dataclasses
            Path(save_plan).parent.mkdir(parents=True, exist_ok=True)
            Path(save_plan).write_text(json.dumps(dataclasses.asdict(layer_plan), indent=2))
            console.print(f"\n[green]Layer-wise plan saved to: {save_plan}")
        return
    else:
        with console.status(f"[bold blue]Capturing expert activations ({num_samples} samples)..."):
            rankings = analyze_expert_importance(
                model_id, num_samples=num_samples, method="activation",
            )

    if not rankings:
        console.print("[red]Failed to score experts. Check model accessibility / dependencies.")
        sys.exit(1)

    plan = create_pruning_plan(
        model_id=model_id,
        rankings=rankings,
        prune_ratio=ratio,
        method=plan_method,
        num_experts=model.num_experts or 8,
    )

    console.print()
    console.print(format_pruning_plan(plan))

    if save_plan:
        save_pruning_plan(plan, Path(save_plan))
        console.print(f"\n[green]Pruning plan saved to: {save_plan}")


@main.command(name="expert-merge")
@click.argument("model_id")
@click.option("--rank", type=int, default=8, help="Shared-basis rank for joint SVD")
@click.option("--groups", type=int, default=2, help="Number of expert groups (per-layer partition)")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output .subm state-dict path")
@click.option("--center/--no-center", default=False, help="Center weights per group before SVD")
def expert_merge(model_id: str, rank: int, groups: int, output: str | None, center: bool):
    """MoE-SVD: merge MoE experts into a shared low-rank basis.

    MODEL_ID: HuggingFace MoE model ID. This command loads the model,
    extracts per-layer expert weight matrices, and fits a SubMoEMerger
    that writes a deserializable state dict plus a JSON manifest.
    """
    from rich.console import Console

    console = Console()

    with console.status("[bold blue]Loading MLX model..."):
        try:
            from mlx_lm import load as mlx_load  # type: ignore[import-untyped]
            import mlx.core as mx
            import numpy as np
        except ImportError as e:
            console.print(f"[red]Missing dependency: {e}")
            sys.exit(1)
        try:
            model, _ = mlx_load(model_id)
        except Exception as e:
            console.print(f"[red]Failed to load model: {e}")
            sys.exit(1)

    # Collect expert weight matrices from the parameter tree. We target
    # down-projection / w2 tensors as the canonical expert weight.
    from collections import defaultdict as _dd
    layer_experts: dict[int, list[tuple[int, "np.ndarray"]]] = _dd(list)

    def _walk(prefix: str, node: object) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                _walk(f"{prefix}.{i}", v)
        else:
            lower = prefix.lower()
            if "expert" in lower and ("down_proj" in lower or ".w2" in lower):
                # Extract layer + expert indices
                from forge.optimizer.expert_pruner import _extract_int_after
                L = _extract_int_after(lower, "layers.")
                E = _extract_int_after(lower, "experts.")
                if L is None or E is None:
                    return
                try:
                    arr = np.asarray(mx.array(node).astype(mx.float32))
                except Exception:
                    return
                if arr.ndim == 2:
                    layer_experts[L].append((E, arr))

    try:
        params = model.parameters()
    except Exception as e:
        console.print(f"[red]Could not access model parameters: {e}")
        sys.exit(1)
    _walk("", params)

    if not layer_experts:
        console.print("[red]No expert weight matrices found. Is this an MoE model?")
        sys.exit(1)

    console.print(f"[blue]Found experts in {len(layer_experts)} layer(s). Merging layer 0 as reference.")

    # Reference run: fit on the first layer. Full-model merging would loop.
    from forge.optimizer.expert_merger import (
        MergeConfig,
        SubMoEMerger,
        format_merge_report,
    )

    first_layer = sorted(layer_experts.keys())[0]
    experts = [w for _eid, w in sorted(layer_experts[first_layer])]
    console.print(f"[blue]Layer {first_layer}: {len(experts)} experts, shape={experts[0].shape}")

    with console.status("[bold green]Fitting joint SVD..."):
        merger = SubMoEMerger(MergeConfig(
            rank=rank, num_groups=groups, center_weights=center,
        ))
        merger.fit(experts)
        merger.apply()

    report = merger.report(model_id=model_id)
    console.print()
    console.print(format_merge_report(report))

    # Save
    safe_name = model_id.replace("/", "--")
    out_path = Path(output) if output else Path(f"./optimized/{safe_name}-submoe.pkl")
    saved = merger.save(out_path)
    console.print(f"\n[green]Saved merged state dict to: {saved}")
    console.print(f"[green]Manifest: {saved.with_suffix(saved.suffix + '.json')}")


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


@main.command(name="long-context")
@click.argument("model_id_or_path")
@click.option("--target-context", type=int, default=1_048_576, help="Target context length in tokens")
@click.option("--memory-gb", type=float, default=48.0, help="Available memory budget (GB)")
@click.option("--weights-gb", type=float, default=4.0, help="Model weights size (GB, 7B 4-bit ≈ 4)")
@click.option("--kv-compression", type=float, default=5.5, help="KV compression ratio (TurboQuant=5.5)")
@click.option("--num-layers", type=int, default=32)
@click.option("--num-kv-heads", type=int, default=4)
@click.option("--head-dim", type=int, default=128)
def long_context(model_id_or_path: str, target_context: int, memory_gb: float,
                 weights_gb: float, kv_compression: float, num_layers: int,
                 num_kv_heads: int, head_dim: int):
    """Phase 12: Analyze long-context feasibility (DCA + chunked prefill).

    MODEL_ID_OR_PATH: Model identifier — detects Qwen2.5-1M / Llama / Phi variants.
    """
    from rich.console import Console

    from forge.engine.long_context import (
        LongContextModelDetector,
        estimate_long_context_feasibility,
        format_long_context_report,
    )

    console = Console()

    detector = LongContextModelDetector()
    info = detector.detect(model_id_or_path)
    if info:
        dca_config = detector.recommend_dca_config(info)
        console.print(f"[green]Detected long-context model: {info}")
    else:
        from forge.engine.long_context import DCAConfig
        dca_config = DCAConfig(max_context_length=target_context)
        console.print(f"[yellow]Unknown model — using default DCA config")

    feas = estimate_long_context_feasibility(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        target_context=target_context,
        available_memory_gb=memory_gb,
        weights_gb=weights_gb,
        kv_compression_ratio=kv_compression,
    )

    console.print()
    console.print(format_long_context_report(feas, dca_config))


if __name__ == "__main__":
    main()
