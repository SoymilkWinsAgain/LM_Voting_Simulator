"""Command line interface for the simulator."""

from __future__ import annotations

from pathlib import Path

import typer

from .anes import build_anes, build_memory_cards
from .ces import build_ces as build_ces_pipeline
from .ces import build_ces_cells as build_ces_cells_pipeline
from .ces import build_ces_memory_cards as build_ces_memory_pipeline
from .ces_anes_persona import build_ces_anes_persona as build_ces_anes_persona_pipeline
from .ces_aggregate_benchmark import run_ces_aggregate_benchmark
from .ces_ablation_benchmark import run_ces_ablation_benchmark
from .ces_benchmark import run_ces_individual_benchmark
from .ces_leakage_benchmark import run_ces_leakage_benchmark
from .config import load_run_config
from .mit import build_mit_results as build_mit_results_pipeline
from .mit import write_mit_processed_artifacts
from .population import build_agents as build_agents_pipeline
from .simulation import evaluate_run, run_simulation

app = typer.Typer(no_args_is_help=True)


@app.command()
def validate_config(config: Path = typer.Option(..., "--config")) -> None:
    cfg = load_run_config(config)
    typer.echo(f"OK: {cfg.run_id}")


@app.command("build-anes")
def build_anes_command(
    config: Path = typer.Option(..., "--config"),
    profile_crosswalk: Path = typer.Option(..., "--profile-crosswalk"),
    question_crosswalk: Path = typer.Option(..., "--question-crosswalk"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    paths = build_anes(config, profile_crosswalk, question_crosswalk, out)
    for name, path in paths.items():
        typer.echo(f"{name}: {path}")


@app.command()
def build_anes_memory(
    respondents: Path = typer.Option(..., "--respondents"),
    answers: Path = typer.Option(..., "--answers"),
    fact_templates: Path = typer.Option(..., "--fact-templates"),
    policy: str = typer.Option("safe_survey_memory_v1", "--policy"),
    out: Path = typer.Option(..., "--out"),
    max_facts: int = typer.Option(24, "--max-facts"),
) -> None:
    paths = build_memory_cards(respondents, answers, fact_templates, policy, out, max_facts=max_facts)
    for name, path in paths.items():
        typer.echo(f"{name}: {path}")


@app.command()
def build_ces(
    config: Path = typer.Option(..., "--config"),
    profile_crosswalk: Path = typer.Option(..., "--profile-crosswalk"),
    question_crosswalk: Path = typer.Option(..., "--question-crosswalk"),
    target_crosswalk: Path = typer.Option(..., "--target-crosswalk"),
    context_crosswalk: Path = typer.Option(..., "--context-crosswalk"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    paths = build_ces_pipeline(
        config,
        profile_crosswalk,
        question_crosswalk,
        target_crosswalk,
        context_crosswalk,
        out,
    )
    for name, path in paths.items():
        typer.echo(f"{name}: {path}")


@app.command()
def build_ces_memory(
    respondents: Path = typer.Option(..., "--respondents"),
    answers: Path = typer.Option(..., "--answers"),
    fact_templates: Path = typer.Option(..., "--fact-templates"),
    policy: str = typer.Option("strict_pre_no_vote_v1", "--policy"),
    out: Path = typer.Option(..., "--out"),
    max_facts: int = typer.Option(24, "--max-facts"),
) -> None:
    paths = build_ces_memory_pipeline(respondents, answers, fact_templates, policy, out, max_facts=max_facts)
    for name, path in paths.items():
        typer.echo(f"{name}: {path}")


@app.command("build-ces-anes-persona")
def build_ces_anes_persona(
    ces_respondents: Path = typer.Option(..., "--ces-respondents"),
    ces_answers: Path = typer.Option(..., "--ces-answers"),
    ces_memory_facts: Path = typer.Option(..., "--ces-memory-facts"),
    ces_memory_cards: Path = typer.Option(..., "--ces-memory-cards"),
    anes_raw: Path = typer.Option(..., "--anes-raw"),
    anes_open_ends: Path = typer.Option(..., "--anes-open-ends"),
    config: Path = typer.Option(..., "--config"),
    out: Path = typer.Option(..., "--out"),
    limit_ces: int | None = typer.Option(None, "--limit-ces"),
    limit_anes: int | None = typer.Option(None, "--limit-anes"),
) -> None:
    paths = build_ces_anes_persona_pipeline(
        ces_respondents_path=ces_respondents,
        ces_answers_path=ces_answers,
        ces_memory_facts_path=ces_memory_facts,
        ces_memory_cards_path=ces_memory_cards,
        anes_raw_path=anes_raw,
        anes_open_ends_path=anes_open_ends,
        config_path=config,
        out_dir=out,
        limit_ces=limit_ces,
        limit_anes=limit_anes,
    )
    for name, path in paths.items():
        typer.echo(f"{name}: {path}")


@app.command()
def build_ces_cells(
    config: Path = typer.Option(..., "--config"),
    crosswalk: Path = typer.Option(..., "--crosswalk"),
    cell_schema: Path = typer.Option(..., "--cell-schema"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    paths = build_ces_cells_pipeline(config, crosswalk, cell_schema, out)
    for name, path in paths.items():
        typer.echo(f"{name}: {path}")


@app.command()
def build_mit_results(
    config: Path = typer.Option(..., "--config"),
    year: int = typer.Option(..., "--year"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    path = build_mit_results_pipeline(config, year, out)
    typer.echo(f"mit_results: {path}")


@app.command("build-mit-president")
def build_mit_president(config: Path = typer.Option(..., "--config")) -> None:
    paths = write_mit_processed_artifacts(config)
    for name, path in paths.items():
        typer.echo(f"{name}: {path}")


@app.command("build-agents")
def build_agents_command(
    run_config: Path = typer.Option(..., "--run-config"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    path = build_agents_pipeline(run_config, out)
    typer.echo(f"agents: {path}")


@app.command("run-simulation")
def run_simulation_command(run_config: Path = typer.Option(..., "--run-config")) -> None:
    outputs = run_simulation(run_config)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-ces-individual-benchmark")
def run_ces_individual_benchmark_command(config: Path = typer.Option(..., "--config")) -> None:
    outputs = run_ces_individual_benchmark(config)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-ces-aggregate-benchmark")
def run_ces_aggregate_benchmark_command(config: Path = typer.Option(..., "--config")) -> None:
    outputs = run_ces_aggregate_benchmark(config)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-ces-ablation-benchmark")
def run_ces_ablation_benchmark_command(config: Path = typer.Option(..., "--config")) -> None:
    outputs = run_ces_ablation_benchmark(config)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-ces-leakage-benchmark")
def run_ces_leakage_benchmark_command(config: Path = typer.Option(..., "--config")) -> None:
    outputs = run_ces_leakage_benchmark(config)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command()
def evaluate(run_id: str = typer.Option(..., "--run-id"), run_dir: Path = typer.Option(..., "--run-dir")) -> None:
    outputs = evaluate_run(run_id, run_dir)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


if __name__ == "__main__":
    app()
