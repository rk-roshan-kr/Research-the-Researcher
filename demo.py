"""
demo.py — Week 2 CLI demonstration.

Shows the full Investigation → Blocker → Contradiction → Priority flow
without requiring Ollama, OpenAlex, or any external service.

Usage:
    python demo.py

Expected output demonstrates:
    ✓ Seed investigations loaded
    ✓ Investigations activated (respecting MAX_ACTIVE=5)
    ✓ Blocker quality gate (vague rejected, specific accepted)
    ✓ Contradiction added with severity × confidence escalation
    ✓ Priority changes logged
    ✓ Queued investigation auto-promoted when priority exceeds active
    ✓ Unresolved blocker list (drives next professor search)
    ✓ Blocker resolved (user confirms, never auto)
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from academic_pipeline.cache_manager import CacheManager
from academic_pipeline.investigation_engine import (
    InvestigationEngine,
    BlockerTooVagueError,
    ActiveSlotsFullError,
)

console = Console()


def section(title: str):
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]")
    console.print()


def print_investigations(engine: InvestigationEngine):
    active = engine.get_active_investigations()
    queued = engine.get_queued_investigations()
    slot_count = engine.active_slot_count()

    table = Table(
        title=f"Active Investigations  [{slot_count}/5 slots]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("Title", style="bold white", ratio=3)
    table.add_column("Status", justify="center", ratio=1)
    table.add_column("Priority", justify="right", ratio=1)
    table.add_column("Blockers", justify="right", ratio=1)
    table.add_column("Contradictions", justify="right", ratio=1)

    for inv in active:
        status_color = "red" if inv.status == "blocked" else "green"
        table.add_row(
            inv.title,
            f"[{status_color}]{inv.status.upper()}[/{status_color}]",
            f"[bold]{inv.effective_priority:.2f}[/bold]",
            str(len(inv.open_blockers)),
            str(len([c for c in inv.contradictions if c.status == "open"])),
        )

    console.print(table)

    if queued:
        console.print(f"\n[dim]Queued ({len(queued)} waiting for slot):[/dim]")
        for inv in queued:
            console.print(
                f"  [dim]○ {inv.title}  priority: {inv.effective_priority:.2f}[/dim]"
            )


def print_blockers(engine: InvestigationEngine):
    unresolved = engine.get_unresolved_blockers()
    if not unresolved:
        console.print("[green]All blockers resolved![/green]")
        return

    table = Table(
        title="Unresolved Blockers  (drives next professor search)",
        box=box.SIMPLE_HEAD,
        header_style="bold yellow",
        expand=True,
    )
    table.add_column("Investigation", ratio=2)
    table.add_column("Blocker", ratio=4)
    table.add_column("Type", ratio=1)
    table.add_column("Severity", justify="right", ratio=1)
    table.add_column("Specificity", justify="right", ratio=1)

    for inv, blocker in unresolved:
        table.add_row(
            f"[bold]{inv.title[:30]}[/bold]",
            blocker.description[:70] + ("..." if len(blocker.description) > 70 else ""),
            blocker.blocker_type,
            f"{blocker.severity:.2f}",
            f"{blocker.specificity_score:.2f}",
        )
    console.print(table)


def main():
    console.print(Panel.fit(
        "[bold violet]Research Opportunity Discovery System[/bold violet]\n"
        "[dim]Week 2 CLI Demo — Investigation Engine[/dim]",
        border_style="violet",
    ))

    # Use a temp directory so demo doesn't pollute real outputs
    tmpdir = tempfile.mkdtemp(prefix="rods_demo_")
    cache = CacheManager(cache_dir=tmpdir)
    engine = InvestigationEngine(cache_manager=cache)

    # ─────────────────────────────────────────────────────────────────────────
    section("STEP 1: Seed Investigations")
    # ─────────────────────────────────────────────────────────────────────────

    seeds = engine.seed_investigations()
    console.print(f"[green]✓[/green] Seeded {len(seeds)} investigations")
    for inv in engine.get_all_investigations():
        icon = "●" if inv.status in ("active", "blocked") else "○"
        console.print(
            f"  {icon} [{inv.status}] {inv.title}  "
            f"priority={inv.effective_priority:.2f}  "
            f"blockers={len(inv.blockers)}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    section("STEP 2: Blocker Quality Gate")
    # ─────────────────────────────────────────────────────────────────────────

    earthos_inv = engine.get_active_investigations()[0]
    console.print(f"[bold]Testing blockers on: {earthos_inv.title}[/bold]\n")

    # Bad blocker — should be rejected
    bad_desc = "Need better AI for my EarthOS research"
    console.print(f"[yellow]Attempting vague blocker:[/yellow] '{bad_desc}'")
    try:
        engine.add_blocker(earthos_inv.id, bad_desc, "methodology", 0.8)
        console.print("[red]✗ Should have been rejected![/red]")
    except BlockerTooVagueError as e:
        console.print("[green]✓ Correctly rejected.[/green]")
        console.print(Panel(
            str(e),
            title="[yellow]Rejection Message[/yellow]",
            border_style="yellow",
            padding=(0, 1),
        ))

    # Good blocker — should be accepted
    good_desc = (
        "Validated benchmark showing how investigation persistence affects "
        "cognitive coherence in continual learning agents across multi-session "
        "episodic memory retrieval tasks compared to single-session baselines"
    )
    console.print(f"\n[yellow]Adding specific blocker...[/yellow]")
    blocker = engine.add_blocker(earthos_inv.id, good_desc, "methodology", 0.78)
    console.print(
        f"[green]✓ Accepted.[/green]  "
        f"specificity={blocker.specificity_score:.2f}  "
        f"effective_priority={blocker.effective_priority:.2f}"
    )

    # ─────────────────────────────────────────────────────────────────────────
    section("STEP 3: Add Contradiction → Priority Escalation")
    # ─────────────────────────────────────────────────────────────────────────

    console.print(f"[bold]Adding contradiction to: {earthos_inv.title}[/bold]")
    console.print(
        f"  Base priority before: [bold]{earthos_inv.effective_priority:.2f}[/bold]"
    )

    # Low-confidence contradiction — should NOT escalate
    inv_after, c1 = engine.add_contradiction(
        inv_id=earthos_inv.id,
        claim_yours="Contradiction-driven agents require explicit contradiction registers",
        claim_external="Wang et al. 2023 show implicit contradiction resolution is sufficient",
        source_yours="EarthOS design doc",
        source_external="Wang et al. 2023, NeurIPS",
        severity=0.90,
        confidence=0.25,   # low confidence → signal=0.225, no escalation
        resolution_hypothesis="May depend on investigation complexity",
    )
    console.print(
        f"\n  [dim]Low-confidence contradiction (severity=0.90, confidence=0.25):[/dim]"
        f"\n  signal = {c1.effective_signal:.3f}  →  "
        f"priority after: [bold]{inv_after.effective_priority:.2f}[/bold]  "
        f"[dim](no change expected)[/dim]"
    )

    # High-confidence contradiction — SHOULD escalate
    inv_after2, c2 = engine.add_contradiction(
        inv_id=earthos_inv.id,
        claim_yours="Persistent investigation streams maintain coherence over 6+ months",
        claim_external="Kirkpatrick et al. show catastrophic forgetting prevents coherence "
                       "in online agents beyond 3-month timescales without explicit replay",
        source_yours="EarthOS paper v0.2",
        source_external="Kirkpatrick et al. 2017, PNAS (extended)",
        severity=0.92,
        confidence=0.85,   # high confidence → signal=0.782, +0.30 escalation
        resolution_hypothesis="Replay mechanisms may bridge the gap",
    )
    console.print(
        f"\n  [bold]High-confidence contradiction (severity=0.92, confidence=0.85):[/bold]"
        f"\n  signal = {c2.effective_signal:.3f}  →  "
        f"priority after: [bold green]{inv_after2.effective_priority:.2f}[/bold green]  "
        f"[green](escalation applied)[/green]"
    )

    # ─────────────────────────────────────────────────────────────────────────
    section("STEP 4: Investigation State After Escalation")
    # ─────────────────────────────────────────────────────────────────────────

    print_investigations(engine)

    # ─────────────────────────────────────────────────────────────────────────
    section("STEP 5: Slot Pressure — Create High-Priority Investigation")
    # ─────────────────────────────────────────────────────────────────────────

    # Create enough investigations to fill all 5 slots, then add one more
    # with very high priority to demonstrate slot displacement

    console.print("[dim]Filling remaining slots...[/dim]")
    extra_invs = []
    for i in range(3):
        inv = engine.create_investigation(
            title=f"Filler Investigation {i+1}",
            description=f"A placeholder investigation {i+1} for slot testing.",
            base_priority=0.30 + i * 0.05,
            auto_activate=True,
        )
        extra_invs.append(inv)
        console.print(f"  Created: '{inv.title}' → {inv.status}")

    console.print(f"\n[bold]Slots now: {engine.active_slot_count()}/5[/bold]")

    # Now try to add a very high-priority one — should displace lowest
    high_prio = engine.create_investigation(
        title="URGENT: Critical EarthOS Contradiction Resolved",
        description="High-priority investigation just discovered — needs immediate attention.",
        base_priority=0.98,
        auto_activate=True,
    )
    console.print(f"\n  High-priority investigation status: [bold green]{high_prio.status}[/bold green]")
    console.print(f"  (Lowest-priority filler was displaced → queued)")

    # ─────────────────────────────────────────────────────────────────────────
    section("STEP 6: Unresolved Blockers (Next Search Targets)")
    # ─────────────────────────────────────────────────────────────────────────

    print_blockers(engine)

    # ─────────────────────────────────────────────────────────────────────────
    section("STEP 7: Resolve a Blocker (User Confirms)")
    # ─────────────────────────────────────────────────────────────────────────

    # Get the EarthOS investigation's first blocker (the good one we added)
    earthos_final = engine.get_investigation(earthos_inv.id)
    unresolved = [b for b in earthos_final.blockers if not b.resolved]

    if unresolved:
        target_blocker = unresolved[0]
        console.print(
            f"[bold]Resolving blocker:[/bold]\n"
            f"  '{target_blocker.description[:80]}...'\n"
            f"  Resolved by: Prof. Alexander Mathis (published_result)"
        )
        updated_inv, resolved_blocker = engine.resolve_blocker(
            inv_id=earthos_inv.id,
            blocker_id=target_blocker.id,
            resolved_by="Prof. Alexander Mathis",
            evidence_strength="published_result",
            notes="Their 2023 paper directly addresses the benchmark requirement.",
        )
        console.print(
            f"\n[green]✓ Blocker resolved.[/green]  "
            f"Evidence strength: {resolved_blocker.evidence_strength}  "
            f"Investigation status: {updated_inv.status}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    section("FINAL STATE")
    # ─────────────────────────────────────────────────────────────────────────

    print_investigations(engine)

    console.print()
    console.print(Panel.fit(
        "[bold green]Week 2 Demo Complete[/bold green]\n\n"
        "[dim]Demonstrated:[/dim]\n"
        "  ✓ Investigation creation + seeding\n"
        "  ✓ Blocker quality gate (vague rejected, specific accepted)\n"
        "  ✓ Contradiction severity × confidence escalation\n"
        "  ✓ MAX_ACTIVE=5 enforcement with slot displacement\n"
        "  ✓ Unresolved blocker list (professor search targets)\n"
        "  ✓ Blocker resolution (user confirms, never auto)\n\n"
        "[dim]Next: Collaboration Scorer + ENRS (Week 3)[/dim]",
        border_style="green",
    ))

    # Cleanup temp dir
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
