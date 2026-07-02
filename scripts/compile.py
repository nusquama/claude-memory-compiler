"""
Compile daily conversation logs into structured knowledge articles.

This is the "LLM compiler" - it reads daily logs (source code) and produces
organized knowledge articles (the executable).

Usage:
    uv run python compile.py                    # compile new/changed logs only
    uv run python compile.py --all              # force recompile everything
    uv run python compile.py --file daily/2026-04-01.md  # compile a specific log
    uv run python compile.py --dry-run          # show what would be compiled
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from config import (
    AGENTS_FILE,
    COMPILE_MODEL,
    CONCEPTS_DIR,
    CONNECTIONS_DIR,
    DAILY_DIR,
    KNOWLEDGE_DIR,
    LOG_FILE,
    PROJECT_DIR,
    now_iso,
)
from utils import (
    file_hash,
    list_raw_files,
    list_wiki_articles,
    load_state,
    read_wiki_index,
    save_state,
)

# ── Paths for the LLM to use ──────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

COMPILE_ATTEMPT_TIMEOUT = max(60, int(os.environ.get("CMC_COMPILE_ATTEMPT_TIMEOUT", "600")))
INLINE_ARTICLES = os.environ.get("CMC_COMPILE_INLINE_ARTICLES", "0") == "1"
ARTICLE_PREVIEW_CHARS = max(500, int(os.environ.get("CMC_COMPILE_ARTICLE_PREVIEW_CHARS", "2500")))


def _article_preview(content: str) -> str:
    """Small prompt preview; the compiler can Read full files if needed."""
    if len(content) <= ARTICLE_PREVIEW_CHARS:
        return content
    return content[:ARTICLE_PREVIEW_CHARS].rstrip() + (
        f"\n\n...[truncated {len(content) - ARTICLE_PREVIEW_CHARS} chars; "
        "use Read on this article before merging claims]..."
    )


def _wiki_link_for_article(article_path: Path) -> str:
    return article_path.relative_to(KNOWLEDGE_DIR).with_suffix("").as_posix()


def _articles_referencing_source(log_path: Path) -> list[str]:
    source = f"daily/{log_path.name}"
    links: list[str] = []
    for article_path in list_wiki_articles():
        try:
            content = article_path.read_text(encoding="utf-8")
        except OSError:
            continue
        if source in content:
            links.append(f"[[{_wiki_link_for_article(article_path)}]]")
    return links


def _ensure_build_log_entry(log_path: Path, timestamp: str, repaired: bool) -> None:
    """Append a minimal build-log entry if the LLM did not write one."""
    marker = f"compile | {log_path.name}"
    try:
        existing = LOG_FILE.read_text(encoding="utf-8") if LOG_FILE.exists() else ""
    except OSError:
        existing = ""
    if marker in existing:
        return

    links = _articles_referencing_source(log_path)
    lines = [
        f"## [{timestamp}] compile | {log_path.name}",
        f"- Source: daily/{log_path.name}",
    ]
    if links:
        lines.append(f"- Articles touched: {', '.join(links)}")
    else:
        lines.append("- Articles touched: (none detected)")
    if repaired:
        lines.append(
            "- Note: entry auto-repaired by compile.py after the Agent SDK "
            "returned a result then raised a post-result error."
        )
    else:
        lines.append("- Note: entry auto-repaired by compile.py because no build-log entry was written.")

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        if existing.strip():
            f.write("\n")
        f.write("\n".join(lines) + "\n")


async def compile_daily_log(log_path: Path, state: dict) -> float | None:
    """Compile a single daily log into knowledge articles.

    Returns the API cost of the compilation.
    """
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    log_content = log_path.read_text(encoding="utf-8")
    schema = AGENTS_FILE.read_text(encoding="utf-8")
    wiki_index = read_wiki_index()

    # Keep the prompt bounded. The index is the retrieval surface; full article
    # text can be loaded on demand through Read/Grep/Glob.
    existing_articles_context = ""
    existing = {}
    for article_path in list_wiki_articles():
        rel = article_path.relative_to(KNOWLEDGE_DIR)
        existing[str(rel)] = article_path.read_text(encoding="utf-8")

    if existing:
        parts = []
        for rel_path, content in existing.items():
            rendered = content if INLINE_ARTICLES else _article_preview(content)
            parts.append(f"### {rel_path}\n```markdown\n{rendered}\n```")
        existing_articles_context = "\n\n".join(parts)

    timestamp = now_iso()

    prompt = f"""Tu es un knowledge compiler. Tu lis un daily log et tu maintiens un wiki
structuré. Tu opères sur le daily log uniquement — tu n'as pas accès aux
sources originales (transcripts, fichiers .md du repo). Donc chaque
imprécision introduite ici reste dans le KB pour de bon.

## Schéma des articles (AGENTS.md)

{schema}

## Index courant du wiki

{wiki_index}

## Articles existants

{existing_articles_context if existing_articles_context else "(Aucun article pour l'instant)"}

Important: les articles existants ci-dessus peuvent être des aperçus tronqués.
Avant de merger, dédupliquer, confirmer une redondance, ou signaler une
contradiction, lis l'article complet sur disque avec `Read`. L'index sert à
choisir les candidats; l'aperçu ne suffit pas pour une décision finale.

## Daily log à compiler

**Fichier:** {log_path.name}

{log_content}

## Structure des daily logs en entrée

Les daily logs sont produits en amont par `flush.py` (sessions Claude Code)
et `scan_md.py` (.md du repo). Chaque entrée a typiquement les sections:

- `**Contexte:**` — sujet + stakes
- `**Déroulé:**` — liste numérotée chronologique des événements de la session
- `**Décisions prises:**` — décisions macro et micro avec rationale
- `**Chemins abandonnés:**` — alternatives explorées et rejetées
- `**Découvertes / gotchas / observations:**` — gotchas, valeurs vues, surprises
- `**Entités mentionnées:**` — IDs, paths, services, custom fields verbatim
- `**Citations notables:**` — quotes verbatim utilisateur/tiers
- `**Artefacts produits:**` — fichiers créés/modifiés avec statut
- `**Action items:**` — TODOs explicites
- `**Questions ouvertes:**` — non résolus à fin de session

Le `flush` est volontairement EXHAUSTIF (atomes redondants ou similaires).
**Ton job est de curer.** Tu vois donc plus de signal que tu n'en gardes.

Comment incorporer les nouvelles sections dans les concept articles:
- `Déroulé` → utile pour identifier l'ordre causal et les pivots, pas
  copié tel quel dans les articles.
- `Citations notables` → à intégrer dans les articles seulement si la
  formulation porte du signal (sinon paraphraser).
- `Artefacts produits` → si un fichier créé devient lui-même un concept
  référençable (spec, ADR), créer un article concept pour ce fichier
  avec son chemin dans `sources:`.
- `Questions ouvertes` → noter dans une section `## Questions ouvertes`
  d'un article existant si pertinent, ou créer un article concept dédié
  si la question structure plusieurs sujets.

## Politique de compilation

### 1. Filtrage — pas de quota

Extrais autant de concepts qu'il y a de signaux distincts dans le log.
- Log mono-thématique → 1 article suffit.
- Log touffu → autant d'articles que de concepts vraiment indépendants.
- Pas de fragmentation artificielle ("X" et "X-config" en deux articles).
- Pas d'agrégation forcée (deux concepts sans rapport dans un même article).

### 2. N'invente RIEN

Si une info n'est pas dans le log, ne la déduis pas. Pas de reconstruction
"vraisemblable" du contexte manquant. En cas de doute, omets. Une omission
est récupérable au prochain cycle ; une hallucination s'auto-renforce.

### 3. Anti-padding

Aucun minimum imposé sur la longueur des articles, le nombre de bullets,
le nombre de liens. Une section qui n'a qu'un bullet a un bullet. Une
section sans contenu n'apparaît pas.

### 4. Politique de merge (article existant couvre déjà le sujet)

a. Lis l'article existant en entier.
b. Compare les claims du daily log à ceux de l'article.
c. **Compatible** (info nouvelle, ne contredit rien) → intègre dans la
   section appropriée, ajoute le daily log dans `sources:`, met à jour
   `updated:`.
d. **Contradictoire** (le log dit non-X alors que l'article affirme X) →
   ne supprime PAS l'ancienne info. Ajoute une section `## Contradictions`
   avec:
   ```
   - {timestamp[:10]}: daily/{log_path.name} indique [non-X], contredit
     [X] (source originale: [source]). Résolution: à clarifier.
   ```
   La résolution est explicitement laissée à l'utilisateur. Une
   contradiction silencieusement écrasée perd un signal fort.
e. **Redondant** (même claim, déjà présent) → ne touche pas au texte,
   ajoute juste le daily log dans `sources:` pour tracer la confirmation.

### 5. Politique de wikilinks

- Un `[[wikilink]]` est ajouté uniquement si l'article cible existe ET est
  réellement pertinent.
- Pas de quota minimum. Un article isolé peut n'avoir aucun lien sortant.
- Pas de lien forcé vers un concept faiblement lié pour atteindre un nombre.
- Format: `[[concepts/slug]]` ou `[[connections/slug]]`, pas d'extension.

### 6. Provenance — préserve les marqueurs du daily log

Le daily log distingue `[Établi]`, `[Décidé]`, `[Hypothèse]`, `[Découvert]`.
Préserve cette distinction dans les articles:
- `[Établi]` → "Comportement vérifié:" ou simple affirmation factuelle.
- `[Décidé]` → "Décidé en {{date}} de…" avec rationale.
- `[Hypothèse]` → "Hypothèse non vérifiée:" — NE doit pas devenir un fait.
  Si une session ultérieure confirme, met à jour vers `[Établi]`.
- `[Découvert]` → "Comportement observé:" avec contexte du repro.

### 7. Connections — seulement si non-évidentes

Crée un article dans `connections/` uniquement quand le log révèle une
relation entre 2+ concepts qui ne saute pas aux yeux. Pas de connection
"X et Y sont tous les deux des outils Python". Une connection vaut le
coup si elle change la façon dont on raisonne sur l'un des deux concepts.

### 8. Format des articles

Suis le schéma de AGENTS.md, mais:
- Le `# Header` explique le sujet, pas le titre ("Pattern X pour Y", pas
  "X est un pattern qui…").
- Toutes les sections sauf `Sources` et frontmatter sont **optionnelles**.
- Si pas de Key Points distincts au-delà des Details, n'inclus pas la
  section.
- `Related Concepts` apparaît seulement si liens réels existent.
- `Sources` est obligatoire et chaque entrée pointe vers un daily log
  spécifique avec une note sur ce qu'il a contribué.

### 9. Langue

Écris dans la langue dominante du daily log. Cohérence: pas de mélange de
langues dans un même article. Termes techniques (noms de libs, erreurs,
commandes) restent en VO.

## Sorties à écrire

- **Concept articles** dans: {CONCEPTS_DIR}
- **Connection articles** dans: {CONNECTIONS_DIR}
- **Index** à jour: {KNOWLEDGE_DIR / 'index.md'}
- **Build log** à appender: {KNOWLEDGE_DIR / 'log.md'}

Format de l'entrée index:
`| [[path/slug]] | One-line summary | source-file | {timestamp[:10]} |`

Format de l'entrée build log:
```
## [{timestamp}] compile | {log_path.name}
- Source: daily/{log_path.name}
- Articles created: [[concepts/x]] (si aucun, omet la ligne)
- Articles updated: [[concepts/y]] (si aucun, omet la ligne)
- Contradictions flagged: [[concepts/z]] (si aucune, omet la ligne)
```
"""

    cost = 0.0
    had_result = False
    repaired_result_error = False

    try:
        async def _run_query() -> None:
            nonlocal cost, had_result
            async for message in query(
                prompt=prompt,
                options=ClaudeAgentOptions(
                    cwd=str(ROOT_DIR),
                    model=COMPILE_MODEL,
                    system_prompt={"type": "preset", "preset": "claude_code"},
                    allowed_tools=["Read", "Write", "Edit", "Glob", "Grep"],
                    permission_mode="acceptEdits",
                    max_turns=30,
                ),
            ):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            pass  # compilation output - LLM writes files directly
                elif isinstance(message, ResultMessage):
                    cost = message.total_cost_usd or 0.0
                    had_result = True
                    print(f"  Cost: ${cost:.4f}")

        await asyncio.wait_for(_run_query(), timeout=COMPILE_ATTEMPT_TIMEOUT)
    except asyncio.TimeoutError:
        print(f"  Error: compile timed out after {COMPILE_ATTEMPT_TIMEOUT}s")
        return None
    except Exception as e:
        if not had_result:
            print(f"  Error: {e}")
            return None
        repaired_result_error = True
        print(f"  Warning: Agent SDK returned a result then raised: {e}")

    # Update state
    _ensure_build_log_entry(log_path, timestamp, repaired_result_error)
    rel_path = log_path.name
    state.setdefault("ingested", {})[rel_path] = {
        "hash": file_hash(log_path),
        "compiled_at": now_iso(),
        "cost_usd": cost,
    }
    state["total_cost"] = state.get("total_cost", 0.0) + cost
    save_state(state)

    return cost


def main():
    if PROJECT_DIR is None:
        print("error: no project detected. Run from inside a git repo.", file=sys.stderr)
        sys.exit(1)
    parser = argparse.ArgumentParser(description="Compile daily logs into knowledge articles")
    parser.add_argument("--all", action="store_true", help="Force recompile all logs")
    parser.add_argument("--file", type=str, help="Compile a specific daily log file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be compiled")
    parser.add_argument(
        "--repair-state",
        action="store_true",
        help="No LLM call: ensure build-log entry and mark selected logs compiled",
    )
    args = parser.parse_args()

    state = load_state()

    # Determine which files to compile
    if args.file:
        target = Path(args.file)
        if not target.is_absolute():
            target = DAILY_DIR / target.name
        if not target.exists():
            # Try resolving relative to project root
            target = ROOT_DIR / args.file
        if not target.exists():
            print(f"Error: {args.file} not found")
            sys.exit(1)
        to_compile = [target]
    else:
        all_logs = list_raw_files()
        if args.all:
            to_compile = all_logs
        else:
            to_compile = []
            for log_path in all_logs:
                rel = log_path.name
                prev = state.get("ingested", {}).get(rel, {})
                if not prev or prev.get("hash") != file_hash(log_path):
                    to_compile.append(log_path)

    if not to_compile:
        print("Nothing to compile - all daily logs are up to date.")
        return

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Files to compile ({len(to_compile)}):")
    for f in to_compile:
        print(f"  - {f.name}")

    if args.dry_run:
        return

    if args.repair_state:
        print("[REPAIR] No LLM calls will be made.")
        for log_path in to_compile:
            timestamp = now_iso()
            _ensure_build_log_entry(log_path, timestamp, repaired=True)
            rel_path = log_path.name
            state.setdefault("ingested", {})[rel_path] = {
                "hash": file_hash(log_path),
                "compiled_at": timestamp,
                "cost_usd": 0.0,
                "repaired": True,
            }
            print(f"  Repaired: {rel_path}")
        save_state(state)
        return

    # Compile each file sequentially
    total_cost = 0.0
    for i, log_path in enumerate(to_compile, 1):
        print(f"\n[{i}/{len(to_compile)}] Compiling {log_path.name}...")
        cost = asyncio.run(compile_daily_log(log_path, state))
        if cost is None:
            print("  Failed; state not marked ingested.")
            continue
        total_cost += cost
        print("  Done.")

    articles = list_wiki_articles()
    print(f"\nCompilation complete. Total cost: ${total_cost:.2f}")
    print(f"Knowledge base: {len(articles)} articles")


if __name__ == "__main__":
    main()
