"""
Memory flush agent - extracts important knowledge from conversation context.

Spawned by session-end.py or pre-compact.py as a background process. Reads
pre-extracted conversation context from a .md file, uses the Claude Agent SDK
to decide what's worth saving, and appends the result to today's daily log.

Usage:
    uv run python flush.py <context_file.md> <session_id>
"""

from __future__ import annotations

# Recursion prevention: set this BEFORE any imports that might trigger Claude
import os
os.environ["CLAUDE_INVOKED_BY"] = "memory_flush"

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import (
    DAILY_DIR,
    FLUSH_CHUNK_SIZE,
    FLUSH_LOG as LOG_FILE,
    FLUSH_MODEL,
    FLUSH_SINGLE_PASS_THRESHOLD,
    FLUSH_STATE_FILE as STATE_FILE,
    PROJECT_DIR,
    SCRIPTS_DIR,
    STATE_DIR,
    TOOL_DIR as ROOT,
)


def _bootstrap_for_main() -> None:
    """Side-effects gated behind main() so the module is import-safe.

    scan_md.py reuses helpers from this module (append_to_daily_log, lock,
    retry constants); importing it must not exit, mkdir, or hijack logging.
    """
    if PROJECT_DIR is None:
        sys.exit(0)
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    # File-based logging is our only observability channel for the
    # detached background process (parent sends stdout/stderr to DEVNULL).
    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_flush_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_flush_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state), encoding="utf-8")


def append_to_daily_log(content: str, section: str = "Session") -> None:
    """Append content to today's daily log (no session-id upsert).

    Used by scan_md.py and the error-marker helper. For session flushes
    where periodic checkpoints can produce multiple extractions of the
    same session, use upsert_session_entry() instead.
    """
    today = datetime.now(timezone.utc).astimezone()
    log_path = DAILY_DIR / f"{today.strftime('%Y-%m-%d')}.md"

    if not log_path.exists():
        DAILY_DIR.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"# Daily Log: {today.strftime('%Y-%m-%d')}\n\n## Sessions\n\n## Memory Maintenance\n\n",
            encoding="utf-8",
        )

    time_str = today.strftime("%H:%M")
    entry = f"### {section} ({time_str})\n\n{content}\n\n"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)


def upsert_session_entry(content: str, session_id: str, section: str = "Session") -> None:
    """Insert or replace a session block in today's daily log.

    Each session's entry is wrapped in HTML comment markers tagged with
    session_id. If a block with the same session_id already exists in
    today's log, replace it in place; otherwise append a new one.

    This makes periodic flush checkpoints idempotent: running flush.py
    five times during the same session produces a single (final) entry,
    not five duplicates.
    """
    today = datetime.now(timezone.utc).astimezone()
    log_path = DAILY_DIR / f"{today.strftime('%Y-%m-%d')}.md"

    if not log_path.exists():
        DAILY_DIR.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"# Daily Log: {today.strftime('%Y-%m-%d')}\n\n## Sessions\n\n## Memory Maintenance\n\n",
            encoding="utf-8",
        )

    time_str = today.strftime("%H:%M")
    open_tag = f"<!-- cmc-session: {session_id} -->"
    close_tag = "<!-- /cmc-session -->"
    block = f"{open_tag}\n### {section} ({time_str})\n\n{content}\n{close_tag}\n\n"

    existing = log_path.read_text(encoding="utf-8")

    import re
    pattern = re.compile(
        re.escape(open_tag) + r".*?" + re.escape(close_tag) + r"\n*",
        re.DOTALL,
    )

    if pattern.search(existing):
        updated = pattern.sub(block, existing, count=1)
        log_path.write_text(updated, encoding="utf-8")
    else:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(block)


def detect_failure_cause(stderr_lines: list[str], exc_message: str) -> str:
    """Best-effort one-line label for why the bundled CLI failed."""
    text = (" ".join(stderr_lines) + " " + exc_message).lower()
    if "401" in text or "invalid x-api-key" in text or "authentication_error" in text:
        return "authentication failed (likely v2.1.92 intermittent 401 bug)"
    if "429" in text or "rate limit" in text or "rate_limit" in text:
        return "rate limited"
    if "shell is already running" in text:
        return "concurrent CLI invocation"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "not a valid model" in text or "invalid model" in text:
        return "invalid model name"
    return "see flush.log for details"


def append_error_marker_to_daily(session_id: str, cause: str) -> None:
    """Write a brief, lisible error marker to today's daily log.

    Replaces the previous behaviour of dumping the full FLUSH_ERROR string
    (with traceback) into the daily log. The detail still lives in
    flush.log; the daily log just gets a one-paragraph signal so the
    failure stays visible without polluting the knowledge base.
    """
    today = datetime.now(timezone.utc).astimezone()
    log_path = DAILY_DIR / f"{today.strftime('%Y-%m-%d')}.md"

    if not log_path.exists():
        DAILY_DIR.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"# Daily Log: {today.strftime('%Y-%m-%d')}\n\n## Sessions\n\n## Memory Maintenance\n\n",
            encoding="utf-8",
        )

    time_str = today.strftime("%H:%M")
    short_id = (session_id or "unknown")[:8]
    entry = (
        f"### [ERROR] Memory Flush Failed ({time_str})\n\n"
        f"Session `{short_id}`: {cause}\n\n"
        f"Full details: `{LOG_FILE}`\n\n"
    )

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)


RETRY_DELAYS = (5, 15, 45)  # seconds before retries 2, 3, 4

# Knobs (overridable via env). Backfill sets MAX_RETRIES=1 to avoid
# multiplying the per-session wait; SessionEnd hooks keep the default.
MAX_ATTEMPTS = max(1, int(os.environ.get("CMC_FLUSH_MAX_RETRIES", "3")))
ATTEMPT_TIMEOUT = max(30, int(os.environ.get("CMC_FLUSH_ATTEMPT_TIMEOUT", "180")))


def _is_transient_error(exc: Exception, stderr_text: str) -> bool:
    """Decide if a query() failure is worth retrying.

    Transient: auth races (401), rate limits (429), concurrent CLI lockfile
    collisions, generic "exit code 1" with no further detail.
    Non-transient: model validation errors, malformed prompts, etc.
    """
    msg = str(exc).lower()
    text = stderr_text.lower()
    # Hard non-transient signals — don't waste time retrying
    if "not a valid model" in text or "invalid model" in text:
        return False
    if "invalid api key" in text or "authentication is currently not supported" in text:
        return False
    # Transient signals
    if "401" in text or "auth" in text:
        return True
    if "429" in text or "rate limit" in text:
        return True
    if "shell is already running" in text:
        return True
    # Default: retry generic "exit code 1" failures (most common)
    if "exit code 1" in msg or "command failed" in msg:
        return True
    return False


# ── Prompt fragments shared across single-pass / partial / consolidation ──

_CAPTURE_RULES = """## Rôle et objectif

Tu es un assistant clair, précis et très efficace. Ton objectif est de
produire un dossier Markdown structuré qui permet à un humain de comprendre
profondément ce qui s'est passé dans la conversation, ce qui a été décidé,
et pourquoi cela compte.

Le dossier sert aussi de source brute pour `compile.py`, qui l'ingère ensuite
dans une base de connaissance. Tu dois donc être lisible pour l'humain ET
strict pour le pipeline CMC: non-hallucination, provenance explicite,
valeurs exactes, compatibilité avec la compilation aval.

## Règles critiques

1. **Écris en français.** Ne pose aucune question. Ne fais pas de quiz.
   Ne mentionne aucun quiz, évaluation ou assessment futur. Ton rôle est
   uniquement d'expliquer, organiser et documenter la conversation.

2. **N'invente RIEN.** Tu n'extrais que ce qui est présent dans la
   conversation. Pas de reconstruction vraisemblable, pas de pont logique
   inventé, pas de rationale deviné. Si une raison n'est pas explicite,
   indique `(raison non explicitée)`.

3. **Capture largement.** Préserve les problèmes, contraintes, options,
   branches, décisions, sous-décisions, vérifications, observations,
   hypothèses, chemins abandonnés, artefacts, commandes, fichiers,
   résultats et suites possibles. Si tu hésites à inclure une information
   concrète, inclus-la avec le bon niveau de prudence.

4. **Marque la provenance** au début des bullets qui portent une affirmation
   exploitable:
   - `[Établi]` — fait vérifié dans le code, la documentation ou par exécution
   - `[Décidé]` — choix fait pendant cette session
   - `[Hypothèse]` — supposition non vérifiée
   - `[Découvert]` — gotcha, observation ou comportement constaté

5. **Préserve VERBATIM les valeurs spécifiques.** Reformuler les phrases ne
   veut PAS dire paraphraser les valeurs concrètes. Garde mot pour mot, sans
   abréger ni arrondir:
   - identifiants, IDs, hashes, task IDs, Zap IDs, deal IDs, channel IDs
   - chemins de fichiers complets et noms de dossiers
   - URLs, IDs Sheets/Drive, endpoints, ports
   - versions, codes HTTP, codes d'erreur, timestamps, deadlines, durées
   - commandes shell, flags, variables d'environnement, noms de fonctions
   - formats, regex, schémas, champs, events, métriques, KPIs, seuils
   - snippets de texte, prompts, messages ou templates cités explicitement

6. **Explique progressivement si le sujet est technique.** D'abord le sens
   haut niveau, ensuite le détail concret, puis un exemple uniquement si la
   conversation en fournit un ou si l'exemple découle directement d'un fait
   explicite.

7. **Ne reproduis pas le dialogue brut.** Reformule en langage neutre et
   factuel. Préserve cependant les citations exactes uniquement quand la
   formulation verbatim porte une intention ou une décision importante.

8. **Ne filtre que le bruit.** Ignore les salutations, accusés de réception,
   répétitions littérales déjà capturées, et lectures/outils sans découverte.
   Tout le reste est conservé avec le bon marqueur de provenance."""

_OUTPUT_FORMAT = """## Format de sortie

Ta réponse DOIT commencer directement par `**Problème original**`. Pas de
préface, pas de titre global, pas de sentinel (`FLUSH_OK` / `PARTIAL_OK`) dans
le corps.

Pour une sortie non silencieuse, conserve les sections ci-dessous dans cet
ordre. Si une section n'a réellement aucun élément explicite, écris une seule
ligne: `- Aucun élément explicite dans la conversation.`

**Problème original**
- `[Établi]` Ce que l'humain voulait résoudre ou comprendre.
- `[Établi]` Pourquoi le problème existait.
- `[Établi]` Contexte, contraintes, branches, options ou interprétations
  mentionnées.

**Solution / résultat**
- `[Établi]` Ce qui a été décidé, créé, corrigé, clarifié ou vérifié.
- `[Décidé]` Pourquoi cette solution a été retenue, si la raison est explicite.
- `[Hypothèse]` Limites, risques ou cas non vérifiés, si présents.

**Raisonnement**
- `[Établi]` Le pourquoi derrière les décisions importantes.
- `[Établi]` Causes, hypothèses, conséquences, détails bas niveau et pivots.
- `[Découvert]` Observations qui ont modifié l'analyse ou la suite du travail.

**Contexte plus large**
- `[Établi]` Pourquoi le sujet compte.
- `[Établi]` Ce que cela affecte: fichiers, scripts, workflows, équipes,
  services, configurations, coûts ou dépendances.
- `[Hypothèse]` Suivi possible, dépendances ou points futurs non confirmés.

**Concepts clés**
- [ ] `[Établi]` Concept ou mécanisme — explication claire et utile.
- [ ] `[Décidé]` Décision structurante — conséquence pratique.
- [ ] `[Hypothèse]` Point à ne pas traiter comme fait établi.

**Décisions prises**
- `[Décidé]` Décision — raison explicite ou `(raison non explicitée)`.

Inclus les décisions macro ET micro: architecture, nommage, format, ordre,
prompt, fichier, commande, outil, seuil, option de configuration.

**Faits établis**
- `[Établi]` Fait vérifié — preuve ou contexte si mentionné.

Inclure les résultats de commandes, observations de fichiers, chemins,
versions, statuts, erreurs et sorties vérifiées.

**Hypothèses**
- `[Hypothèse]` Supposition non vérifiée — ce qui manque pour la confirmer.

Ne transforme jamais une hypothèse en fait.

**Découvertes**
- `[Découvert]` Gotcha, comportement, contrainte, surprise ou observation.

Inclure les bugs, limites, contraintes d'outils, comportements runtime et
détails utiles pour éviter de refaire la même erreur.

**Artefacts produits**
- `[Établi]` Type — chemin/référence verbatim — statut: créé / modifié /
  draft / posté / non posté / vérifié.

Inclure fichiers, scripts, configs, prompts, rapports, commandes importantes,
messages rédigés, docs ou outputs créés/modifiés.

**Actions / suites possibles**
- `[Établi]` Action explicitement demandée ou effectuée partiellement.
- `[Hypothèse]` Suite possible non décidée, si elle a été mentionnée comme
  telle.

Ne crée pas de todo inventé. Ne pose pas de question à l'humain."""

_LANG_RULE = """## Langue

La sortie finale doit être écrite en français. Garde les termes techniques,
noms de fonctions, bibliothèques, erreurs, commandes, chemins, URLs et valeurs
exactes dans leur forme originale."""


def _build_single_pass_prompt(context: str) -> str:
    return f"""Tu lis le contexte complet d'une conversation et tu produis directement
un dossier Markdown de daily log. Ce dossier doit aider l'humain à comprendre
la session plus tard, tout en restant exploitable par `compile.py`.

Le résultat alimente une base de connaissance via une compilation aval. Ce
que tu jettes ici est perdu pour de bon. Ce que tu inventes ici contamine
tout le KB.

N'utilise AUCUN outil. Réponds en texte brut uniquement.

{_CAPTURE_RULES}

{_OUTPUT_FORMAT}

## Mode silence

Si la session n'a aucun signal préservable (debug routinier sans insight,
exploration sans décision, exécution mécanique d'un plan), réponds
exactement: FLUSH_OK

{_LANG_RULE}

## Conversation

{context}"""


def _build_partial_prompt(chunk: str, idx: int, total: int) -> str:
    return f"""Tu lis UNE PARTIE d'un contexte de conversation plus long (partie {idx}/{total}).
Tu produis un dossier partiel avec la même structure que le daily log final.
Une étape de consolidation aval fusionnera tous les dossiers partiels en un
daily log unique.

Implications:
- Ne te soucie pas des doublons cross-parties — la consolidation déduplique.
- Capture exhaustivement ce qui est dans CETTE partie. Tu ne sauras pas ce
  qu'il y avait avant ou après.
- Si tu vois une référence à quelque chose qui semble venir d'avant
  ("le Zap mentionné", "la décision précédente"), garde-la telle quelle —
  la consolidation résoudra.

N'utilise AUCUN outil. Réponds en texte brut uniquement.

{_CAPTURE_RULES}

{_OUTPUT_FORMAT}

## Mode silence partiel

Si CETTE PARTIE n'a aucun signal préservable (uniquement des Read/Edit
routiniers, par exemple), réponds exactement: PARTIAL_OK

{_LANG_RULE}

## Partie de conversation ({idx}/{total})

{chunk}"""


def _build_consolidation_prompt(partials: list[str], total_chunks: int) -> str:
    parts_text = "\n\n".join(
        f"### Extraction partielle {i}/{total_chunks}\n\n{p}"
        for i, p in enumerate(partials, 1)
    )
    return f"""Tu reçois {total_chunks} dossiers partiels d'une même conversation
(traitée par chunks pour ne perdre aucun contexte). Fusionne-les en UN seul
daily log entry cohérent, lisible en français et compatible avec `compile.py`.

N'utilise AUCUN outil. Réponds en texte brut uniquement.

## Règles de fusion

1. **Fusionne par section dans le format final.** La sortie finale doit
   commencer par `**Problème original**` et suivre l'ordre exact défini
   ci-dessous. Les parties arrivent dans l'ordre temporel (partie 1 = début
   de session, partie N = fin). Préserve cette chronologie dans
   `**Raisonnement**`, `**Découvertes**`, `**Décisions prises**` et
   `**Actions / suites possibles**` quand elle est utile à la compréhension.

2. **Déduplique seulement les répétitions LITTÉRALES.** Si deux parties
   énoncent exactement le même fait avec les mêmes valeurs, garde une
   seule occurrence (préfère la version avec le rationale le plus
   explicite). MAIS si deux parties parlent d'un sujet voisin avec des
   nuances différentes (ex: une partie dit "Zap a 23 branches", une
   autre "12 branches actives"), garde les deux — c'est `compile.py`
   aval qui résoudra. Tu ne curates pas, tu agrèges.

3. **Résous les références cross-parties sans inventer.** Si une partie dit
   "le Zap mentionné" et une autre cite `Zap 174442936`, harmonise vers la
   valeur explicite. Si la référence reste ambiguë, garde l'ambiguïté avec
   un marqueur prudent au lieu de deviner.

4. **Préserve VERBATIM les valeurs spécifiques.** Mêmes règles que pour
   l'extraction: IDs, paths, URLs, latences, KPIs, events, formats — mot
   pour mot. Si une partie a une version précise et une autre a une
   version paraphrasée, garde la précise.

5. **Conserve les marqueurs de provenance** `[Établi]` / `[Décidé]` /
   `[Hypothèse]` / `[Découvert]`. Si deux parties tagguent différemment le
   même fait, prends le marqueur le plus prudent ([Hypothèse] > [Décidé] >
   [Établi] en termes de prudence).

6. **Si deux parties se contredisent**, garde les deux avec un marqueur
   temporel: "Initialement, X. Plus tard dans la session, après [trigger],
   non-X." Ne choisis pas silencieusement et n'écrase pas une contradiction.

7. **N'invente RIEN.** Aucun claim qui ne soit dans au moins une partie.
   Pas de "synthèse" qui extrapole au-delà du contenu. Pas de pont
   narratif entre deux chunks qui invente la transition — utilise les
   transitions explicites du contenu.

8. **Pas de remplissage inventé.** Pour une section sans information
   explicite, écris `- Aucun élément explicite dans la conversation.`
   plutôt que d'inventer un résumé.

{_OUTPUT_FORMAT}

## Mode silence consolidé

Si toutes les parties sont PARTIAL_OK ou si la fusion n'a aucun signal
préservable, réponds exactement: FLUSH_OK

{_LANG_RULE}

## Extractions partielles à fusionner

{parts_text}"""


def _chunk_at_turn_boundaries(context: str, max_chunk_size: int) -> list[str]:
    """Split context into chunks, snapping cuts to turn boundaries.

    Turn boundaries are lines starting with `**User:**`, `**Assistant:**`,
    or `**[Subagent: ...]**` (the format produced by extract_turns_from_jsonl
    in the hooks/backfill).

    Pathological case: a single turn larger than max_chunk_size. We then
    fall back to char-based split for that turn (rare, but possible if a
    user pastes a giant log).
    """
    import re
    if len(context) <= max_chunk_size:
        return [context]

    # Split keeping the boundary as part of the next segment.
    pattern = r"(?=\n\*\*(?:User|Assistant|\[Subagent))"
    turns = re.split(pattern, context)

    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for turn in turns:
        turn_len = len(turn)
        if turn_len > max_chunk_size:
            # Flush whatever we have, then char-split the giant turn
            if current:
                chunks.append("".join(current))
                current = []
                current_size = 0
            for i in range(0, turn_len, max_chunk_size):
                chunks.append(turn[i:i + max_chunk_size])
            continue

        if current_size + turn_len > max_chunk_size and current:
            chunks.append("".join(current))
            current = [turn]
            current_size = turn_len
        else:
            current.append(turn)
            current_size += turn_len

    if current:
        chunks.append("".join(current))
    return chunks


async def _llm_call(prompt: str, captured_stderr: list[str]) -> tuple[str, Exception | None]:
    """Single Claude Agent SDK call with retry on transient errors.

    Returns (response_text, terminal_exception_or_None). On success the
    exception is None. On terminal failure (non-transient or exhausted
    retries), exception is set and response_text is "".
    """
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    def stderr_callback(line: str) -> None:
        text = line.rstrip()
        if text:
            captured_stderr.append(text)
            logging.warning("[bundled CLI] %s", text)

    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            delay = RETRY_DELAYS[min(attempt - 2, len(RETRY_DELAYS) - 1)]
            logging.info("Retry attempt %d/%d after %ds", attempt, MAX_ATTEMPTS, delay)
            await asyncio.sleep(delay)

        attempt_stderr_start = len(captured_stderr)

        async def _run_query() -> str:
            local = ""
            async for message in query(
                prompt=prompt,
                options=ClaudeAgentOptions(
                    cwd=str(ROOT),
                    model=FLUSH_MODEL,
                    allowed_tools=[],
                    max_turns=2,
                    stderr=stderr_callback,
                ),
            ):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            local += block.text
                elif isinstance(message, ResultMessage):
                    pass
            return local

        try:
            response = await asyncio.wait_for(_run_query(), timeout=ATTEMPT_TIMEOUT)
            return response, None
        except asyncio.TimeoutError:
            last_exc = TimeoutError(f"bundled CLI hung for >{ATTEMPT_TIMEOUT}s — killed")
            logging.warning("Attempt %d/%d timed out after %ds", attempt, MAX_ATTEMPTS, ATTEMPT_TIMEOUT)
            continue
        except Exception as e:
            last_exc = e
            this_attempt_stderr = "\n".join(captured_stderr[attempt_stderr_start:])
            logging.warning("Attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, e)
            if not _is_transient_error(e, this_attempt_stderr):
                logging.info("Error is non-transient — skipping further retries")
                break

    import traceback
    logging.error("Agent SDK error after retries: %s\n%s", last_exc, traceback.format_exc())
    return "", last_exc


async def run_flush(context: str) -> tuple[str, list[str]]:
    """Extract knowledge from a conversation context.

    Architecture: dispatcher.
      - len(context) <= FLUSH_SINGLE_PASS_THRESHOLD → single LLM call
      - else → map-reduce: chunk at turn boundaries, partial flush per
        chunk, then a single consolidation call.

    Map-reduce ensures no content is dropped regardless of session length.

    Returns (response_text, captured_stderr_lines). On terminal failure the
    response is "FLUSH_ERROR: ...".
    """
    captured_stderr: list[str] = []

    # ── Single-pass path ──────────────────────────────────────────────
    if len(context) <= FLUSH_SINGLE_PASS_THRESHOLD:
        prompt = _build_single_pass_prompt(context)
        logging.info("Single-pass flush: %d chars", len(context))
        response, exc = await _llm_call(prompt, captured_stderr)
        if exc is not None:
            return f"FLUSH_ERROR: {type(exc).__name__}: {exc}", captured_stderr
        return response, captured_stderr

    # ── Map-reduce path ───────────────────────────────────────────────
    chunks = _chunk_at_turn_boundaries(context, FLUSH_CHUNK_SIZE)
    n = len(chunks)
    logging.info(
        "Map-reduce flush: %d chars → %d chunks (chunk size %d, threshold %d)",
        len(context), n, FLUSH_CHUNK_SIZE, FLUSH_SINGLE_PASS_THRESHOLD,
    )

    partials: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        logging.info("Partial flush %d/%d: %d chars", i, n, len(chunk))
        prompt = _build_partial_prompt(chunk, i, n)
        partial, exc = await _llm_call(prompt, captured_stderr)
        if exc is not None:
            # One chunk failed terminally — keep going if we have other
            # successful partials, otherwise bail.
            logging.warning("Partial %d/%d failed terminally: %s — continuing", i, n, exc)
            continue
        if partial.strip() == "PARTIAL_OK":
            logging.info("Partial %d/%d returned PARTIAL_OK (no signal)", i, n)
            continue
        partials.append(partial)

    if not partials:
        # Either every partial said PARTIAL_OK or every partial failed.
        # Treat as "nothing to save" if no terminal errors; otherwise FLUSH_ERROR.
        if captured_stderr:
            return "FLUSH_ERROR: all chunks failed or returned no signal", captured_stderr
        return "FLUSH_OK", captured_stderr

    if len(partials) == 1:
        # Only one chunk had signal — skip the consolidation pass.
        logging.info("Only one chunk had signal — using its output directly")
        return partials[0], captured_stderr

    # Consolidate all partial extractions
    logging.info("Consolidating %d partial extractions", len(partials))
    cons_prompt = _build_consolidation_prompt(partials, n)
    response, exc = await _llm_call(cons_prompt, captured_stderr)
    if exc is not None:
        # Consolidation failed — fall back to concatenated partials with a
        # marker, so we at least don't lose the extracted content.
        logging.error("Consolidation failed: %s — emitting raw concat fallback", exc)
        fallback = (
            f"**[Consolidation failed: {type(exc).__name__}. Raw partial extractions below.]**\n\n"
            + "\n\n---\n\n".join(
                f"### Partie {i}/{n}\n\n{p}" for i, p in enumerate(partials, 1)
            )
        )
        return fallback, captured_stderr
    return response, captured_stderr


COMPILE_AFTER_HOUR = 18  # 6 PM local time


def maybe_trigger_compilation() -> None:
    """If it's past the compile hour and today's log hasn't been compiled, run compile.py."""
    import subprocess as _sp

    now = datetime.now(timezone.utc).astimezone()
    if now.hour < COMPILE_AFTER_HOUR:
        return

    # Check if today's log has already been compiled
    today_log = f"{now.strftime('%Y-%m-%d')}.md"
    from config import STATE_FILE as compile_state_file
    if compile_state_file.exists():
        try:
            compile_state = json.loads(compile_state_file.read_text(encoding="utf-8"))
            ingested = compile_state.get("ingested", {})
            if today_log in ingested:
                # Already compiled today - check if the log has changed since
                from hashlib import sha256
                log_path = DAILY_DIR / today_log
                if log_path.exists():
                    current_hash = sha256(log_path.read_bytes()).hexdigest()[:16]
                    if ingested[today_log].get("hash") == current_hash:
                        return  # log unchanged since last compile
        except (json.JSONDecodeError, OSError):
            pass

    compile_script = SCRIPTS_DIR / "compile.py"
    if not compile_script.exists():
        return

    logging.info("End-of-day compilation triggered (after %d:00)", COMPILE_AFTER_HOUR)

    cmd = ["uv", "run", "--directory", str(ROOT), "python", str(compile_script)]

    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = _sp.CREATE_NEW_PROCESS_GROUP | _sp.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True

    try:
        log_handle = open(str(STATE_DIR / "compile.log"), "a")
        _sp.Popen(cmd, stdout=log_handle, stderr=_sp.STDOUT, cwd=str(ROOT), **kwargs)
    except Exception as e:
        logging.error("Failed to spawn compile.py: %s", e)


# Concurrency lock: prevents two flush.py instances racing for the bundled
# Claude CLI. Both SessionEnd hooks firing across multiple projects and
# /cmc-scan running while a Claude Code session is active can lead to
# concurrent invocations of the bundled CLI. The CLI v2.1.92 has an open
# auth race bug (anthropics/claude-code#44100) that surfaces as `exit code
# 1, Check stderr output for details`. The lock is VAULT-WIDE (lives in
# _config/.state/, not in per-project .state/) so that flushes from
# different projects also serialise.
LOCK_FILE = ROOT / ".state" / "flush.lock"
LOCK_STALE_SECONDS = 600  # 10 min — beyond any realistic flush duration
LOCK_WAIT_TIMEOUT = 90    # max seconds to wait for another flush to finish
LOCK_POLL_INTERVAL = 2    # check every 2s


def acquire_flush_lock() -> bool:
    """Acquire an exclusive flush lock for the whole vault.

    Waits up to LOCK_WAIT_TIMEOUT seconds for another flush to finish
    before giving up. Returns True if the lock was acquired, False if
    another flush.py is still holding it after the wait window.

    Caller can bypass the lock by setting CMC_FLUSH_SKIP_LOCK=1 — used
    by backfill.py's --parallel mode where the caller manages the
    concurrency budget itself.
    """
    if os.environ.get("CMC_FLUSH_SKIP_LOCK") == "1":
        return True
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    waited = 0
    while LOCK_FILE.exists():
        try:
            age = time.time() - LOCK_FILE.stat().st_mtime
        except OSError:
            age = 0
        if age >= LOCK_STALE_SECONDS:
            logging.warning("Stale lock found (%.1fs old) — overriding", age)
            break
        if waited >= LOCK_WAIT_TIMEOUT:
            logging.info(
                "Another flush.py held the lock for >%ds, skipping (lock %.1fs old)",
                waited, age,
            )
            return False
        time.sleep(LOCK_POLL_INTERVAL)
        waited += LOCK_POLL_INTERVAL
    try:
        LOCK_FILE.write_text(f"{os.getpid()} {time.time()}", encoding="utf-8")
    except OSError as e:
        logging.error("Failed to write lock file: %s", e)
        return False
    if waited > 0:
        logging.info("Acquired flush lock after waiting %ds", waited)
    return True


def release_flush_lock() -> None:
    if os.environ.get("CMC_FLUSH_SKIP_LOCK") == "1":
        return
    LOCK_FILE.unlink(missing_ok=True)


def main():
    _bootstrap_for_main()
    if len(sys.argv) < 3:
        logging.error(
            "Usage: %s <context_file.md> <session_id> [--label LABEL]",
            sys.argv[0],
        )
        sys.exit(1)

    context_file = Path(sys.argv[1])
    session_id = sys.argv[2]

    # Optional --label flag: identifies the flush trigger in the daily log
    # section header (e.g., "checkpoint", "final", "pre-compact"). Defaults
    # to bare "Session" for backward compatibility (backfill, manual runs).
    label = ""
    extra_args = sys.argv[3:]
    if "--label" in extra_args:
        idx = extra_args.index("--label")
        if idx + 1 < len(extra_args):
            label = extra_args[idx + 1]

    logging.info(
        "flush.py started for session %s, context: %s, label: %s",
        session_id, context_file, label or "(none)",
    )

    if not context_file.exists():
        logging.error("Context file not found: %s", context_file)
        return

    # Deduplication: skip if same session was flushed within 60 seconds.
    # The cursor mechanism in the hooks already prevents same-slice double
    # extraction, so this guard mainly protects against duplicate hook
    # firings (e.g., global+project-local both configured for SessionEnd)
    # racing past the cursor advance.
    state = load_flush_state()
    if (
        state.get("session_id") == session_id
        and time.time() - state.get("timestamp", 0) < 60
    ):
        logging.info("Skipping duplicate flush for session %s", session_id)
        context_file.unlink(missing_ok=True)
        return

    # Acquire concurrency lock — only one flush.py may invoke the Agent SDK
    # across the entire vault at a time. See comment near LOCK_FILE definition.
    if not acquire_flush_lock():
        # Don't unlink the context file — let the running instance finish
        # with it, or leave it for the next manual run.
        return

    try:
        # Read pre-extracted context
        context = context_file.read_text(encoding="utf-8").strip()
        if not context:
            logging.info("Context file is empty, skipping")
            context_file.unlink(missing_ok=True)
            return

        logging.info("Flushing session %s: %d chars", session_id, len(context))

        # Run the LLM extraction (now retries internally + captures stderr)
        response, captured_stderr = asyncio.run(run_flush(context))

        # Exact-match the sentinel rather than substring-match. The model
        # often echoes back fragments of the prompt (the prompt itself
        # contains the literal string "FLUSH_OK" as part of its
        # instructions), so a substring check silently discards full
        # extractions. Strip whitespace so trailing newlines don't matter.
        stripped = response.strip()
        if stripped == "FLUSH_OK":
            logging.info("Result: FLUSH_OK (skipped)")
        elif stripped.startswith("FLUSH_ERROR"):
            logging.error("Result: %s", response)
            cause = detect_failure_cause(captured_stderr, response)
            logging.error("Detected cause: %s", cause)
            append_error_marker_to_daily(session_id, cause)
        else:
            logging.info("Result: saved to daily log (%d chars)", len(response))
            short_id = (session_id or "unknown")[:8]
            section = (
                f"Session {short_id} {label}".strip()
                if label
                else f"Session {short_id}"
            )
            append_to_daily_log(response, section)

        # Update dedup state
        save_flush_state({"session_id": session_id, "timestamp": time.time()})

        # Clean up context file
        context_file.unlink(missing_ok=True)

        # End-of-day auto-compilation: if it's past the compile hour and today's
        # log hasn't been compiled yet, trigger compile.py in the background.
        maybe_trigger_compilation()

        logging.info("Flush complete for session %s", session_id)
    finally:
        release_flush_lock()


if __name__ == "__main__":
    main()
