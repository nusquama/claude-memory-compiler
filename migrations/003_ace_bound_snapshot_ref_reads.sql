-- Keep identity reads small and avoid carrying complete snapshot JSON through
-- the ranking CTE. The old SELECT r.* shape made a native tick exceed the
-- launchd transport when the queue contained long conversations.
BEGIN;

CREATE OR REPLACE FUNCTION ace.pending_snapshot_refs_window(
    p_limit integer,
    p_stage text,
    p_project_id uuid,
    p_source_after timestamptz DEFAULT NULL,
    p_source_before timestamptz DEFAULT NULL
)
RETURNS TABLE (
    project_id uuid,
    source text,
    session_id text,
    revision text,
    source_path text,
    host_id text,
    started_at timestamptz,
    updated_at timestamptz,
    message_count integer
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ace, pg_temp
AS $$
    WITH latest AS (
        SELECT
            r.project_id,
            r.source,
            r.session_id,
            r.revision,
            r.source_path,
            r.host_id,
            r.started_at,
            r.updated_at,
            r.received_at,
            jsonb_array_length(COALESCE(r.snapshot->'messages', '[]'::jsonb)) AS message_count,
            row_number() OVER (
                PARTITION BY r.project_id, r.source, r.session_id
                ORDER BY r.updated_at DESC NULLS LAST, r.received_at DESC, r.revision DESC
            ) AS row_number
        FROM revisions r
        JOIN projects p ON p.id = r.project_id
        WHERE p.enabled
          AND p.initialized
          AND (p_project_id IS NULL OR r.project_id = p_project_id)
          AND (p_source_after IS NULL OR COALESCE(r.updated_at, r.received_at) >= p_source_after)
          AND (p_source_before IS NULL OR COALESCE(r.updated_at, r.received_at) < p_source_before)
    ),
    candidates AS (
        SELECT
            latest.project_id,
            latest.source,
            latest.session_id,
            latest.revision,
            latest.source_path,
            latest.host_id,
            latest.started_at,
            latest.updated_at,
            latest.received_at,
            latest.message_count,
            pr.run_id AS attempt_id,
            COALESCE(pr.finished_at, pr.started_at, pr.created_at) AS last_attempt_at
        FROM latest
        LEFT JOIN processing_runs pr
          ON pr.project_id = latest.project_id
         AND pr.source = latest.source
         AND pr.session_id = latest.session_id
         AND pr.revision = latest.revision
         AND pr.stage = p_stage
        WHERE latest.row_number = 1
          AND (pr.run_id IS NULL OR pr.status <> 'succeeded')
    )
    SELECT
        candidates.project_id,
        candidates.source,
        candidates.session_id,
        candidates.revision,
        candidates.source_path,
        candidates.host_id,
        candidates.started_at,
        candidates.updated_at,
        candidates.message_count
    FROM candidates
    WHERE p_limit BETWEEN 1 AND 500
      AND p_stage IN ('extraction', 'analysis', 'result', 'decision', 'evaluation', 'compile')
    ORDER BY CASE WHEN candidates.attempt_id IS NULL THEN 0 ELSE 1 END,
             candidates.last_attempt_at ASC NULLS FIRST,
             candidates.updated_at ASC NULLS LAST,
             candidates.received_at ASC,
             candidates.revision ASC
    LIMIT p_limit;
$$;

CREATE OR REPLACE FUNCTION ace.pending_snapshot_refs(
    p_limit integer DEFAULT 100,
    p_stage text DEFAULT 'extraction',
    p_project_id uuid DEFAULT NULL
)
RETURNS TABLE (
    project_id uuid,
    source text,
    session_id text,
    revision text,
    source_path text,
    host_id text,
    started_at timestamptz,
    updated_at timestamptz,
    message_count integer
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ace, pg_temp
AS $$
    SELECT * FROM ace.pending_snapshot_refs_window(
        p_limit, p_stage, p_project_id, NULL::timestamptz, NULL::timestamptz
    );
$$;

CREATE OR REPLACE FUNCTION ace.pending_snapshot_refs_since(
    p_limit integer,
    p_stage text,
    p_project_id uuid,
    p_source_after timestamptz
)
RETURNS TABLE (
    project_id uuid,
    source text,
    session_id text,
    revision text,
    source_path text,
    host_id text,
    started_at timestamptz,
    updated_at timestamptz,
    message_count integer
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ace, pg_temp
AS $$
    SELECT * FROM ace.pending_snapshot_refs_window(
        p_limit, p_stage, p_project_id, p_source_after, NULL::timestamptz
    );
$$;

INSERT INTO ace.schema_migrations (version, description)
VALUES (3, 'ACE bounded snapshot reference reads')
ON CONFLICT (version) DO NOTHING;

COMMIT;
