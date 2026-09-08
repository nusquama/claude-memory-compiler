-- ACE additive stage leases and source-time windows.
--
-- Migration 001 remains unchanged.  A lease is keyed by the complete
-- processing identity and is acquired by one SECURITY DEFINER RPC, so two
-- hosts cannot process the same stage concurrently.  Expiry is reclaimable;
-- the old owner cannot mark a stage after another owner has taken it.

BEGIN;

CREATE TABLE IF NOT EXISTS ace.processing_leases (
    lease_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL,
    source text NOT NULL CHECK (source ~ '^[a-z0-9][a-z0-9_.:-]{0,63}$'),
    session_id text NOT NULL CHECK (char_length(session_id) BETWEEN 1 AND 512),
    revision text NOT NULL CHECK (revision ~ '^[0-9a-f]{64}$'),
    stage text NOT NULL CHECK (stage IN ('extraction', 'analysis', 'result', 'decision', 'evaluation', 'compile')),
    lease_owner text NOT NULL CHECK (char_length(lease_owner) BETWEEN 1 AND 256),
    host_id text NOT NULL CHECK (char_length(host_id) BETWEEN 1 AND 256),
    lease_until timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT processing_leases_stage_key UNIQUE (project_id, source, session_id, revision, stage),
    FOREIGN KEY (project_id, source, session_id, revision)
        REFERENCES ace.revisions(project_id, source, session_id, revision)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS processing_leases_expiry_idx
    ON ace.processing_leases (lease_until);

ALTER TABLE ace.processing_leases ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON ace.processing_leases FROM PUBLIC;

-- The caller supplies a bounded owner token.  A repeated call by that same
-- owner renews the existing lease, while an active lease owned by another
-- host produces claimed=false without exposing its token.
CREATE OR REPLACE FUNCTION ace.claim_stage(
    p_project_id uuid,
    p_source text,
    p_session_id text,
    p_revision text,
    p_stage text,
    p_lease_owner text,
    p_host_id text,
    p_lease_seconds integer DEFAULT 1800
)
RETURNS TABLE (
    claimed boolean,
    lease_id uuid,
    project_id uuid,
    source text,
    session_id text,
    revision text,
    stage text,
    lease_owner text,
    host_id text,
    lease_until timestamptz
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ace, pg_temp
AS $$
DECLARE
    v_now timestamptz := clock_timestamp();
    v_lease_until timestamptz;
    v_lease processing_leases;
    v_source text := lower(p_source);
    v_revision text := lower(p_revision);
BEGIN
    IF p_project_id IS NULL
       OR p_source IS NULL OR v_source !~ '^[a-z0-9][a-z0-9_.:-]{0,63}$'
       OR p_session_id IS NULL OR char_length(p_session_id) NOT BETWEEN 1 AND 512
       OR p_revision IS NULL OR v_revision !~ '^[0-9a-f]{64}$'
       OR p_stage IS NULL OR p_stage NOT IN ('extraction', 'analysis', 'result', 'decision', 'evaluation', 'compile')
       OR p_lease_owner IS NULL OR char_length(p_lease_owner) NOT BETWEEN 1 AND 256
       OR p_host_id IS NULL OR char_length(p_host_id) NOT BETWEEN 1 AND 256
       OR p_lease_seconds IS NULL OR p_lease_seconds NOT BETWEEN 1 AND 86400 THEN
        RAISE EXCEPTION 'invalid ACE stage lease bounds' USING ERRCODE = '22023';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM revisions r
        JOIN projects p ON p.id = r.project_id
        WHERE r.project_id = p_project_id
          AND r.source = v_source
          AND r.session_id = p_session_id
          AND r.revision = v_revision
          AND p.enabled AND p.initialized
    ) THEN
        RAISE EXCEPTION 'ACE revision is not eligible for stage processing' USING ERRCODE = '22023';
    END IF;

    -- A terminal stage is already complete and must never be reopened by a
    -- late worker.  The unique stage key serialises this read with a claim.
    IF EXISTS (
        SELECT 1
        FROM processing_runs pr
        WHERE pr.project_id = p_project_id
          AND pr.source = v_source
          AND pr.session_id = p_session_id
          AND pr.revision = v_revision
          AND pr.stage = p_stage
          AND pr.status = 'succeeded'
    ) THEN
        RETURN QUERY SELECT false, NULL::uuid, p_project_id, v_source,
                            p_session_id, v_revision, p_stage, NULL::text,
                            p_host_id, NULL::timestamptz;
        RETURN;
    END IF;

    v_lease_until := v_now + make_interval(secs => p_lease_seconds::double precision);
    INSERT INTO processing_leases (
        project_id, source, session_id, revision, stage,
        lease_owner, host_id, lease_until, created_at, updated_at
    )
    VALUES (
        p_project_id, v_source, p_session_id, v_revision, p_stage,
        p_lease_owner, p_host_id, v_lease_until, v_now, v_now
    )
    ON CONFLICT ON CONSTRAINT processing_leases_stage_key DO UPDATE
    SET lease_owner = EXCLUDED.lease_owner,
        host_id = EXCLUDED.host_id,
        lease_until = EXCLUDED.lease_until,
        updated_at = v_now
    WHERE processing_leases.lease_until <= v_now
       OR (processing_leases.lease_owner = p_lease_owner
           AND processing_leases.host_id = p_host_id)
    RETURNING * INTO v_lease;

    IF NOT FOUND THEN
        RETURN QUERY SELECT false, NULL::uuid, p_project_id, v_source,
                            p_session_id, v_revision, p_stage, NULL::text,
                            p_host_id, NULL::timestamptz;
        RETURN;
    END IF;

    -- A competing owner may have been waiting on the same unique lease row
    -- while the previous owner committed a terminal success.  Recheck after
    -- the INSERT/UPDATE lock is acquired, otherwise that waiter could claim
    -- work that is already complete.
    IF EXISTS (
        SELECT 1
        FROM processing_runs pr
        WHERE pr.project_id = p_project_id
          AND pr.source = v_source
          AND pr.session_id = p_session_id
          AND pr.revision = v_revision
          AND pr.stage = p_stage
          AND pr.status = 'succeeded'
    ) THEN
        DELETE FROM processing_leases
        WHERE processing_leases.lease_id = v_lease.lease_id;
        RETURN QUERY SELECT false, NULL::uuid, p_project_id, v_source,
                            p_session_id, v_revision, p_stage, NULL::text,
                            p_host_id, NULL::timestamptz;
        RETURN;
    END IF;

    RETURN QUERY SELECT true, v_lease.lease_id, v_lease.project_id,
                        v_lease.source, v_lease.session_id, v_lease.revision,
                        v_lease.stage, v_lease.lease_owner, v_lease.host_id,
                        v_lease.lease_until;
END
$$;

-- Release is idempotent.  A stale owner cannot remove a lease reclaimed by a
-- newer owner because both the owner and host are part of the predicate.
CREATE OR REPLACE FUNCTION ace.release_stage(
    p_project_id uuid,
    p_source text,
    p_session_id text,
    p_revision text,
    p_stage text,
    p_lease_owner text,
    p_host_id text,
    p_outcome text DEFAULT 'failed'
)
RETURNS TABLE (
    released boolean,
    lease_id uuid,
    project_id uuid,
    source text,
    session_id text,
    revision text,
    stage text,
    outcome text
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ace, pg_temp
AS $$
DECLARE
    v_lease processing_leases;
    v_source text := lower(p_source);
    v_revision text := lower(p_revision);
BEGIN
    IF p_project_id IS NULL
       OR p_source IS NULL OR v_source !~ '^[a-z0-9][a-z0-9_.:-]{0,63}$'
       OR p_session_id IS NULL OR char_length(p_session_id) NOT BETWEEN 1 AND 512
       OR p_revision IS NULL OR v_revision !~ '^[0-9a-f]{64}$'
       OR p_stage IS NULL OR p_stage NOT IN ('extraction', 'analysis', 'result', 'decision', 'evaluation', 'compile')
       OR p_lease_owner IS NULL OR char_length(p_lease_owner) NOT BETWEEN 1 AND 256
       OR p_host_id IS NULL OR char_length(p_host_id) NOT BETWEEN 1 AND 256
       OR p_outcome IS NULL OR char_length(p_outcome) NOT BETWEEN 1 AND 64 THEN
        RAISE EXCEPTION 'invalid ACE stage release bounds' USING ERRCODE = '22023';
    END IF;

    DELETE FROM processing_leases
    WHERE processing_leases.project_id = p_project_id
      AND processing_leases.source = v_source
      AND processing_leases.session_id = p_session_id
      AND processing_leases.revision = v_revision
      AND processing_leases.stage = p_stage
      AND processing_leases.lease_owner = p_lease_owner
      AND processing_leases.host_id = p_host_id
    RETURNING * INTO v_lease;

    IF NOT FOUND THEN
        RETURN QUERY SELECT false, NULL::uuid, p_project_id, v_source,
                            p_session_id, v_revision, p_stage, p_outcome;
        RETURN;
    END IF;

    RETURN QUERY SELECT true, v_lease.lease_id, v_lease.project_id,
                        v_lease.source, v_lease.session_id, v_lease.revision,
                        v_lease.stage, p_outcome;
END
$$;

-- Reclaim is bounded so a maintenance call cannot turn an expired backlog
-- into an unbounded transaction.  Claims also reclaim one matching expired
-- row inline, so this function is optional housekeeping.
CREATE OR REPLACE FUNCTION ace.expire_stage_leases(
    p_limit integer DEFAULT 500
)
RETURNS TABLE (expired_count integer)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ace, pg_temp
AS $$
BEGIN
    IF p_limit IS NULL OR p_limit NOT BETWEEN 1 AND 500 THEN
        RAISE EXCEPTION 'invalid ACE expired-lease limit' USING ERRCODE = '22023';
    END IF;

    RETURN QUERY
    WITH doomed AS (
        SELECT lease_id
        FROM processing_leases
        WHERE lease_until <= clock_timestamp()
        ORDER BY lease_until, lease_id
        LIMIT p_limit
        FOR UPDATE SKIP LOCKED
    ), deleted AS (
        DELETE FROM processing_leases l
        USING doomed d
        WHERE l.lease_id = d.lease_id
        RETURNING l.lease_id
    )
    SELECT count(*)::integer FROM deleted;
END
$$;

-- A stage outcome is accepted only while the exact owner still holds an
-- unexpired lease.  Terminal outcomes remove that lease in the same
-- transaction; a later owner can claim the next pending stage immediately.
CREATE OR REPLACE FUNCTION ace.mark_stage(
    p_source text,
    p_session_id text,
    p_revision text,
    p_project_id uuid,
    p_stage text,
    p_status text,
    p_lease_owner text,
    p_host_id text,
    p_error text DEFAULT NULL
)
RETURNS TABLE (
    run_id uuid,
    project_id uuid,
    source text,
    session_id text,
    revision text,
    stage text,
    status text,
    error_type text
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ace, pg_temp
AS $$
DECLARE
    v_run processing_runs;
    v_lease processing_leases;
    v_attempted_at timestamptz := clock_timestamp();
    v_source text := lower(p_source);
    v_revision text := lower(p_revision);
BEGIN
    IF p_source IS NULL OR v_source !~ '^[a-z0-9][a-z0-9_.:-]{0,63}$'
       OR p_session_id IS NULL OR char_length(p_session_id) NOT BETWEEN 1 AND 512
       OR p_revision IS NULL OR v_revision !~ '^[0-9a-f]{64}$'
       OR p_project_id IS NULL
       OR p_stage IS NULL OR p_stage NOT IN ('extraction', 'analysis', 'result', 'decision', 'evaluation', 'compile')
       OR p_status IS NULL OR p_status NOT IN ('pending', 'running', 'succeeded', 'failed', 'skipped')
       OR p_lease_owner IS NULL OR char_length(p_lease_owner) NOT BETWEEN 1 AND 256
       OR p_host_id IS NULL OR char_length(p_host_id) NOT BETWEEN 1 AND 256
       OR (p_error IS NOT NULL AND char_length(p_error) > 256) THEN
        RAISE EXCEPTION 'invalid ACE lease-bound processing status' USING ERRCODE = '22023';
    END IF;

    SELECT * INTO v_lease
    FROM processing_leases
    WHERE processing_leases.project_id = p_project_id
      AND processing_leases.source = v_source
      AND processing_leases.session_id = p_session_id
      AND processing_leases.revision = v_revision
      AND processing_leases.stage = p_stage
      AND processing_leases.lease_owner = p_lease_owner
      AND processing_leases.host_id = p_host_id
      AND processing_leases.lease_until > v_attempted_at
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'ACE stage lease is missing or expired' USING ERRCODE = '55000';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM revisions
        WHERE revisions.project_id = p_project_id
          AND revisions.source = v_source
          AND revisions.session_id = p_session_id
          AND revisions.revision = v_revision
    ) THEN
        RAISE EXCEPTION 'ACE revision is not registered' USING ERRCODE = '22023';
    END IF;

    INSERT INTO processing_runs (
        project_id, source, session_id, revision, stage, status,
        error_type, started_at, finished_at
    )
    VALUES (
        p_project_id, v_source, p_session_id, v_revision, p_stage, p_status,
        NULLIF(p_error, ''), v_attempted_at,
        CASE WHEN p_status IN ('succeeded', 'failed', 'skipped')
             THEN v_attempted_at ELSE NULL END
    )
    ON CONFLICT ON CONSTRAINT processing_runs_project_id_source_session_id_revision_stage_key DO UPDATE
    SET status = EXCLUDED.status,
        error_type = EXCLUDED.error_type,
        started_at = EXCLUDED.started_at,
        finished_at = EXCLUDED.finished_at;

    SELECT * INTO v_run
    FROM processing_runs
    WHERE processing_runs.project_id = p_project_id
      AND processing_runs.source = v_source
      AND processing_runs.session_id = p_session_id
      AND processing_runs.revision = v_revision
      AND processing_runs.stage = p_stage;

    IF p_status IN ('succeeded', 'failed', 'skipped') THEN
        DELETE FROM processing_leases WHERE lease_id = v_lease.lease_id;
    END IF;

    RETURN QUERY SELECT v_run.run_id, v_run.project_id, v_run.source,
                        v_run.session_id, v_run.revision, v_run.stage,
                        v_run.status, v_run.error_type;
END
$$;

-- Source-time windows are additive because the original pending functions may
-- already be installed.  The half-open [after, before) bounds are applied in
-- the candidate CTE before LIMIT, using the same source timestamp preference
-- as the pipeline: updated_at, then received_at.  Session started_at is not a
-- proxy for a message arriving on the requested day.
CREATE OR REPLACE FUNCTION ace.pending_snapshots_window(
    p_limit integer,
    p_stage text,
    p_project_id uuid,
    p_source_after timestamptz DEFAULT NULL,
    p_source_before timestamptz DEFAULT NULL
)
RETURNS TABLE (
    envelope jsonb,
    project_id uuid,
    source text,
    session_id text,
    revision text
)
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = ace, pg_temp
AS $$
BEGIN
    IF p_limit IS NULL OR p_limit NOT BETWEEN 1 AND 500
       OR p_stage IS NULL OR p_stage NOT IN ('extraction', 'analysis', 'result', 'decision', 'evaluation', 'compile')
       OR (p_source_after IS NOT NULL AND p_source_before IS NOT NULL AND p_source_after > p_source_before) THEN
        RAISE EXCEPTION 'invalid ACE source-time window bounds' USING ERRCODE = '22023';
    END IF;

    RETURN QUERY
    WITH latest AS (
        SELECT r.*,
               row_number() OVER (
                   PARTITION BY r.project_id, r.source, r.session_id
                   ORDER BY r.updated_at DESC NULLS LAST, r.received_at DESC, r.revision DESC
               ) AS row_number
        FROM revisions r
        JOIN projects p ON p.id = r.project_id
        WHERE p.enabled AND p.initialized
          AND (p_project_id IS NULL OR r.project_id = p_project_id)
          AND (p_source_after IS NULL OR COALESCE(r.updated_at, r.received_at) >= p_source_after)
          AND (p_source_before IS NULL OR COALESCE(r.updated_at, r.received_at) < p_source_before)
    ), candidates AS (
        SELECT latest.*,
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
    SELECT candidates.snapshot, candidates.project_id, candidates.source,
           candidates.session_id, candidates.revision
    FROM candidates
    ORDER BY CASE WHEN candidates.attempt_id IS NULL THEN 0 ELSE 1 END,
             candidates.last_attempt_at ASC NULLS FIRST,
             candidates.updated_at ASC NULLS FIRST,
             candidates.received_at ASC, candidates.revision ASC
    LIMIT p_limit;
END
$$;

CREATE OR REPLACE FUNCTION ace.pending_snapshots_since(
    p_limit integer,
    p_stage text,
    p_project_id uuid,
    p_source_after timestamptz
)
RETURNS TABLE (
    envelope jsonb,
    project_id uuid,
    source text,
    session_id text,
    revision text
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ace, pg_temp
AS $$
    SELECT * FROM ace.pending_snapshots_window(
        p_limit, p_stage, p_project_id, p_source_after, NULL::timestamptz
    );
$$;

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
        SELECT r.*,
               row_number() OVER (
                   PARTITION BY r.project_id, r.source, r.session_id
                   ORDER BY r.updated_at DESC NULLS LAST, r.received_at DESC, r.revision DESC
               ) AS row_number
        FROM revisions r
        JOIN projects p ON p.id = r.project_id
        WHERE p.enabled AND p.initialized
          AND (p_project_id IS NULL OR r.project_id = p_project_id)
          AND (p_source_after IS NULL OR COALESCE(r.updated_at, r.received_at) >= p_source_after)
          AND (p_source_before IS NULL OR COALESCE(r.updated_at, r.received_at) < p_source_before)
    ), candidates AS (
        SELECT latest.*,
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
    SELECT candidates.project_id, candidates.source, candidates.session_id,
           candidates.revision, candidates.source_path, candidates.host_id,
           candidates.started_at, candidates.updated_at,
           jsonb_array_length(COALESCE(candidates.snapshot->'messages', '[]'::jsonb))
    FROM candidates
    WHERE p_limit BETWEEN 1 AND 500
      AND p_stage IN ('extraction', 'analysis', 'result', 'decision', 'evaluation', 'compile')
    ORDER BY CASE WHEN candidates.attempt_id IS NULL THEN 0 ELSE 1 END,
             candidates.last_attempt_at ASC NULLS FIRST,
             candidates.updated_at ASC NULLS LAST,
             candidates.received_at ASC, candidates.revision ASC
    LIMIT p_limit;
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

-- The old seven-argument function cannot prove ownership.  Keep its
-- definition for already-recorded history, but remove processor execution so
-- every new stage outcome goes through mark_stage's lease check.
REVOKE EXECUTE ON FUNCTION ace.mark_processed(text, text, text, uuid, text, text, text) FROM ace_processor;

REVOKE EXECUTE ON FUNCTION ace.claim_stage(uuid, text, text, text, text, text, text, integer) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION ace.release_stage(uuid, text, text, text, text, text, text, text) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION ace.expire_stage_leases(integer) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION ace.mark_stage(text, text, text, uuid, text, text, text, text, text) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION ace.pending_snapshots_window(integer, text, uuid, timestamptz, timestamptz) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION ace.pending_snapshots_since(integer, text, uuid, timestamptz) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION ace.pending_snapshot_refs_window(integer, text, uuid, timestamptz, timestamptz) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION ace.pending_snapshot_refs_since(integer, text, uuid, timestamptz) FROM PUBLIC;

GRANT EXECUTE ON FUNCTION ace.claim_stage(uuid, text, text, text, text, text, text, integer) TO ace_processor;
GRANT EXECUTE ON FUNCTION ace.release_stage(uuid, text, text, text, text, text, text, text) TO ace_processor;
GRANT EXECUTE ON FUNCTION ace.expire_stage_leases(integer) TO ace_processor;
GRANT EXECUTE ON FUNCTION ace.mark_stage(text, text, text, uuid, text, text, text, text, text) TO ace_processor;
GRANT EXECUTE ON FUNCTION ace.pending_snapshots_window(integer, text, uuid, timestamptz, timestamptz) TO ace_processor;
GRANT EXECUTE ON FUNCTION ace.pending_snapshots_since(integer, text, uuid, timestamptz) TO ace_processor;
GRANT EXECUTE ON FUNCTION ace.pending_snapshot_refs_window(integer, text, uuid, timestamptz, timestamptz) TO ace_processor;
GRANT EXECUTE ON FUNCTION ace.pending_snapshot_refs_since(integer, text, uuid, timestamptz) TO ace_processor;

INSERT INTO ace.schema_migrations (version, description)
VALUES (2, 'ACE atomic stage leases and source-time windows')
ON CONFLICT (version) DO NOTHING;

COMMIT;
