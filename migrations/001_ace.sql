-- ACE Supabase transport foundation.
--
-- This migration is deliberately self-contained.  It creates only the
-- canonical, unquoted `ace` schema and the three dedicated NOLOGIN roles
-- used by the transport.  The application talks to the fixed SECURITY
-- DEFINER functions below; it never receives an arbitrary SQL surface.

BEGIN;

CREATE SCHEMA IF NOT EXISTS ace;

-- UUID generation is used for internal records.  The project UUID itself is
-- supplied by the caller and is never generated from an untrusted envelope.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS ace.schema_migrations (
    version integer PRIMARY KEY,
    description text NOT NULL CHECK (char_length(description) BETWEEN 1 AND 200),
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ace.projects (
    id uuid PRIMARY KEY,
    name text NOT NULL CHECK (char_length(name) BETWEEN 1 AND 200),
    root text NOT NULL CHECK (char_length(root) BETWEEN 1 AND 2048),
    vault_dir text NOT NULL CHECK (char_length(vault_dir) BETWEEN 1 AND 2048),
    enabled boolean NOT NULL DEFAULT false,
    initialized boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ace.sessions (
    project_id uuid NOT NULL REFERENCES ace.projects(id) ON DELETE RESTRICT,
    source text NOT NULL CHECK (source ~ '^[a-z0-9][a-z0-9_.:-]{0,63}$'),
    session_id text NOT NULL CHECK (char_length(session_id) BETWEEN 1 AND 512),
    started_at timestamptz,
    updated_at timestamptz,
    latest_revision text CHECK (latest_revision IS NULL OR latest_revision ~ '^[0-9a-f]{64}$'),
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (project_id, source, session_id)
);

CREATE TABLE IF NOT EXISTS ace.revisions (
    project_id uuid NOT NULL REFERENCES ace.projects(id) ON DELETE RESTRICT,
    source text NOT NULL CHECK (source ~ '^[a-z0-9][a-z0-9_.:-]{0,63}$'),
    session_id text NOT NULL CHECK (char_length(session_id) BETWEEN 1 AND 512),
    revision text NOT NULL CHECK (revision ~ '^[0-9a-f]{64}$'),
    source_path text CHECK (source_path IS NULL OR char_length(source_path) <= 2048),
    host_id text CHECK (host_id IS NULL OR char_length(host_id) <= 256),
    started_at timestamptz,
    updated_at timestamptz,
    snapshot jsonb NOT NULL,
    received_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (project_id, source, session_id, revision)
);

CREATE TABLE IF NOT EXISTS ace.messages (
    message_pk uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL,
    source text NOT NULL,
    session_id text NOT NULL,
    revision text NOT NULL,
    message_id text NOT NULL CHECK (char_length(message_id) BETWEEN 1 AND 512),
    ordinal integer NOT NULL CHECK (ordinal >= 0 AND ordinal <= 1000000),
    role text NOT NULL CHECK (char_length(role) BETWEEN 1 AND 64),
    message_type text NOT NULL CHECK (char_length(message_type) BETWEEN 1 AND 64),
    message_timestamp timestamptz,
    content jsonb NOT NULL,
    call_id text CHECK (call_id IS NULL OR char_length(call_id) <= 512),
    status text CHECK (status IS NULL OR char_length(status) <= 128),
    model text CHECK (model IS NULL OR char_length(model) <= 256),
    refs jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (project_id, source, session_id, revision)
        REFERENCES ace.revisions(project_id, source, session_id, revision)
        ON DELETE RESTRICT
);

-- message_id is intentionally not unique.  A conversation can contain
-- repeated stable identifiers, and two conversations must never collide.
CREATE INDEX IF NOT EXISTS messages_revision_idx
    ON ace.messages (project_id, source, session_id, revision, ordinal);

CREATE TABLE IF NOT EXISTS ace.attachments (
    attachment_pk uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL,
    source text NOT NULL,
    session_id text NOT NULL,
    revision text NOT NULL,
    attachment_id text CHECK (attachment_id IS NULL OR char_length(attachment_id) <= 512),
    name text CHECK (name IS NULL OR char_length(name) <= 512),
    mime_type text CHECK (mime_type IS NULL OR char_length(mime_type) <= 256),
    kind text CHECK (kind IS NULL OR char_length(kind) <= 128),
    size_bytes bigint CHECK (size_bytes IS NULL OR (size_bytes >= 0 AND size_bytes <= 1073741824)),
    sha256 text CHECK (sha256 IS NULL OR sha256 ~ '^[0-9a-f]{64}$'),
    uri text CHECK (uri IS NULL OR char_length(uri) <= 4096),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (project_id, source, session_id, revision)
        REFERENCES ace.revisions(project_id, source, session_id, revision)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS revisions_pending_idx
    ON ace.revisions (project_id, source, session_id, updated_at DESC, received_at DESC);

CREATE TABLE IF NOT EXISTS ace.processing_runs (
    run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL,
    source text NOT NULL,
    session_id text NOT NULL,
    revision text NOT NULL,
    stage text NOT NULL CHECK (stage IN ('extraction', 'analysis', 'result', 'decision', 'evaluation', 'compile')),
    status text NOT NULL CHECK (status IN ('pending', 'running', 'succeeded', 'failed', 'skipped')),
    error_type text CHECK (error_type IS NULL OR char_length(error_type) <= 256),
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (project_id, source, session_id, revision, stage),
    FOREIGN KEY (project_id, source, session_id, revision)
        REFERENCES ace.revisions(project_id, source, session_id, revision)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS processing_runs_pending_idx
    ON ace.processing_runs (project_id, source, session_id, stage, status, created_at);

CREATE TABLE IF NOT EXISTS ace.observations (
    observation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL,
    source text NOT NULL,
    session_id text NOT NULL,
    revision text NOT NULL,
    problem_signature text CHECK (problem_signature IS NULL OR char_length(problem_signature) <= 1000),
    success boolean,
    preference_evidence jsonb,
    evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (project_id, source, session_id, revision)
        REFERENCES ace.revisions(project_id, source, session_id, revision)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS observations_history_idx
    ON ace.observations (project_id, created_at DESC);

CREATE TABLE IF NOT EXISTS ace.recommendations (
    recommendation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL,
    source text NOT NULL,
    session_id text NOT NULL,
    revision text NOT NULL,
    recommendation text NOT NULL CHECK (char_length(recommendation) BETWEEN 1 AND 4000),
    rationale jsonb NOT NULL DEFAULT '{}'::jsonb,
    evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (project_id, source, session_id, revision)
        REFERENCES ace.revisions(project_id, source, session_id, revision)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS recommendations_history_idx
    ON ace.recommendations (project_id, created_at DESC);

CREATE TABLE IF NOT EXISTS ace.results (
    result_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL,
    source text NOT NULL,
    session_id text NOT NULL,
    revision text NOT NULL,
    result jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (project_id, source, session_id, revision)
        REFERENCES ace.revisions(project_id, source, session_id, revision)
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS ace.decisions (
    decision_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL,
    source text NOT NULL,
    session_id text NOT NULL,
    revision text NOT NULL,
    decision jsonb NOT NULL,
    actor text CHECK (actor IS NULL OR char_length(actor) <= 256),
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (project_id, source, session_id, revision)
        REFERENCES ace.revisions(project_id, source, session_id, revision)
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS ace.corrections (
    correction_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL,
    source text NOT NULL,
    session_id text NOT NULL,
    revision text NOT NULL,
    correction jsonb NOT NULL,
    actor text CHECK (actor IS NULL OR char_length(actor) <= 256),
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (project_id, source, session_id, revision)
        REFERENCES ace.revisions(project_id, source, session_id, revision)
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS ace.evaluations (
    evaluation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL,
    source text NOT NULL,
    session_id text NOT NULL,
    revision text NOT NULL,
    evaluation jsonb NOT NULL,
    score numeric(8,4),
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (project_id, source, session_id, revision)
        REFERENCES ace.revisions(project_id, source, session_id, revision)
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS ace.knowledge_versions (
    knowledge_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES ace.projects(id) ON DELETE RESTRICT,
    version integer NOT NULL CHECK (version >= 1),
    snapshot jsonb NOT NULL,
    checksum text NOT NULL CHECK (checksum ~ '^[0-9a-f]{64}$'),
    published_at timestamptz NOT NULL DEFAULT now(),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (project_id, version)
);

CREATE INDEX IF NOT EXISTS knowledge_versions_latest_idx
    ON ace.knowledge_versions (project_id, version DESC);

-- The native role IDs and the project UUID are the stable mapped keys.  The
-- source/session/revision tuple remains present on every consumer-facing row.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ace_ingest') THEN
        EXECUTE 'CREATE ROLE ace_ingest NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ace_processor') THEN
        EXECUTE 'CREATE ROLE ace_processor NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ace_reader') THEN
        EXECUTE 'CREATE ROLE ace_reader NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT';
    END IF;
END
$$;

REVOKE ALL ON SCHEMA ace FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA ace FROM PUBLIC;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA ace FROM PUBLIC;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA ace FROM PUBLIC;
GRANT USAGE ON SCHEMA ace TO ace_ingest, ace_processor, ace_reader;

-- Explicit project registration is the only path that enables ingestion.
CREATE OR REPLACE FUNCTION ace.register_project(
    p_project_id uuid,
    p_name text,
    p_root text,
    p_vault_dir text,
    p_enabled boolean DEFAULT true,
    p_initialized boolean DEFAULT true
)
RETURNS ace.projects
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ace, pg_temp
AS $$
DECLARE
    v_project ace.projects;
BEGIN
    IF p_project_id IS NULL
       OR p_name IS NULL OR char_length(p_name) NOT BETWEEN 1 AND 200
       OR p_root IS NULL OR char_length(p_root) NOT BETWEEN 1 AND 2048
       OR p_vault_dir IS NULL OR char_length(p_vault_dir) NOT BETWEEN 1 AND 2048 THEN
        RAISE EXCEPTION 'invalid ACE project descriptor' USING ERRCODE = '22023';
    END IF;

    INSERT INTO projects (id, name, root, vault_dir, enabled, initialized, updated_at)
    VALUES (p_project_id, p_name, p_root, p_vault_dir,
            COALESCE(p_enabled, false), COALESCE(p_initialized, false), now())
    ON CONFLICT (id) DO UPDATE
    SET name = EXCLUDED.name,
        root = EXCLUDED.root,
        vault_dir = EXCLUDED.vault_dir,
        enabled = EXCLUDED.enabled,
        initialized = EXCLUDED.initialized,
        updated_at = now()
    RETURNING * INTO v_project;

    RETURN v_project;
END
$$;

-- Validate the source-adapter contract without rewriting the envelope.  The
-- Python adapter already performs the canonical cleanup; this second gate
-- protects direct RPC callers and keeps revisions/outbox keys stable.
CREATE OR REPLACE FUNCTION ace.jsonb_is_clean(
    p_value jsonb,
    p_key text DEFAULT NULL
)
RETURNS boolean
LANGUAGE plpgsql
IMMUTABLE
SET search_path = ace, pg_temp
AS $$
DECLARE
    child record;
    key_name text;
    normalized_key text;
    text_value text;
    secret_check_value text;
    decoded bytea;
    binary_hint boolean := false;
BEGIN
    IF p_value IS NULL OR jsonb_typeof(p_value) = 'null' THEN
        RETURN true;
    END IF;

    IF jsonb_typeof(p_value) = 'object' THEN
        FOR child IN SELECT key, value FROM jsonb_each(p_value) LOOP
            key_name := lower(child.key);
            normalized_key := regexp_replace(key_name, '[_-]', '', 'g');
            IF key_name IN ('last_token_usage', 'total_token_usage', 'thread_token_usage',
                            'turn_token_usage', 'latest_token_usage_record',
                            'time_to_first_token_ms') THEN
                IF NOT ace.jsonb_is_clean(child.value, child.key) THEN
                    RETURN false;
                END IF;
                CONTINUE;
            END IF;
            IF key_name ~ '(^|[_-])(authorization|proxy-authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password|passwd|secret|token|cookie|set-cookie)([_-]|$)' THEN
                IF jsonb_typeof(child.value) <> 'string'
                   OR child.value #>> '{}' <> '<REDACTED>' THEN
                    RETURN false;
                END IF;
                CONTINUE;
            END IF;
            IF normalized_key IN ('analysis', 'thinking', 'reasoning', 'redactedthinking', 'reasoningcontent') THEN
                RETURN false;
            END IF;
            IF normalized_key IN ('type', 'channel', 'phase')
               AND jsonb_typeof(child.value) = 'string'
               AND lower(child.value #>> '{}') IN ('analysis', 'thinking', 'reasoning', 'redactedthinking', 'redacted_thinking') THEN
                RETURN false;
            END IF;
            IF NOT ace.jsonb_is_clean(child.value, child.key) THEN
                RETURN false;
            END IF;
        END LOOP;
        RETURN true;
    END IF;

    IF jsonb_typeof(p_value) = 'array' THEN
        FOR child IN SELECT value FROM jsonb_array_elements(p_value) LOOP
            IF NOT ace.jsonb_is_clean(child.value, p_key) THEN
                RETURN false;
            END IF;
        END LOOP;
        RETURN true;
    END IF;

    IF jsonb_typeof(p_value) = 'string' THEN
        text_value := p_value #>> '{}';
        -- A canonical adapter may retain a redacted credential marker inside
        -- a code block.  Remove the complete label/assignment/marker pair
        -- before testing for a live value; removing the marker alone turns
        -- ``token=<REDACTED>\nnext`` into a false positive ``token=next``.
        secret_check_value := regexp_replace(
            text_value,
            '(^|[^[:alnum:]_])["'']?(authorization|proxy-authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password|passwd|secret|token|cookie|set-cookie)["'']?[[:space:]]*[:=][[:space:]]*(<REDACTED>|''<REDACTED>''|"<REDACTED>")([^[:alnum:]_]|$)',
            '',
            'gi'
        );
        -- The source adapter also redacts CLI credentials such as
        --access-token <REDACTED>.  Treat that explicit marker as clean while
        -- still rejecting an unredacted value after the flag.
        secret_check_value := regexp_replace(
            secret_check_value,
            '(^|[^[:alnum:]_])--(authorization|proxy-authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password|passwd|secret|token|cookie|set-cookie)[[:space:]]+(<REDACTED>|''<REDACTED>''|"<REDACTED>")([^[:alnum:]_]|$)',
            '',
            'gi'
        );
        secret_check_value := regexp_replace(
            secret_check_value,
            '(<REDACTED>|''<REDACTED>''|"<REDACTED>")',
            '',
            'gi'
        );
        IF secret_check_value ~ '-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----'
           OR secret_check_value ~* '(^|[^[:alnum:]_])["'']?(authorization|proxy-authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password|passwd|secret|token|cookie|set-cookie)["'']?[[:space:]]*[:=][[:space:]]*[^[:space:],;&]+'
           OR secret_check_value ~* '--(authorization|proxy-authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password|passwd|secret|token|cookie|set-cookie)[[:space:]]+[^[:space:]]+'
           OR secret_check_value ~* '(^|[^[:alnum:]_])(sk-(live-|test-|proj-)?[A-Za-z0-9_-]{16,}|sbp_[A-Za-z0-9_]{20,}|sb_secret_[A-Za-z0-9_-]{16,}|github_pat_[A-Za-z0-9_]{16,}|gh[pousr]_[A-Za-z0-9]{16,}|xox[baprs]-[A-Za-z0-9-]{16,}|AKIA[A-Z0-9]{16})([^A-Za-z0-9_-]|$)' THEN
            RETURN false;
        END IF;
        IF text_value ~* '^data:[^,;]*;base64,' THEN
            RETURN false;
        END IF;

        normalized_key := regexp_replace(lower(coalesce(p_key, '')), '[_-]', '', 'g');
        binary_hint := normalized_key IN ('data', 'base64', 'bytes', 'binary', 'blob', 'image', 'imagedata', 'rawimage');
        IF char_length(text_value) >= (CASE WHEN binary_hint THEN 16 ELSE 64 END)
           AND char_length(text_value) % 4 = 0
           AND text_value ~ '^[A-Za-z0-9+/]+={0,2}$' THEN
            BEGIN
                decoded := decode(text_value, 'base64');
            EXCEPTION WHEN others THEN
                decoded := NULL;
            END;
            IF decoded IS NOT NULL
               AND octet_length(decoded) >= (CASE WHEN binary_hint THEN 8 ELSE 32 END) THEN
                RETURN false;
            END IF;
        END IF;
    END IF;
    RETURN true;
END
$$;

CREATE OR REPLACE FUNCTION ace.ingest_snapshot(p_envelope jsonb)
RETURNS TABLE (
    project_id uuid,
    source text,
    session_id text,
    revision text,
    inserted boolean,
    message_count integer,
    attachment_count integer,
    status text
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ace, pg_temp
AS $$
DECLARE
    v_project ace.projects;
    v_project_id uuid;
    v_source text;
    v_session_id text;
    v_revision text;
    v_source_path text;
    v_host_id text;
    v_started_at timestamptz;
    v_updated_at timestamptz;
    v_messages jsonb;
    v_attachments jsonb;
    v_normalized_messages jsonb := '[]'::jsonb;
    v_normalized_attachments jsonb := '[]'::jsonb;
    v_message jsonb;
    v_attachment jsonb;
    v_revision_row ace.revisions;
    v_snapshot jsonb;
    v_message_id text;
    v_attachment_id text;
    v_role text;
    v_type text;
    v_sha256 text;
    v_name text;
    v_mime_type text;
    v_kind text;
    v_uri text;
    v_call_id text;
    v_message_status text;
    v_model text;
    v_ordinal integer;
    v_size_bytes bigint;
    v_inserted boolean := false;
    v_message_count integer := 0;
    v_attachment_count integer := 0;
BEGIN
    IF p_envelope IS NULL OR jsonb_typeof(p_envelope) <> 'object'
       OR octet_length(p_envelope::text) > 8388608 THEN
        RAISE EXCEPTION 'invalid ACE snapshot envelope' USING ERRCODE = '22023';
    END IF;
    IF p_envelope->>'schema_version' IS DISTINCT FROM '1' THEN
        RAISE EXCEPTION 'unsupported ACE snapshot schema' USING ERRCODE = '22023';
    END IF;

    BEGIN
        v_project_id := (p_envelope #>> '{project,id}')::uuid;
    EXCEPTION WHEN invalid_text_representation THEN
        RAISE EXCEPTION 'invalid ACE project id' USING ERRCODE = '22023';
    END;
    v_source := lower(trim(p_envelope->>'source'));
    v_session_id := p_envelope->>'session_id';
    v_revision := lower(trim(p_envelope->>'revision'));
    v_source_path := p_envelope->>'source_path';
    v_host_id := p_envelope->>'host_id';

    IF v_source IS NULL OR v_source !~ '^[a-z0-9][a-z0-9_.:-]{0,63}$'
       OR v_session_id IS NULL OR char_length(v_session_id) NOT BETWEEN 1 AND 512
       OR v_revision IS NULL OR v_revision !~ '^[0-9a-f]{64}$'
       OR (v_source_path IS NOT NULL AND char_length(v_source_path) > 2048)
       OR (v_host_id IS NOT NULL AND char_length(v_host_id) > 256) THEN
        RAISE EXCEPTION 'invalid ACE snapshot identity' USING ERRCODE = '22023';
    END IF;

    SELECT * INTO v_project
    FROM projects
    WHERE projects.id = v_project_id AND projects.enabled AND projects.initialized;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'ACE project is not registered or enabled' USING ERRCODE = '42501';
    END IF;

    BEGIN
        v_started_at := NULLIF(p_envelope->>'started_at', '')::timestamptz;
        v_updated_at := NULLIF(p_envelope->>'updated_at', '')::timestamptz;
    EXCEPTION WHEN datetime_field_overflow OR invalid_datetime_format THEN
        RAISE EXCEPTION 'invalid ACE snapshot timestamp' USING ERRCODE = '22023';
    END;

    v_messages := COALESCE(p_envelope->'messages', '[]'::jsonb);
    v_attachments := COALESCE(p_envelope->'attachments', '[]'::jsonb);
    IF jsonb_typeof(v_messages) <> 'array' OR jsonb_array_length(v_messages) > 10000
       OR jsonb_typeof(v_attachments) <> 'array' OR jsonb_array_length(v_attachments) > 1000 THEN
        RAISE EXCEPTION 'ACE snapshot collection bounds exceeded' USING ERRCODE = '22023';
    END IF;

    -- Build a normalized snapshot and validate each entry before the first
    -- write.  Unknown message/attachment fields are never copied to storage.
    FOR v_message IN SELECT value FROM jsonb_array_elements(v_messages) LOOP
        IF jsonb_typeof(v_message) <> 'object' THEN
            RAISE EXCEPTION 'invalid ACE message' USING ERRCODE = '22023';
        END IF;
        IF NOT (v_message ? 'content') THEN
            RAISE EXCEPTION 'invalid ACE message content' USING ERRCODE = '22023';
        END IF;
        IF NOT ace.jsonb_is_clean(COALESCE(v_message->'content', 'null'::jsonb))
           OR NOT ace.jsonb_is_clean(COALESCE(v_message->'refs', 'null'::jsonb)) THEN
            RAISE EXCEPTION 'ACE message is not sanitized' USING ERRCODE = '22023';
        END IF;
        v_message_id := v_message->>'id';
        v_ordinal := (v_message->>'ordinal')::integer;
        v_role := v_message->>'role';
        v_type := v_message->>'type';
        IF v_message_id IS NULL OR char_length(v_message_id) NOT BETWEEN 1 AND 512
           OR v_ordinal IS NULL OR v_ordinal NOT BETWEEN 0 AND 1000000
           OR v_role IS NULL OR char_length(v_role) NOT BETWEEN 1 AND 64
           OR v_type IS NULL OR char_length(v_type) NOT BETWEEN 1 AND 64
           OR octet_length(COALESCE((v_message->'content')::text, 'null')) > 1048576
           OR octet_length(COALESCE((v_message->'refs')::text, 'null')) > 262144 THEN
            RAISE EXCEPTION 'invalid ACE message bounds' USING ERRCODE = '22023';
        END IF;
        v_call_id := v_message->>'call_id';
        v_message_status := v_message->>'status';
        v_model := v_message->>'model';
        IF (v_call_id IS NOT NULL AND char_length(v_call_id) > 512)
           OR (v_message_status IS NOT NULL AND char_length(v_message_status) > 128)
           OR (v_model IS NOT NULL AND char_length(v_model) > 256) THEN
            RAISE EXCEPTION 'invalid ACE message metadata' USING ERRCODE = '22023';
        END IF;
        v_normalized_messages := v_normalized_messages || jsonb_build_array(
            jsonb_build_object(
                'id', v_message_id,
                'ordinal', v_ordinal,
                'role', v_role,
                'type', v_type,
                'content', COALESCE(v_message->'content', 'null'::jsonb)
            ) || jsonb_strip_nulls(jsonb_build_object(
                'timestamp', NULLIF(v_message->>'timestamp', ''),
                'call_id', v_call_id,
                'status', v_message_status,
                'model', v_model,
                'refs', v_message->'refs'
            ))
        );
        v_message_count := v_message_count + 1;
    END LOOP;

    FOR v_attachment IN SELECT value FROM jsonb_array_elements(v_attachments) LOOP
        IF jsonb_typeof(v_attachment) <> 'object' THEN
            RAISE EXCEPTION 'invalid ACE attachment' USING ERRCODE = '22023';
        END IF;
        IF jsonb_typeof(COALESCE(v_attachment->'metadata', '{}'::jsonb)) <> 'object' THEN
            RAISE EXCEPTION 'invalid ACE attachment metadata' USING ERRCODE = '22023';
        END IF;
        IF NOT ace.jsonb_is_clean(COALESCE(v_attachment->'metadata', '{}'::jsonb)) THEN
            RAISE EXCEPTION 'ACE attachment metadata is not sanitized' USING ERRCODE = '22023';
        END IF;
        v_attachment_id := v_attachment->>'id';
        v_name := v_attachment->>'name';
        v_mime_type := v_attachment->>'mime_type';
        v_kind := v_attachment->>'kind';
        v_size_bytes := NULLIF(v_attachment->>'size', '')::bigint;
        v_sha256 := lower(NULLIF(v_attachment->>'sha256', ''));
        v_uri := v_attachment->>'uri';
        IF (v_attachment_id IS NOT NULL AND char_length(v_attachment_id) > 512)
           OR (v_name IS NOT NULL AND char_length(v_name) > 512)
           OR (v_mime_type IS NOT NULL AND char_length(v_mime_type) > 256)
           OR (v_kind IS NOT NULL AND char_length(v_kind) > 128)
           OR (v_size_bytes IS NOT NULL AND v_size_bytes NOT BETWEEN 0 AND 1073741824)
           OR (v_sha256 IS NOT NULL AND v_sha256 !~ '^[0-9a-f]{64}$')
           OR (v_uri IS NOT NULL AND char_length(v_uri) > 4096)
           OR octet_length(COALESCE((v_attachment->'metadata')::text, 'null')) > 262144 THEN
            RAISE EXCEPTION 'invalid ACE attachment bounds' USING ERRCODE = '22023';
        END IF;
        v_normalized_attachments := v_normalized_attachments || jsonb_build_array(
            jsonb_strip_nulls(jsonb_build_object(
                'id', v_attachment_id,
                'name', v_name,
                'mime_type', v_mime_type,
                'kind', v_kind,
                'size', v_size_bytes,
                'sha256', v_sha256,
                'uri', v_uri,
                'metadata', COALESCE(v_attachment->'metadata', '{}'::jsonb)
            ))
        );
        v_attachment_count := v_attachment_count + 1;
    END LOOP;

    v_snapshot := jsonb_build_object(
        'schema_version', 1,
        'project', jsonb_build_object(
            'id', v_project.id,
            'name', v_project.name,
            'root', v_project.root,
            'vault_dir', v_project.vault_dir
        ),
        'source', v_source,
        'session_id', v_session_id,
        'revision', v_revision,
        'source_path', v_source_path,
        'host_id', v_host_id,
        'started_at', p_envelope->>'started_at',
        'updated_at', p_envelope->>'updated_at',
        'messages', v_normalized_messages,
        'attachments', v_normalized_attachments
    );

    INSERT INTO sessions (project_id, source, session_id, started_at, updated_at, latest_revision)
    VALUES (v_project_id, v_source, v_session_id, v_started_at, v_updated_at, v_revision)
    ON CONFLICT ON CONSTRAINT sessions_pkey DO UPDATE
    SET started_at = COALESCE(sessions.started_at, EXCLUDED.started_at),
        updated_at = CASE
            WHEN sessions.updated_at IS NULL THEN EXCLUDED.updated_at
            WHEN EXCLUDED.updated_at IS NULL THEN sessions.updated_at
            ELSE GREATEST(sessions.updated_at, EXCLUDED.updated_at)
        END,
        latest_revision = CASE
            WHEN sessions.updated_at IS NULL THEN EXCLUDED.latest_revision
            WHEN EXCLUDED.updated_at IS NULL THEN sessions.latest_revision
            WHEN EXCLUDED.updated_at >= sessions.updated_at THEN EXCLUDED.latest_revision
            ELSE sessions.latest_revision
        END;

    INSERT INTO revisions (
        project_id, source, session_id, revision, source_path, host_id,
        started_at, updated_at, snapshot
    )
    VALUES (
        v_project_id, v_source, v_session_id, v_revision, v_source_path, v_host_id,
        v_started_at, v_updated_at, v_snapshot
    )
    ON CONFLICT ON CONSTRAINT revisions_pkey DO NOTHING
    RETURNING * INTO v_revision_row;

    IF FOUND THEN
        v_inserted := true;
        INSERT INTO messages (
            project_id, source, session_id, revision, message_id, ordinal,
            role, message_type, message_timestamp, content, call_id, status, model, refs
        )
        SELECT v_project_id, v_source, v_session_id, v_revision,
               item->>'id', (item->>'ordinal')::integer, item->>'role', item->>'type',
               NULLIF(item->>'timestamp', '')::timestamptz,
               COALESCE(item->'content', 'null'::jsonb), item->>'call_id',
               item->>'status', item->>'model', item->'refs'
        FROM jsonb_array_elements(v_normalized_messages) AS items(item);

        INSERT INTO attachments (
            project_id, source, session_id, revision, attachment_id, name,
            mime_type, kind, size_bytes, sha256, uri, metadata
        )
        SELECT v_project_id, v_source, v_session_id, v_revision,
               item->>'id', item->>'name', item->>'mime_type', item->>'kind',
               NULLIF(item->>'size', '')::bigint, item->>'sha256', item->>'uri',
               COALESCE(item->'metadata', '{}'::jsonb)
        FROM jsonb_array_elements(v_normalized_attachments) AS items(item);

        INSERT INTO processing_runs (
            project_id, source, session_id, revision, stage, status
        )
        VALUES (v_project_id, v_source, v_session_id, v_revision, 'extraction', 'pending')
        ON CONFLICT ON CONSTRAINT processing_runs_project_id_source_session_id_revision_stage_key DO NOTHING;
    ELSE
        SELECT * INTO v_revision_row
        FROM revisions
        WHERE revisions.project_id = v_project_id AND revisions.source = v_source
          AND revisions.session_id = v_session_id AND revisions.revision = v_revision;
    END IF;

    RETURN QUERY SELECT v_project_id, v_source, v_session_id, v_revision,
                        v_inserted, v_message_count, v_attachment_count, 'accepted'::text;
END
$$;

CREATE OR REPLACE FUNCTION ace.pending_snapshots(
    p_limit integer,
    p_stage text,
    p_project_id uuid
)
RETURNS TABLE (
    envelope jsonb,
    project_id uuid,
    source text,
    session_id text,
    revision text
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ace, pg_temp
AS $$
BEGIN
    IF p_limit IS NULL OR p_limit NOT BETWEEN 1 AND 500
       OR p_stage IS NULL OR p_stage NOT IN ('extraction', 'analysis', 'result', 'decision', 'evaluation', 'compile') THEN
        RAISE EXCEPTION 'invalid ACE pending-snapshot bounds' USING ERRCODE = '22023';
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

CREATE OR REPLACE FUNCTION ace.pending_snapshots(
    p_limit integer DEFAULT 100,
    p_stage text DEFAULT 'extraction'
)
RETURNS TABLE (envelope jsonb, project_id uuid, source text, session_id text, revision text)
LANGUAGE sql
SECURITY DEFINER
SET search_path = ace, pg_temp
AS $$
    SELECT * FROM ace.pending_snapshots(p_limit, p_stage, NULL::uuid);
$$;

-- The processor normally needs only identities first.  Returning a complete
-- snapshot here would make a long conversation exceed the bounded SQL
-- transport before the local cursor can select its delta.
CREATE OR REPLACE FUNCTION ace.pending_snapshot_refs(
    p_limit integer,
    p_stage text,
    p_project_id uuid
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

CREATE OR REPLACE FUNCTION ace.pending_snapshot_refs(
    p_limit integer DEFAULT 100,
    p_stage text DEFAULT 'extraction'
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
    SELECT * FROM ace.pending_snapshot_refs(p_limit, p_stage, NULL::uuid);
$$;

CREATE OR REPLACE FUNCTION ace.snapshot_delta(
    p_project_id uuid,
    p_source text,
    p_session_id text,
    p_revision text,
    p_last_ordinal integer
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
    IF p_project_id IS NULL
       OR p_source IS NULL OR lower(p_source) !~ '^[a-z0-9][a-z0-9_.:-]{0,63}$'
       OR p_session_id IS NULL OR char_length(p_session_id) NOT BETWEEN 1 AND 512
       OR p_revision IS NULL OR lower(p_revision) !~ '^[0-9a-f]{64}$'
       OR p_last_ordinal IS NULL OR p_last_ordinal < -1 THEN
        RAISE EXCEPTION 'invalid ACE snapshot delta bounds' USING ERRCODE = '22023';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM projects p
        WHERE p.id = p_project_id AND p.enabled AND p.initialized
    ) THEN
        RETURN;
    END IF;

    RETURN QUERY
    SELECT jsonb_set(
               jsonb_set(
                   r.snapshot,
                   '{messages}',
                   COALESCE(
                       (
                           SELECT jsonb_agg(item ORDER BY (item->>'ordinal')::integer)
                           FROM jsonb_array_elements(COALESCE(r.snapshot->'messages', '[]'::jsonb)) AS items(item)
                           WHERE (item->>'ordinal')::integer > p_last_ordinal
                       ),
                       '[]'::jsonb
                   ),
                   true
               ),
               '{attachments}', '[]'::jsonb, true
           ),
           r.project_id, r.source, r.session_id, r.revision
    FROM revisions r
    WHERE r.project_id = p_project_id
      AND r.source = lower(p_source)
      AND r.session_id = p_session_id
      AND r.revision = lower(p_revision);
END
$$;

CREATE OR REPLACE FUNCTION ace.mark_processed(
    p_source text,
    p_session_id text,
    p_revision text,
    p_project_id uuid,
    p_stage text,
    p_status text,
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
    v_attempted_at timestamptz;
BEGIN
    IF p_source IS NULL OR lower(p_source) !~ '^[a-z0-9][a-z0-9_.:-]{0,63}$'
       OR p_session_id IS NULL OR char_length(p_session_id) NOT BETWEEN 1 AND 512
       OR p_revision IS NULL OR lower(p_revision) !~ '^[0-9a-f]{64}$'
       OR p_stage IS NULL OR p_stage NOT IN ('extraction', 'analysis', 'result', 'decision', 'evaluation', 'compile')
       OR p_status IS NULL OR p_status NOT IN ('pending', 'running', 'succeeded', 'failed', 'skipped')
       OR (p_error IS NOT NULL AND char_length(p_error) > 256) THEN
        RAISE EXCEPTION 'invalid ACE processing status' USING ERRCODE = '22023';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM revisions
        WHERE revisions.project_id = p_project_id AND revisions.source = lower(p_source)
          AND revisions.session_id = p_session_id AND revisions.revision = lower(p_revision)
    ) THEN
        RAISE EXCEPTION 'ACE revision is not registered' USING ERRCODE = '22023';
    END IF;

    v_attempted_at := clock_timestamp();

    INSERT INTO processing_runs (
        project_id, source, session_id, revision, stage, status,
        error_type, started_at, finished_at
    )
    VALUES (
        p_project_id, lower(p_source), p_session_id, lower(p_revision), p_stage, p_status,
        NULLIF(p_error, ''), v_attempted_at,
        CASE WHEN p_status IN ('succeeded', 'failed', 'skipped') THEN v_attempted_at ELSE NULL END
    )
    ON CONFLICT ON CONSTRAINT processing_runs_project_id_source_session_id_revision_stage_key DO UPDATE
    SET status = EXCLUDED.status,
        error_type = EXCLUDED.error_type,
        started_at = EXCLUDED.started_at,
        finished_at = EXCLUDED.finished_at;

    SELECT * INTO v_run
    FROM processing_runs
    WHERE processing_runs.project_id = p_project_id AND processing_runs.source = lower(p_source)
      AND processing_runs.session_id = p_session_id AND processing_runs.revision = lower(p_revision)
      AND processing_runs.stage = p_stage;
    RETURN QUERY SELECT v_run.run_id, v_run.project_id, v_run.source, v_run.session_id,
                        v_run.revision, v_run.stage, v_run.status, v_run.error_type;
END
$$;

CREATE OR REPLACE FUNCTION ace.list_projects()
RETURNS TABLE (
    id uuid,
    name text,
    root text,
    vault_dir text,
    enabled boolean
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ace, pg_temp
AS $$
    SELECT p.id, p.name, p.root, p.vault_dir, p.enabled
    FROM projects p
    ORDER BY p.name, p.id;
$$;

CREATE OR REPLACE FUNCTION ace.search_history(
    p_project_id uuid,
    p_query text,
    p_limit integer DEFAULT 50
)
RETURNS TABLE (
    kind text,
    item_id uuid,
    created_at timestamptz,
    payload jsonb
)
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = ace, pg_temp
AS $$
BEGIN
    IF p_project_id IS NULL OR p_query IS NULL OR char_length(p_query) > 200
       OR p_limit IS NULL OR p_limit NOT BETWEEN 1 AND 100 THEN
        RAISE EXCEPTION 'invalid ACE history search bounds' USING ERRCODE = '22023';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM projects p
        WHERE p.id = p_project_id AND p.enabled AND p.initialized
    ) THEN
        RETURN;
    END IF;

    RETURN QUERY
    SELECT x.kind, x.item_id, x.created_at, x.payload
    FROM (
        SELECT 'message'::text AS kind, m.message_pk AS item_id,
               COALESCE(m.message_timestamp, m.created_at) AS created_at,
               jsonb_build_object('source', m.source, 'session_id', m.session_id,
                                  'revision', m.revision, 'message_id', m.message_id,
                                  'ordinal', m.ordinal, 'role', m.role,
                                  'type', m.message_type, 'refs', m.refs,
                                  'content_excerpt', left(m.content::text, 4000)) AS payload
        FROM messages m
        JOIN sessions s ON s.project_id = m.project_id AND s.source = m.source
                       AND s.session_id = m.session_id AND s.latest_revision = m.revision
        WHERE m.project_id = p_project_id AND m.content::text ILIKE '%' || p_query || '%'
        UNION ALL
        SELECT 'observation'::text AS kind, o.observation_id AS item_id,
               o.created_at,
               jsonb_build_object('problem_signature', o.problem_signature,
                                  'success', o.success,
                                  'preference_evidence', o.preference_evidence,
                                  'evidence', o.evidence) AS payload
        FROM observations o
        WHERE o.project_id = p_project_id
          AND (o.problem_signature ILIKE '%' || p_query || '%'
               OR o.evidence::text ILIKE '%' || p_query || '%'
               OR o.preference_evidence::text ILIKE '%' || p_query || '%')
        UNION ALL
        SELECT 'recommendation', r.recommendation_id, r.created_at,
               jsonb_build_object('recommendation', r.recommendation,
                                  'rationale', r.rationale, 'evidence', r.evidence)
        FROM recommendations r
        WHERE r.project_id = p_project_id
          AND (r.recommendation ILIKE '%' || p_query || '%'
               OR r.rationale::text ILIKE '%' || p_query || '%'
               OR r.evidence::text ILIKE '%' || p_query || '%')
        UNION ALL
        SELECT 'result', r.result_id, r.created_at, r.result
        FROM results r
        WHERE r.project_id = p_project_id AND r.result::text ILIKE '%' || p_query || '%'
        UNION ALL
        SELECT 'decision', d.decision_id, d.created_at,
               jsonb_build_object('decision', d.decision, 'actor', d.actor)
        FROM decisions d
        WHERE d.project_id = p_project_id AND d.decision::text ILIKE '%' || p_query || '%'
        UNION ALL
        SELECT 'correction', c.correction_id, c.created_at,
               jsonb_build_object('correction', c.correction, 'actor', c.actor)
        FROM corrections c
        WHERE c.project_id = p_project_id AND c.correction::text ILIKE '%' || p_query || '%'
        UNION ALL
        SELECT 'evaluation', e.evaluation_id, e.created_at,
               jsonb_build_object('evaluation', e.evaluation, 'score', e.score)
        FROM evaluations e
        WHERE e.project_id = p_project_id AND e.evaluation::text ILIKE '%' || p_query || '%'
    ) AS x
    ORDER BY x.created_at DESC
    LIMIT p_limit;
END
$$;

CREATE OR REPLACE FUNCTION ace.save_analysis(
    p_project_id uuid,
    p_source text,
    p_session_id text,
    p_revision text,
    p_analysis jsonb
)
RETURNS TABLE (observations_saved integer, recommendations_saved integer)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ace, pg_temp
AS $$
DECLARE
    v_item jsonb;
    v_observations jsonb;
    v_recommendations jsonb;
    v_observations_saved integer := 0;
    v_recommendations_saved integer := 0;
    v_problem text;
    v_recommendation text;
BEGIN
    IF p_analysis IS NULL OR octet_length(p_analysis::text) > 2097152
       OR NOT EXISTS (
           SELECT 1 FROM revisions
           WHERE revisions.project_id = p_project_id AND revisions.source = lower(p_source)
             AND revisions.session_id = p_session_id AND revisions.revision = lower(p_revision)
       ) THEN
        RAISE EXCEPTION 'invalid ACE analysis context' USING ERRCODE = '22023';
    END IF;

    -- Serialize retries for one revision without locking other conversations.
    PERFORM 1 FROM revisions r
    WHERE r.project_id = p_project_id AND r.source = lower(p_source)
      AND r.session_id = p_session_id AND r.revision = lower(p_revision)
    FOR UPDATE;

    v_observations := COALESCE(p_analysis->'observations', '[]'::jsonb);
    IF jsonb_typeof(v_observations) <> 'array' OR jsonb_array_length(v_observations) > 1000 THEN
        RAISE EXCEPTION 'invalid ACE observations' USING ERRCODE = '22023';
    END IF;
    FOR v_item IN SELECT value FROM jsonb_array_elements(v_observations) LOOP
        v_problem := COALESCE(v_item->>'problem_signature', v_item->>'problem');
        INSERT INTO observations (
            project_id, source, session_id, revision, problem_signature,
            success, preference_evidence, evidence
        ) SELECT
            p_project_id, lower(p_source), p_session_id, lower(p_revision),
            v_problem,
            CASE WHEN v_item ? 'success' THEN (v_item->>'success')::boolean ELSE NULL END,
            COALESCE(v_item->'preference_evidence', v_item->'preference'),
            COALESCE(v_item->'evidence', v_item)
        WHERE NOT EXISTS (
            SELECT 1 FROM observations o
            WHERE o.project_id = p_project_id AND o.source = lower(p_source)
              AND o.session_id = p_session_id AND o.revision = lower(p_revision)
              AND o.problem_signature IS NOT DISTINCT FROM v_problem
              AND o.success IS NOT DISTINCT FROM
                  CASE WHEN v_item ? 'success' THEN (v_item->>'success')::boolean ELSE NULL END
              AND o.preference_evidence IS NOT DISTINCT FROM
                  COALESCE(v_item->'preference_evidence', v_item->'preference')
              AND o.evidence = COALESCE(v_item->'evidence', v_item)
        );
        IF FOUND THEN
            v_observations_saved := v_observations_saved + 1;
        END IF;
    END LOOP;

    v_recommendations := COALESCE(p_analysis->'recommendations', '[]'::jsonb);
    IF jsonb_typeof(v_recommendations) <> 'array' OR jsonb_array_length(v_recommendations) > 1000 THEN
        RAISE EXCEPTION 'invalid ACE recommendations' USING ERRCODE = '22023';
    END IF;
    FOR v_item IN SELECT value FROM jsonb_array_elements(v_recommendations) LOOP
        v_recommendation := COALESCE(v_item->>'recommendation', v_item->>'text');
        IF v_recommendation IS NULL OR char_length(v_recommendation) NOT BETWEEN 1 AND 4000 THEN
            RAISE EXCEPTION 'invalid ACE recommendation' USING ERRCODE = '22023';
        END IF;
        INSERT INTO recommendations (
            project_id, source, session_id, revision, recommendation, rationale, evidence
        ) SELECT
            p_project_id, lower(p_source), p_session_id, lower(p_revision), v_recommendation,
            COALESCE(v_item->'rationale', '{}'::jsonb),
            COALESCE(v_item->'evidence', v_item)
        WHERE NOT EXISTS (
            SELECT 1 FROM recommendations r
            WHERE r.project_id = p_project_id AND r.source = lower(p_source)
              AND r.session_id = p_session_id AND r.revision = lower(p_revision)
              AND r.recommendation = v_recommendation
              AND r.rationale = COALESCE(v_item->'rationale', '{}'::jsonb)
              AND r.evidence = COALESCE(v_item->'evidence', v_item)
        );
        IF FOUND THEN
            v_recommendations_saved := v_recommendations_saved + 1;
        END IF;
    END LOOP;

    RETURN QUERY SELECT v_observations_saved, v_recommendations_saved;
END
$$;

CREATE OR REPLACE FUNCTION ace.save_result(
    p_project_id uuid,
    p_source text,
    p_session_id text,
    p_revision text,
    p_result jsonb
)
RETURNS TABLE (result_id uuid)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ace, pg_temp
AS $$
DECLARE
    v_id uuid;
BEGIN
    IF p_result IS NULL OR octet_length(p_result::text) > 2097152 THEN
        RAISE EXCEPTION 'invalid ACE result' USING ERRCODE = '22023';
    END IF;
    INSERT INTO results (project_id, source, session_id, revision, result)
    VALUES (p_project_id, lower(p_source), p_session_id, lower(p_revision), p_result)
    RETURNING results.result_id INTO v_id;
    RETURN QUERY SELECT v_id;
END
$$;

CREATE OR REPLACE FUNCTION ace.save_decision(
    p_project_id uuid,
    p_source text,
    p_session_id text,
    p_revision text,
    p_decision jsonb,
    p_actor text DEFAULT NULL
)
RETURNS TABLE (decision_id uuid)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ace, pg_temp
AS $$
DECLARE
    v_id uuid;
BEGIN
    IF p_decision IS NULL OR octet_length(p_decision::text) > 1048576
       OR (p_actor IS NOT NULL AND char_length(p_actor) > 256) THEN
        RAISE EXCEPTION 'invalid ACE decision' USING ERRCODE = '22023';
    END IF;
    INSERT INTO decisions (project_id, source, session_id, revision, decision, actor)
    VALUES (p_project_id, lower(p_source), p_session_id, lower(p_revision), p_decision, p_actor)
    RETURNING decisions.decision_id INTO v_id;
    RETURN QUERY SELECT v_id;
END
$$;

CREATE OR REPLACE FUNCTION ace.save_correction(
    p_project_id uuid,
    p_source text,
    p_session_id text,
    p_revision text,
    p_correction jsonb,
    p_actor text DEFAULT NULL
)
RETURNS TABLE (correction_id uuid)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ace, pg_temp
AS $$
DECLARE
    v_id uuid;
BEGIN
    IF p_correction IS NULL OR octet_length(p_correction::text) > 1048576
       OR (p_actor IS NOT NULL AND char_length(p_actor) > 256) THEN
        RAISE EXCEPTION 'invalid ACE correction' USING ERRCODE = '22023';
    END IF;
    INSERT INTO corrections (project_id, source, session_id, revision, correction, actor)
    VALUES (p_project_id, lower(p_source), p_session_id, lower(p_revision), p_correction, p_actor)
    RETURNING corrections.correction_id INTO v_id;
    RETURN QUERY SELECT v_id;
END
$$;

CREATE OR REPLACE FUNCTION ace.save_evaluation(
    p_project_id uuid,
    p_source text,
    p_session_id text,
    p_revision text,
    p_evaluation jsonb,
    p_score numeric DEFAULT NULL
)
RETURNS TABLE (evaluation_id uuid)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ace, pg_temp
AS $$
DECLARE
    v_id uuid;
BEGIN
    IF p_evaluation IS NULL OR octet_length(p_evaluation::text) > 1048576 THEN
        RAISE EXCEPTION 'invalid ACE evaluation' USING ERRCODE = '22023';
    END IF;
    INSERT INTO evaluations (project_id, source, session_id, revision, evaluation, score)
    VALUES (p_project_id, lower(p_source), p_session_id, lower(p_revision), p_evaluation, p_score)
    RETURNING evaluations.evaluation_id INTO v_id;
    RETURN QUERY SELECT v_id;
END
$$;

CREATE OR REPLACE FUNCTION ace.publish_compiled_snapshot(
    p_project_id uuid,
    p_version integer,
    p_snapshot jsonb,
    p_checksum text
)
RETURNS TABLE (project_id uuid, version integer, checksum text, published_at timestamptz)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ace, pg_temp
AS $$
DECLARE
    v_row knowledge_versions;
BEGIN
    IF p_project_id IS NULL OR p_version IS NULL OR p_version < 1
       OR p_snapshot IS NULL OR octet_length(p_snapshot::text) > 8388608
       OR p_checksum IS NULL OR lower(p_checksum) !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION 'invalid ACE compiled snapshot' USING ERRCODE = '22023';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM projects p
        WHERE p.id = p_project_id AND p.enabled AND p.initialized
    ) THEN
        RAISE EXCEPTION 'ACE project is not registered or enabled' USING ERRCODE = '42501';
    END IF;
    INSERT INTO knowledge_versions (project_id, version, snapshot, checksum)
    VALUES (p_project_id, p_version, p_snapshot, lower(p_checksum))
    ON CONFLICT ON CONSTRAINT knowledge_versions_project_id_version_key DO NOTHING
    RETURNING * INTO v_row;
    IF NOT FOUND THEN
        SELECT * INTO v_row FROM knowledge_versions
        WHERE knowledge_versions.project_id = p_project_id
          AND knowledge_versions.version = p_version;
        IF v_row.checksum <> lower(p_checksum) THEN
            RAISE EXCEPTION 'ACE compiled snapshot version already exists' USING ERRCODE = '23505';
        END IF;
    END IF;
    RETURN QUERY SELECT v_row.project_id, v_row.version, v_row.checksum, v_row.published_at;
END
$$;

CREATE OR REPLACE FUNCTION ace.read_compiled_snapshot(
    p_project_id uuid,
    p_version integer DEFAULT NULL
)
RETURNS TABLE (version integer, snapshot jsonb, checksum text, published_at timestamptz)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ace, pg_temp
AS $$
    SELECT k.version, k.snapshot, k.checksum, k.published_at
    FROM knowledge_versions k
    WHERE k.project_id = p_project_id
      AND EXISTS (
          SELECT 1 FROM projects p
          WHERE p.id = p_project_id AND p.enabled AND p.initialized
      )
      AND (p_version IS NULL OR k.version = p_version)
    ORDER BY k.version DESC
    LIMIT 1;
$$;

-- RLS remains enabled on every table.  No general table privileges or PUBLIC
-- policies are granted; callers use the small function surface above.  The
-- reader gets only the two read APIs and the processor gets only processing
-- and persistence APIs.
DO $$
DECLARE
    v_table text;
BEGIN
    FOREACH v_table IN ARRAY ARRAY[
        'schema_migrations', 'projects', 'sessions', 'revisions', 'messages',
        'attachments', 'processing_runs', 'observations', 'recommendations',
        'results', 'decisions', 'corrections', 'evaluations', 'knowledge_versions'
    ] LOOP
        EXECUTE format('ALTER TABLE ace.%I ENABLE ROW LEVEL SECURITY', v_table);
        EXECUTE format('REVOKE ALL ON ace.%I FROM PUBLIC', v_table);
    END LOOP;
END
$$;

-- PostgreSQL grants EXECUTE on newly-created functions to PUBLIC by default;
-- revoke that default after all function definitions exist, then add only the
-- dedicated role grants below.
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA ace FROM PUBLIC;

GRANT EXECUTE ON FUNCTION ace.register_project(uuid, text, text, text, boolean, boolean) TO ace_ingest;
GRANT EXECUTE ON FUNCTION ace.ingest_snapshot(jsonb) TO ace_ingest;
GRANT EXECUTE ON FUNCTION ace.list_projects() TO ace_ingest, ace_processor, ace_reader;
GRANT EXECUTE ON FUNCTION ace.pending_snapshots(integer, text) TO ace_processor;
GRANT EXECUTE ON FUNCTION ace.pending_snapshots(integer, text, uuid) TO ace_processor;
GRANT EXECUTE ON FUNCTION ace.pending_snapshot_refs(integer, text) TO ace_processor;
GRANT EXECUTE ON FUNCTION ace.pending_snapshot_refs(integer, text, uuid) TO ace_processor;
GRANT EXECUTE ON FUNCTION ace.snapshot_delta(uuid, text, text, text, integer) TO ace_processor;
GRANT EXECUTE ON FUNCTION ace.mark_processed(text, text, text, uuid, text, text, text) TO ace_processor;
GRANT EXECUTE ON FUNCTION ace.save_analysis(uuid, text, text, text, jsonb) TO ace_processor;
GRANT EXECUTE ON FUNCTION ace.save_result(uuid, text, text, text, jsonb) TO ace_processor;
GRANT EXECUTE ON FUNCTION ace.save_decision(uuid, text, text, text, jsonb, text) TO ace_processor;
GRANT EXECUTE ON FUNCTION ace.save_correction(uuid, text, text, text, jsonb, text) TO ace_processor;
GRANT EXECUTE ON FUNCTION ace.save_evaluation(uuid, text, text, text, jsonb, numeric) TO ace_processor;
GRANT EXECUTE ON FUNCTION ace.publish_compiled_snapshot(uuid, integer, jsonb, text) TO ace_processor;
GRANT EXECUTE ON FUNCTION ace.search_history(uuid, text, integer) TO ace_processor, ace_reader;
GRANT EXECUTE ON FUNCTION ace.read_compiled_snapshot(uuid, integer) TO ace_processor, ace_reader;

-- The configured wrapper uses the migration owner connection. Allow that
-- principal to assume the narrow ACE roles for individual transactions.
DO $$ BEGIN
    EXECUTE format('GRANT ace_ingest, ace_processor, ace_reader TO %I', current_user);
END $$;

INSERT INTO ace.schema_migrations (version, description)
VALUES (1, 'ACE Supabase schema and transport foundation')
ON CONFLICT (version) DO NOTHING;

COMMIT;
