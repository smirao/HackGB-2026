# DigitalOcean App Platform alert hub

Use the **Python buildpack**, **Ubuntu 22.04**, and the **repository root** as
Source Directory. Keep Build Command at its default so the buildpack runs
`uv sync --locked --no-default-groups` using `pyproject.toml` and `uv.lock`.
Keep `requirements.txt`, `Pipfile`, `poetry.lock`, and `setup.py` absent so only
uv is detected. The main dependencies install only the PostgreSQL hub; camera,
ML, MongoDB, and development dependencies are optional extras.
The root `.python-version` selects Python 3.14 and allows buildpack patch updates.
Keep `runtime.txt` absent: the uv buildpack rejects it even when `.python-version`
is also present.

Set Run Command to the command in the root Procfile:

```sh
python -m stationwatch.hub
```

The Procfile contains this single `web:` process. The buildpack installs packages
with uv into its Python runtime and puts that runtime on `PATH`, so `python`
uses the uv-installed environment. uv itself is only available during the build;
do not change the production command to `uv run` or hard-code `.venv/bin/python`.

Set HTTP Port to **8080**, health check protocol to **HTTP**, and health check
path to **/healthz**. The process binds to `0.0.0.0` and defaults to port 8080.
`stationwatch.hub:app` is also available for ASGI servers. The entrypoint calls
`create_app(pipeline=None, ...)` and imports no camera, OpenCV, NumPy, ML, Gemini,
or MongoDB packages. The older `stationwatch ... hub` CLI still uses MongoDB.

## Runtime environment

These are the application variables read by this entrypoint and its store.
Add them with **RUN_TIME** scope in App Platform (or keep existing injected
database bindings). Use secret values for the token, password and URI.

| Variable | Value / behavior |
| --- | --- |
| `DASHBOARD_TOKEN` | **Set this before public deployment**, using a random secret of at least 32 characters. When absent or empty, API authentication is skipped; startup logs a warning. The module entrypoint does not enforce a minimum length. |
| `PORT` | `8080`; optional because this is the default. Must match the component's HTTP port. |
| `FORWARDED_ALLOW_IPS` | Set to `*` on App Platform so Uvicorn trusts its ingress proxy's HTTPS scheme. This preserves same-origin login and Secure cookies behind TLS termination. Default is `127.0.0.1`; use `*` only behind the managed ingress. |
| `TIMESCALE_SERVICE_URL` | PostgreSQL connection URI (`postgresql://…` or `postgres://…`); fallback connection configuration. This is not an HTTP URL. |
| `PGUSER` | Injected database user; overrides the URI user. |
| `PGDATABASE` | Injected database name; overrides the URI database. |
| `PGHOST` | Injected database hostname; overrides the URI host. |
| `PGPORT` | Injected database port; overrides the URI port. |
| `PGSSLMODE` | Injected TLS mode; use the provider's setting (at least `require` for the managed database). Overrides URI sslmode. |
| `PGPASSWORD` | Injected database password; overrides the URI password. |

The URI or PG fields must supply complete credentials; there is no need to
duplicate injected values manually. Keep all six existing PG bindings and the
Timescale URI if already provided. Psycopg/libpq also honors standard PostgreSQL
environment variables, e.g. `PGSSLROOTCERT` if your provider requires a CA file.
No `GEMINI_API_KEY`, `MONGODB_URI`, JSON config, model files, or camera is required.
For local runs, the app also loads these settings from the repository-root `.env`
file. Exported environment variables take precedence over `.env` values. Run
`.venv/bin/python -m stationwatch.hub` and open http://127.0.0.1:8080.
Python's temporary-directory selection also honors `TMPDIR`, `TEMP`, and `TMP`;
these are optional OS settings and do not need App Platform values.

Generate a dashboard token locally, save it as a secret environment value, and
enter it in the dashboard login dialog:

```sh
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

The dashboard is public HTML; status, event data, evidence, and review actions
require the token when set. `/healthz` always returns `200 {"ok":true}` without
authentication or a database call. Data endpoints return a sanitized 503 when
the database is unreachable or the schema is missing. Liveness does not certify
database readiness. Startup makes no database call.

## PostgreSQL schema and alert source

This checkout previously supported SQLite and MongoDB only. The new PostgreSQL
store reads `stationwatch_alerts` and `stationwatch_evidence`, defined in
[`postgres.sql`](postgres.sql). Apply that script once to the target database
with a database administration tool or `psql` before using data endpoints:

```sh
# Uses the same exported PG* environment variables as the hub.
psql -v ON_ERROR_STOP=1 -f deploy/postgres.sql
# If using only the URI instead:
psql "$TIMESCALE_SERVICE_URL" -v ON_ERROR_STOP=1 -f deploy/postgres.sql
```

The runtime user needs SELECT on both tables and UPDATE on `stationwatch_alerts`.
No Timescale extension or hypertable is required. Alert producers must populate
these tables with the existing event JSON format, plus the event ID and capture
timestamp columns. JPEG/MP4 evidence is optional and cached in a disposable OS
temporary directory; it must remain in the database across container restarts.

**The current Pi agent uploads to MongoDB.** This deployment does not migrate
MongoDB alerts or change that uploader. Connecting a PostgreSQL hub will show
only alerts written to the PostgreSQL tables; connecting the current Pi agent
requires a separate producer/bridge change. If an existing Timescale service
uses another schema, map the queries in `stationwatch/postgres.py` to that schema.

## Local smoke test

```sh
# Use a new directory for a clean install without changing your local .venv.
export UV_PROJECT_ENVIRONMENT="$(mktemp -d)/venv"
uv sync --locked --no-default-groups
PORT=8080 "$UV_PROJECT_ENVIRONMENT/bin/python" -m stationwatch.hub
# In another terminal (no token or database required for liveness):
curl --fail -i http://127.0.0.1:8080/healthz
```

Buildpack setup follows DigitalOcean's [Python buildpack reference](https://docs.digitalocean.com/products/app-platform/reference/buildpacks/python/).
Proxy trust is described in [Uvicorn's settings reference](https://uvicorn.dev/settings/).
