"""Camera-free App Platform entrypoint: python -m stationwatch.hub."""

import logging
import os

from .app import create_app

# Store construction does not connect to the database. /healthz remains available
# during an outage; data endpoints report the existing sanitized HTTP 503 response.
app = create_app()


def main():
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535")
    if not os.environ.get("DASHBOARD_TOKEN"):
        logging.getLogger(__name__).warning(
            "DASHBOARD_TOKEN is unset: dashboard data and review actions are unauthenticated"
        )
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,
        proxy_headers=True,
        forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1"),
        access_log=False,
        timeout_graceful_shutdown=3,
    )


if __name__ == "__main__":
    main()
