# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:55:04 2024

"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from derivapro import create_app

app = create_app()

if __name__ == "__main__":
    debug_enabled = app.config.get("DEBUG", False)
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5001"))
    print(f"Starting DerivaPro at http://{host}:{port}", flush=True)
    app.run(
        host=host,
        port=port,
        debug=debug_enabled,
        use_reloader=debug_enabled,
    )
