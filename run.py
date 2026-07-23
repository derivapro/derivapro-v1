# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:55:04 2024

"""

from derivapro import create_app

app = create_app()

if __name__ == "__main__":
    debug_enabled = app.config.get("DEBUG", False)
    app.run(
        debug=debug_enabled,
        use_reloader=debug_enabled,
    )
