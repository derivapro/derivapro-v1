from derivapro import create_app

app = create_app()
app.testing = True
print("SECRET_KEY set:", bool(app.config.get("SECRET_KEY")))
print("WTF_CSRF_ENABLED:", app.config.get("WTF_CSRF_ENABLED"))
print("csrf extension:", "csrf" in app.extensions)
print("csrf object:", app.extensions.get("csrf"))
print("csrf_token global:", "csrf_token" in app.jinja_env.globals)
print("csrf_token obj:", app.jinja_env.globals.get("csrf_token"))
print("csrf_token callable:", callable(app.jinja_env.globals.get("csrf_token")))

from flask import render_template_string

print("rendered direct token:", render_template_string("{{ csrf_token() }}"))
