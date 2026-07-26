from derivapro import create_app
import re

app = create_app()
app.testing = True
with app.app_context():
    with app.test_client() as client:
        resp = client.get("/auth/login")
        print("status", resp.status_code)
        print("set-cookie", resp.headers.get("Set-Cookie"))
        html = resp.data.decode("utf-8", errors="replace")
        print("contains csrf_token text:", "csrf_token" in html)
        print('contains name="csrf_token"', 'name="csrf_token"' in html)
        print("form snippet:")
        match = re.search(
            r'(<form[^>]*method="post"[^>]*>.*?</form>)', html, re.S | re.I
        )
        if match:
            snippet = match.group(1)
            print(snippet[:1200])
        else:
            print("form not found")
        token_match = re.search(
            r'<input[^>]+name="csrf_token"[^>]+value="([^"]+)"', html
        )
        print("token match:", bool(token_match))
        if token_match:
            print("token value length", len(token_match.group(1)))
        print("csrf_token function available:", "csrf_token" in app.jinja_env.globals)
