from html import unescape
import re

from derivapro import create_app

app = create_app()
app.testing = True
with app.test_client() as client:
    get_resp = client.get("/auth/login")
    print("GET /auth/login status:", get_resp.status_code)
    body = get_resp.data.decode("utf-8", errors="replace")
    token_match = re.search(r'<input[^>]+name="csrf_token"[^>]+value="([^"]+)"', body)
    print("csrf present:", bool(token_match))
    if token_match:
        token = unescape(token_match.group(1))
        print("csrf token length:", len(token))
    else:
        print(body[:1000])
        raise SystemExit(1)

    post_resp = client.post(
        "/auth/login",
        data={"username": "faikha", "password": "admin", "csrf_token": token},
        follow_redirects=True,
    )
    print("POST /auth/login status:", post_resp.status_code)
    print("Final URL:", post_resp.request.path)
    print("Response snippet:")
    print(post_resp.data.decode("utf-8", errors="replace")[:2000])
