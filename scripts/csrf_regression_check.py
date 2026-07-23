import re

from derivapro import create_app
from derivapro.models.db_models import User


def extract_csrf_token(html: str) -> str | None:
    patterns = [
        r'name="csrf_token"\s+value="([^"]+)"',
        r"name='csrf_token'\s+value='([^']+)'",
        r'value="([^"]+)"\s+name="csrf_token"',
        r"value='([^']+)'\s+name='csrf_token'",
    ]
    for p in patterns:
        m = re.search(p, html)
        if m:
            return m.group(1)
    return None


def main() -> int:
    app = create_app()
    with app.app_context():
        client = app.test_client()

        user = User.query.first()
        if user:
            with client.session_transaction() as sess:
                sess["_user_id"] = str(user.id)
                sess["_fresh"] = True

        get_resp = client.get("/volatility_surface/volatility_surface")
        print(f"GET /volatility_surface/volatility_surface -> {get_resp.status_code}")
        html = get_resp.data.decode("utf-8", errors="ignore")
        token = extract_csrf_token(html)
        if not token:
            print("FAIL: no csrf token found in volatility surface form")
            return 1

        post_ok = client.post(
            "/volatility_surface/volatility_surface",
            data={"symbol": "AAPL", "csrf_token": token},
        )
        print(
            "POST /volatility_surface/volatility_surface with token"
            f" -> {post_ok.status_code}"
        )
        if post_ok.status_code == 400:
            print("FAIL: still returns 400 with csrf token")
            return 1

        post_blocked = client.post(
            "/volatility_surface/volatility_surface",
            data={"symbol": "AAPL"},
        )
        print(
            "POST /volatility_surface/volatility_surface without token"
            f" -> {post_blocked.status_code}"
        )
        if post_blocked.status_code != 400:
            print("FAIL: csrf protection not enforced for missing token")
            return 1

        prepay_get = client.get("/prepayment/prepayment-probability-calculator")
        print(
            "GET /prepayment/prepayment-probability-calculator"
            f" -> {prepay_get.status_code}"
        )
        prepay_html = prepay_get.data.decode("utf-8", errors="ignore")
        if "csrf_token" not in prepay_html:
            print("FAIL: prepayment form does not render csrf token")
            return 1

        print("PASS: CSRF checks passed")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
