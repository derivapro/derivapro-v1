from derivapro import create_app

app = create_app()
with app.app_context():
    rules = sorted([(r.endpoint, r.rule) for r in app.url_map.iter_rules()])
    for ep, rule in rules:
        print(f"{ep} -> {rule}")
