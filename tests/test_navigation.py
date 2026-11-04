import re
from pathlib import Path


def _nav_link(html: str, label: str) -> str:
    match = re.search(
        rf'<a\b[^>]*>\s*{re.escape(label)}\s*</a>',
        html,
    )
    assert match is not None, f"Navigation link not found: {label}"
    return match.group(0)


def test_overview_is_the_only_current_platform_link(authenticated_client):
    response = authenticated_client.get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    overview = _nav_link(html, "Overview")
    featured = _nav_link(html, "Featured Workflows")

    assert 'class="nav-link selected"' in overview
    assert 'aria-current="page"' in overview
    assert "selected" not in featured
    assert "aria-current" not in featured
    assert 'data-nav-page="false"' in featured
    assert "nav-callout" not in featured


def test_featured_workflows_is_excluded_from_route_matching():
    template = Path("derivapro/templates/base.html").read_text(encoding="utf-8")

    assert "link.dataset.navPage !== 'false'" in template
    assert "&& !link.hash" in template
    assert "var currentLink = navLinks.find" in template
