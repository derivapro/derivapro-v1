import re


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
    assert 'href="/featured-workflows"' in featured
    assert "nav-callout" not in featured
    assert "Representative end-to-end valuation workflows" not in html
    assert "Callable Amortizing Bond" not in html
    assert "Puttable Bond" not in html
    assert "Product-level pricing" not in html
    assert "Operating Model" not in html
    assert "Use Limitation" not in html


def test_featured_workflows_has_a_separate_page(authenticated_client):
    response = authenticated_client.get("/featured-workflows")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    overview = _nav_link(html, "Overview")
    featured = _nav_link(html, "Featured Workflows")

    assert "selected" not in overview
    assert "aria-current" not in overview
    assert 'class="nav-link selected"' in featured
    assert 'aria-current="page"' in featured
    assert "Representative end-to-end valuation workflows" in html
    assert "Callable Amortizing Bond" in html
    assert "Puttable Bond" in html
    assert "Product-level pricing" in html
    assert "Operating Model" in html
    assert "Use Limitation" in html


def test_equity_navigation_has_vanilla_and_non_vanilla_subcategories(
    authenticated_client,
):
    response = authenticated_client.get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    equity_group = re.search(
        r'<details class="nav-group" data-nav-key="equity-options">(.*?)</details>',
        html,
        re.DOTALL,
    )
    assert equity_group is not None
    markup = equity_group.group(1)

    assert 'href="/vanilla-options/"' in markup
    assert ">Vanilla Options</a>" in markup
    assert 'href="/exotic-options/"' in markup
    assert ">Non-Vanilla Options</a>" in markup
    assert markup.index("Vanilla Options") < markup.index("European Options")
    assert markup.index("European Options") < markup.index("American Options")
    assert markup.index("Non-Vanilla Options") < markup.index("Barrier Options")
    assert markup.index("Barrier Options") < markup.index("Asian Options")
    assert 'data-nav-page="false">Free Preview' in html


def test_european_page_prefers_the_vanilla_subcategory_link(
    authenticated_client,
):
    response = authenticated_client.get("/vanilla-options/european")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    preview = _nav_link(html, "Free Preview: European Option")
    european = _nav_link(html, "European Options")

    assert 'data-nav-page="false"' in preview
    assert "selected" not in preview
    assert 'class="nav-link selected"' in european
    assert 'aria-current="page"' in european


def test_vanilla_options_workspace_links_both_exercise_styles(
    authenticated_client,
):
    response = authenticated_client.get("/vanilla-options/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)

    assert "Vanilla Options Workspace" in html
    assert 'href="/vanilla-options/european"' in html
    assert 'href="/vanilla-options/american"' in html
    assert "Pricing Methodologies" in html
    assert "Workspace Capabilities" in html


def test_non_vanilla_workspace_uses_navigation_category_name(
    authenticated_client,
):
    response = authenticated_client.get("/exotic-options/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)

    assert "Non-Vanilla Options" in html
    assert "Barrier Options" in html
    assert "Asian Options" in html










