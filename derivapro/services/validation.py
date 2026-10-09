"""Shared numeric input validation for product pricing routes.

The product forms enforce bounds client-side via
``static/js/positive-validation.js``, but the browser is not a trust boundary:
a direct POST (curl, DevTools, a stale tab) can still send a negative notional
or a zero volatility straight into QuantLib and produce a raw 500.

These helpers give routes a uniform way to reject such input *before* pricing
and to hand the template a list of plain-English messages, which
``components/product_sections.html::validation_summary`` renders identically on
every page. Message wording matches the existing ``"<X> must be positive."``
convention already used in ``routes/vanilla_options.py`` and
``routes/exotic_options.py``.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional


class ValidationError(ValueError):
    """Raised when user-supplied numeric input violates its declared bound.

    Carries the full list of messages so a route can surface every problem at
    once rather than making the user fix them one refresh at a time.
    """

    def __init__(self, messages: Iterable[str]):
        self.messages = list(messages)
        super().__init__("; ".join(self.messages))


def _humanize(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


def to_float(value: Any, label: str, *, default: Optional[float] = None) -> float:
    """Parse ``value`` as a float, raising :class:`ValidationError` on junk."""
    if value is None or (isinstance(value, str) and not value.strip()):
        if default is not None:
            return float(default)
        raise ValidationError([f"Enter a value for {label}."])
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValidationError([f"Enter a valid number for {label}."])


def to_int(value: Any, label: str, *, default: Optional[int] = None) -> int:
    """Parse ``value`` as an int, raising :class:`ValidationError` on junk."""
    if value is None or (isinstance(value, str) and not value.strip()):
        if default is not None:
            return int(default)
        raise ValidationError([f"Enter a value for {label}."])
    try:
        return int(float(value))
    except (TypeError, ValueError):
        raise ValidationError([f"Enter a whole number for {label}."])


def parse_positive_float(value: Any, label: str, *, allow_zero: bool = False) -> float:
    """Parse a float that must be positive (or non-negative if ``allow_zero``)."""
    number = to_float(value, label)
    if allow_zero:
        if number < 0:
            raise ValidationError([f"Enter a value of 0 or greater for {label}."])
    elif number <= 0:
        raise ValidationError([f"Enter a positive value for {label}."])
    return number


def parse_positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    """Parse an int that must be positive (or non-negative if ``allow_zero``)."""
    number = to_int(value, label)
    if allow_zero:
        if number < 0:
            raise ValidationError([f"Enter a value of 0 or greater for {label}."])
    elif number <= 0:
        raise ValidationError([f"Enter a positive whole number for {label}."])
    return number


def parse_fraction(value: Any, label: str) -> float:
    """Parse a probability-like input constrained to ``[0, 1]``."""
    number = to_float(value, label)
    if not 0.0 <= number <= 1.0:
        raise ValidationError([f"Enter a value between 0 and 1 for {label}."])
    return number


def check_positive(
    values: Mapping[str, Any],
    labels: Optional[Mapping[str, str]] = None,
    *,
    allow_zero: Iterable[str] = (),
) -> list[str]:
    """Validate a mapping of ``field -> value`` and return every failure message.

    Unlike the ``parse_*`` helpers this collects all problems instead of
    stopping at the first, so a form can be corrected in a single pass::

        errors = check_positive(
            {"notional": notional_raw, "coupon_rate": coupon_raw},
            {"notional": "Notional", "coupon_rate": "Coupon rate"},
            allow_zero=["coupon_rate"],
        )
        if errors:
            return render_template(..., validation_errors=errors)
    """
    labels = labels or {}
    zero_ok = set(allow_zero)
    messages: list[str] = []

    for field, raw in values.items():
        label = labels.get(field) or _humanize(field)
        try:
            parse_positive_float(raw, label, allow_zero=field in zero_ok)
        except ValidationError as error:
            messages.extend(error.messages)

    return messages


def check_fractions(
    values: Mapping[str, Any],
    labels: Optional[Mapping[str, str]] = None,
) -> list[str]:
    """Validate a mapping of ``field -> value`` constrained to ``[0, 1]``."""
    labels = labels or {}
    messages: list[str] = []

    for field, raw in values.items():
        label = labels.get(field) or _humanize(field)
        try:
            parse_fraction(raw, label)
        except ValidationError as error:
            messages.extend(error.messages)

    return messages
