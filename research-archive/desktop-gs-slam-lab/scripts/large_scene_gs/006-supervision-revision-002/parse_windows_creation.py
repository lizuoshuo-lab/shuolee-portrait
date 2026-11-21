"""CPU Windows/CIM process creation parser, explicit zones and arbitrary ISO fractions."""
from datetime import datetime, timezone
import re


def creation_ms(value):
    text = str(value)
    legacy = re.fullmatch(r'/Date\((-?\d+)(?:[+-]\d{4})?\)/', text)
    if legacy:
        return int(legacy.group(1))
    match = re.fullmatch(r'(\d{4}-\d\d-\d\d)[T ](\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?(Z|[+-]\d\d:\d\d)', text)
    if not match:
        raise ValueError('CIM creation time must be zoned ISO or explicit Microsoft /Date/: ' + text)
    date, clock, fraction, zone = match.groups()
    fraction = '.' + ((fraction + '000000')[:6]) if fraction else ''
    zone = '+00:00' if zone == 'Z' else zone
    instant = datetime.fromisoformat(date + 'T' + clock + fraction + zone).astimezone(timezone.utc)
    delta = instant - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1000 + delta.microseconds // 1000
