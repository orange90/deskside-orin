"""Literal, read-only search over locally captured ideas."""
import base64
import hashlib
import math
import struct
import time
from datetime import datetime, timezone

import store


# fromtimestamp converts through UTC before applying the Beijing zone. Keep both
# UTC and the resulting local calendar date in Python's representable range.
MIN_IDEA_AT = datetime.min.replace(tzinfo=timezone.utc).timestamp()
END_IDEA_AT = datetime(9999, 12, 31, 23, 59, 59, tzinfo=store.TZ).timestamp() + 1
ASCII_LOWER = str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')
POSITION = struct.Struct('>dq')  # SQLite REAL timestamp and signed 64-bit integer primary key


def beijing_day(at):
    # datetime rounds to microseconds. SQL day boundaries compare the original
    # REAL, so the preceding fraction of a second belongs to the previous day.
    return datetime.fromtimestamp(math.floor(at), store.TZ).date().isoformat()


def _cursor(keyword, at, note_id):
    position = POSITION.pack(at, note_id)
    tag = hashlib.sha256(keyword + b'\0' + position).digest()[:16]
    return base64.urlsafe_b64encode(position + tag).decode('ascii').rstrip('=')


def _position(cursor, keyword):
    try:
        if not isinstance(cursor, str) or len(cursor) != 43:
            raise ValueError()
        raw = base64.b64decode(cursor + '=', altchars=b'-_', validate=True)
        at, note_id = POSITION.unpack(raw[:POSITION.size])
        if not MIN_IDEA_AT <= at < END_IDEA_AT or _cursor(keyword, at, note_id) != cursor:
            raise ValueError()
        return at, note_id
    except (ValueError, struct.error):
        raise ValueError('灵感分页游标无效，请重新搜索') from None


def search_ideas(query, before=None):
    query = query.strip() if isinstance(query, str) else ''
    if not 1 <= len(query) <= 80:
        raise ValueError('灵感检索词需填写 1 至 80 字')
    try:
        # Match SQLite lower()'s ASCII-only case folding when binding the cursor.
        keyword = query.translate(ASCII_LOWER).encode('utf-8')
    except UnicodeEncodeError:
        raise ValueError('灵感检索词含无效字符') from None
    # Match the daily view's strict at < cutoff, including if the local clock
    # moved backwards between pages. No signed SQLite rowid precedes -2**63.
    upper = (min(time.time(), END_IDEA_AT), -(2**63))
    if before is not None:
        upper = min(upper, _position(before, keyword))
    with store.database() as db:
        rows = db.execute(
            "SELECT id,at,body FROM notes WHERE kind='idea' AND at>=? AND (at,id)<(?,?) "
            "AND typeof(at) IN ('integer','real') AND typeof(body)='text' AND instr(lower(body),lower(?))>0 "
            "ORDER BY at DESC,id DESC LIMIT 21", (MIN_IDEA_AT, *upper, query)).fetchall()
    page = rows[:20]
    return {'items': [{'id': str(r['id']), 'at': r['at'], 'body': r['body'],
                       'date': beijing_day(r['at'])} for r in page],
            'next_cursor': _cursor(keyword, page[-1]['at'], page[-1]['id']) if len(rows) > len(page) else None}
