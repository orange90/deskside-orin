"""Personal, literal substitutions offered only during manual review of a voice draft."""
import hashlib
import json
from uuid import UUID

import store


class RuleConflict(Exception):
    pass


def _identifier(value):
    try:
        return str(UUID(str(value)))
    except ValueError:
        raise ValueError("改字规则编号必须是 UUID") from None


def _term(value):
    text = value.strip() if isinstance(value, str) else ''
    if not 1 <= len(text) <= 40 or len(text.splitlines()) != 1:
        raise ValueError("听到的词和改成的词均须为 1 至 40 字的单行文字")
    return text


def list_rules():
    with store.database() as db:
        rows = db.execute("SELECT id,heard,written FROM voice_replacements ORDER BY sequence")
        return {"items": [dict(row) for row in rows]}


def save_rule(id, heard, written):
    request_id = _identifier(id)
    heard, written = _term(heard), _term(written)
    if heard == written:
        raise ValueError("听到的词和改成的词须不同")
    try:
        # An unambiguous pair encoding also handles literal punctuation and NUL.
        pair = json.dumps([heard, written], ensure_ascii=False, separators=(',', ':')).encode("utf-8")
        pair_hash = hashlib.sha256(pair).digest()
    except UnicodeEncodeError:
        raise ValueError("改字规则含无效字符") from None
    with store.database() as db:
        db.execute("BEGIN IMMEDIATE")
        receipt = db.execute(
            "SELECT r.rule_id,r.pair_hash,v.id IS NOT NULL AS active FROM voice_replacement_requests r "
            "LEFT JOIN voice_replacements v ON v.id=r.rule_id WHERE r.request_id=?", (request_id,)).fetchone()
        if receipt:
            if receipt['pair_hash'] != pair_hash:
                raise RuleConflict("该编号已用于其他改字规则")
            return {"id": receipt['rule_id'], "heard": heard, "written": written, "active": bool(receipt['active'])}

        previous = db.execute("SELECT id,written FROM voice_replacements WHERE heard=?", (heard,)).fetchone()
        if previous:
            if previous['written'] != written:
                raise RuleConflict("该听写词已有不同的改法")
            rule_id = previous['id']
        else:
            if db.execute("SELECT count(*) FROM voice_replacements").fetchone()[0] >= 32:
                raise RuleConflict("最多可保存 32 条改字规则")
            rule_id = request_id
            db.execute("INSERT INTO voice_replacements(id,heard,written) VALUES (?,?,?)", (rule_id, heard, written))
        # Receipts survive the active rule. Even a second client's duplicate
        # save must not create it again after deletion or target its successor.
        db.execute("INSERT INTO voice_replacement_requests(request_id,rule_id,pair_hash) VALUES (?,?,?)",
                   (request_id, rule_id, pair_hash))
        return {"id": rule_id, "heard": heard, "written": written, "active": True}


def remove_rule(id):
    rule_id = _identifier(id)
    with store.database() as db:
        db.execute("BEGIN IMMEDIATE")
        receipt = db.execute("SELECT rule_id FROM voice_replacement_requests WHERE request_id=?", (rule_id,)).fetchone()
        if receipt:
            rule_id = receipt['rule_id']
        db.execute("DELETE FROM voice_replacements WHERE id=?", (rule_id,))
        return {"id": rule_id, "active": False}
