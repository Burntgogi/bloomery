#!/usr/bin/env python3
"""Check and post the review bot's output (.github/workflows/review.yml).

    tools/review/post.py check  --out DIR --diff FILE --pr-json FILE [--files FILE] [--base DIR] [--secret-env NAME]...
                                [--usage FILE]...
    tools/review/post.py issues --out DIR --repo OWNER/NAME --pr N
    tools/review/post.py post   --out DIR --repo OWNER/NAME --pr N
    tools/review/post.py --self-test

`check` reads what the reviewer wrote to DIR (review.md, decision.txt, optional comments.json) and rewrites it in
place into the shape `post` sends: the decision follows the body's verdict line, inline comments that do not sit on a
line the diff adds are dropped (GitHub refuses the whole review for one), at most MAX_COMMENTS remain, the body is cut
to MAX_BODY characters, and the footer marker that the next run reads (head sha, verdict, counts) is appended, with a
the slag ledger's version and the workflow run when pr.json carries them (`ledger=`, `run=`: the weekly
learner reads the run's artifact back, tools/review/signals.py), and a
second marker carrying each --usage file's accounting (tools/review/usage.py: turns, cost, context, tool calls) kept
in DIR/usage.txt so a later check without the files keeps it. Issue candidates (issues.json: defects outside the
diff) are kept when well formed, at most MAX_ISSUES, and with --base only when every evidence path exists there. It
refuses (exit 3, nothing written) when the output carries a credential: the value of each --secret-env variable, or a
token-shaped string. A missing or unreadable output is exit 65.

`issues` files each candidate as an issue labelled ISSUE_LABEL, unless an issue with that label and the same title
exists (open or closed), and writes the numbers to DIR/filed.txt.

`check` also decides the review's event (OUT/event.txt; a later check keeps a COMMENT): REQUEST_CHANGES for a
MUST_FIX verdict; APPROVE for a CLEAN one when decision.txt agrees, the validator ran (its line is in review.md), and
no file of the pull request (--files, the `pulls/<n>/files` list, else the diff's paths) is under APPROVE_GUARD, the
paths that change the bot itself, CI or the toolchain and so need a person; else COMMENT, with a line saying why the
CLEAN review did not approve. With no approvals required, a change request does not lock the merge button; it says
"fix this first", and the bot's next review of a new head replaces it.
`post` sends DIR as one pull request review with that event, using
GH_TOKEN, with a line naming the issues `issues` filed. When GitHub refuses the inline comments (422), it posts the
body alone and says so in the body.
"""

import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request

MARKER = "bloomery-review"
# Shingler: the worker who hammers the slag out of a fresh bloom. The name is for people; MARKER stays as it is, since
# the learner finds past reviews by it.
SIGNATURE = ("<sub>🔨 Shingler, bloomery's review bot: approves a clean pull request and requests changes on a must-fix "
             "([how it works](https://github.com/{repo}/blob/main/.github/review/README.md))</sub>")
MAX_BODY = 12000
MAX_COMMENTS = 10
APPROVE_GUARD = (".github/", "tools/review/", "AGENTS.md", "CLAUDE.md", "CONTRIBUTING.md", "CONTRIBUTING.ko.md",
                 "rust-toolchain.toml", "deny.toml", ".cargo/")
MAX_ISSUES = 2
ISSUE_LABEL = "review-bot"
VERDICTS = ("MUST_FIX", "CLEAN")
TOKEN_RE = re.compile(
    r"sk-ant-[A-Za-z0-9_-]{16,}"
    r"|gh[pousr]_[A-Za-z0-9]{30,}"
    r"|github_pat_[A-Za-z0-9_]{30,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)
VERDICT_RE = re.compile(r"^\*\*Verdict:\s*(MUST_FIX|CLEAN)\*\*", re.M)


class Refused(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def added_lines(diff_text):
    """{path: set of new-file line numbers the diff adds} from a unified diff."""
    out, path, line = {}, None, 0
    for raw in diff_text.splitlines():
        if raw.startswith("+++ "):
            name = raw[4:].strip()
            path = None if name == "/dev/null" else name[2:] if name.startswith("b/") else name
            continue
        if raw.startswith("--- ") or raw.startswith("diff --git"):
            continue
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
        if m:
            line = int(m.group(1))
            continue
        if path is None or line == 0:
            continue
        if raw.startswith("+"):
            out.setdefault(path, set()).add(line)
            line += 1
        elif raw.startswith("-") or raw.startswith("\\"):
            pass
        else:
            line += 1
    return out


def secrets_in(texts, secret_values):
    """Names of what leaked: 'env:<NAME>' for a secret's value, 'token-shape' for a token-shaped string."""
    found = []
    for name, value in secret_values:
        if value and len(value) >= 8 and any(value in t for t in texts):
            found.append("env:" + name)
    if any(TOKEN_RE.search(t) for t in texts):
        found.append("token-shape")
    return found


def keep_comments(comments, added):
    """The inline comments GitHub will accept, at most MAX_COMMENTS, and how many were dropped."""
    kept, dropped = [], 0
    for c in comments:
        ok = (
            isinstance(c, dict)
            and isinstance(c.get("path"), str)
            and isinstance(c.get("line"), int)
            and isinstance(c.get("body"), str)
            and c["body"].strip()
        )
        lines = added.get(c.get("path"), set()) if ok else set()
        start = c.get("start_line", c.get("line")) if ok else None
        if ok and not isinstance(start, int):
            ok = False
        if ok and all(n in lines for n in range(start, c["line"] + 1)) and start <= c["line"]:
            item = {"path": c["path"], "line": c["line"], "side": "RIGHT", "body": c["body"]}
            if start != c["line"]:
                item["start_line"] = start
                item["start_side"] = "RIGHT"
            kept.append(item)
        else:
            dropped += 1
    dropped += max(0, len(kept) - MAX_COMMENTS)
    return kept[:MAX_COMMENTS], dropped


def keep_issues(items, base):
    """Well-formed issue candidates, at most MAX_ISSUES, and how many were dropped."""
    kept, dropped = [], 0
    for it in items if isinstance(items, list) else []:
        ev = it.get("evidence") if isinstance(it, dict) else None
        ok = (
            isinstance(it, dict)
            and isinstance(it.get("title"), str) and 0 < len(it["title"].strip()) <= 120
            and isinstance(it.get("body"), str) and 0 < len(it["body"].strip()) <= 4000
            and isinstance(ev, list) and ev
            and all(isinstance(e, dict) and isinstance(e.get("path"), str) and isinstance(e.get("line"), int)
                    and ".." not in e["path"].split("/") and not e["path"].startswith("/") for e in ev)
        )
        if ok and base is not None:
            ok = all(os.path.isfile(os.path.join(base, e["path"])) for e in ev)
        if ok:
            kept.append({"title": it["title"].strip(), "body": it["body"].strip(),
                         "evidence": [{"path": e["path"], "line": e["line"]} for e in ev]})
        else:
            dropped += 1
    dropped += max(0, len(kept) - MAX_ISSUES)
    return kept[:MAX_ISSUES], dropped


def footer(head, verdict, body, mode, ledger=None, run=None):
    must = len(re.findall(r"^#### \d+\.", body, re.M))
    warn = len(re.findall(r"^⚠", body, re.M))
    tail = (f" ledger={ledger}" if ledger else "") + (f" run={run}" if run and str(run).isdigit() else "")
    return f"<!-- {MARKER} head_sha={head} verdict={verdict} must_fix={must} warn={warn} mode={mode}{tail} -->"


def usage_marker(paths):
    parts = []
    for p in paths:
        try:
            u = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            parts.append(f"{os.path.basename(p)}: unreadable")
            continue
        tools = ",".join(f"{k}:{v}" for k, v in (u.get("tools") or {}).items()) or "none"
        cost = u.get("cost_usd")
        parts.append(f"{u.get('role')}: turns={u.get('num_turns')} cost_usd={cost if cost is None else round(cost, 4)} "
                     f"ctx_max={u.get('context_max')} ctx_sum={u.get('context_sum')} "
                     f"out={(u.get('tokens') or {}).get('output')} tools={tools} denied={u.get('denied')}")
    return "<!-- bloomery-usage " + "; ".join(parts) + " -->" if parts else ""


def pr_paths(files_path, diff):
    """The pull request's file paths: from the `pulls/<n>/files` list (`status\t+a\t-d\tpath` lines) when given,
    else from the diff's headers (an incremental diff holds only the new commits)."""
    if files_path and os.path.exists(files_path):
        return [ln.split("\t")[-1] for ln in open(files_path, encoding="utf-8").read().splitlines() if ln.strip()]
    return sorted(set(re.findall(r"^\+\+\+ b/(\S+)", diff, re.M) + re.findall(r"^--- a/(\S+)", diff, re.M)))


def event_for(verdict, decision, body, paths):
    """("APPROVE" | "REQUEST_CHANGES" | "COMMENT", why a CLEAN review does not approve or None)."""
    if verdict == "MUST_FIX":
        return "REQUEST_CHANGES", None
    if decision != verdict:
        return "COMMENT", "decision.txt disagreed with the verdict line"
    if "<!-- validator:" not in body:
        return "COMMENT", "the validator did not finish"
    guarded = [p for p in paths if p.startswith(APPROVE_GUARD)]
    if guarded:
        return "COMMENT", (f"this pull request changes {', '.join(guarded[:3])}{' …' if len(guarded) > 3 else ''}, "
                           "which changes the bot, CI or the toolchain and needs a person's approval")
    return "APPROVE", None


def check(out, diff_path, pr_json, secret_env, usage_paths=(), base=None, files_path=None):
    try:
        body = open(os.path.join(out, "review.md"), encoding="utf-8").read()
        decision = open(os.path.join(out, "decision.txt"), encoding="utf-8").read().strip()
        pr = json.load(open(pr_json, encoding="utf-8"))
        diff = open(diff_path, encoding="utf-8", errors="replace").read()
    except (OSError, ValueError) as e:
        raise Refused(65, f"unreadable review output: {e}")
    comments_path = os.path.join(out, "comments.json")
    comments_text = ""
    comments = []
    if os.path.exists(comments_path):
        comments_text = open(comments_path, encoding="utf-8").read()
        try:
            comments = json.loads(comments_text or "[]")
        except ValueError as e:
            raise Refused(65, f"comments.json is not JSON: {e}")
        if not isinstance(comments, list):
            raise Refused(65, "comments.json is not a JSON array")
    issues_path = os.path.join(out, "issues.json")
    issues_text = open(issues_path, encoding="utf-8").read() if os.path.exists(issues_path) else ""
    try:
        issues = json.loads(issues_text or "[]")
    except ValueError as e:
        raise Refused(65, f"issues.json is not JSON: {e}")
    leaked = secrets_in([body, decision, comments_text, issues_text],
                        [(n, os.environ.get(n, "")) for n in secret_env])
    if leaked:
        raise Refused(3, "the review output carries a credential (" + ", ".join(leaked) + "); nothing is posted")
    if not body.strip():
        raise Refused(65, "review.md is empty")
    m = VERDICT_RE.search(body)
    if not m:
        raise Refused(65, "review.md has no '**Verdict: MUST_FIX|CLEAN**' line")
    verdict = m.group(1)
    notes = []
    if decision != verdict:
        notes.append(f"decision.txt said {decision or 'nothing'}; the body's verdict line ({verdict}) is used")
    body = re.sub(rf"(\n*<!-- (?:{MARKER}|bloomery-usage) [^>]*-->)+\s*$", "", body.rstrip())
    body = re.sub(r"\n*<sub>🔨 Shingler[^\n]*</sub>\s*$", "", body)  # a second check (the post job)
    kept, dropped = keep_comments(comments, added_lines(diff))
    if dropped:
        notes.append(f"{dropped} inline suggestion(s) were dropped: not on lines this diff adds, or over the cap")
    issues, idropped = keep_issues(issues, base)
    if idropped:
        notes.append(f"{idropped} issue candidate(s) were dropped: malformed, evidence missing, or over the cap")
    cap = MAX_BODY - len(SIGNATURE) - 100  # the signature still fits under the cap
    if len(body) > cap:
        body = body[:cap].rsplit("\n", 1)[0] + "\n\n(The review was cut at the length cap.)"
    if notes:
        body += "\n\n<sub>" + " · ".join(notes) + "</sub>"
    usage_txt = os.path.join(out, "usage.txt")
    marker = usage_marker(usage_paths)
    if marker:
        with open(usage_txt, "w", encoding="utf-8") as f:
            f.write(marker + "\n")
    elif os.path.exists(usage_txt):
        marker = open(usage_txt, encoding="utf-8").read().strip()
    event, why = event_for(verdict, decision, body, pr_paths(files_path, diff))
    # A second check never raises the event: the first one saw the reviewer's own decision.txt, which it then rewrote.
    ev_path = os.path.join(out, "event.txt")
    if os.path.exists(ev_path) and open(ev_path, encoding="utf-8").read().strip() == "COMMENT":
        event = "COMMENT"
    if "\n<sub>Not approved:" in body:
        why = None
    if why:
        body += f"\n\n<sub>Not approved: {why}.</sub>"
    body += "\n\n" + SIGNATURE.format(repo=pr.get("repo") or "midagedev/bloomery")
    body += "\n\n" + footer(pr["head_sha"], verdict, body, pr.get("mode", "full"), pr.get("ledger"), pr.get("run"))
    if marker:
        body += "\n" + marker
    with open(os.path.join(out, "review.md"), "w", encoding="utf-8") as f:
        f.write(body + "\n")
    with open(os.path.join(out, "decision.txt"), "w", encoding="utf-8") as f:
        f.write(verdict + "\n")
    with open(os.path.join(out, "event.txt"), "w", encoding="utf-8") as f:
        f.write(event + "\n")
    with open(os.path.join(out, "comments.json"), "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=1)
    with open(issues_path, "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=1)
    with open(os.path.join(out, "head_sha.txt"), "w", encoding="utf-8") as f:
        f.write(pr["head_sha"] + "\n")
    with open(os.path.join(out, "base_sha.txt"), "w", encoding="utf-8") as f:
        # Issue evidence is verified in BASE_TREE, so its links point at the commit that tree was checked out at.
        f.write((pr.get("tree_sha") or pr.get("base_sha", "")) + "\n")
    return verdict, len(kept), notes


def api(method, url, token, payload=None):
    req = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def issue_body(it, repo, base_sha, pr):
    links = "\n".join(f"- [{e['path']}:{e['line']}](https://github.com/{repo}/blob/{base_sha}/{e['path']}#L{e['line']})"
                      for e in it["evidence"])
    return (f"{it['body']}\n\nEvidence (at the base of #{pr}):\n{links}\n\n"
            f"<sub>Filed by Shingler, the review bot, while reviewing #{pr}; the defect is outside that pull request.</sub>\n"
            f"<!-- {MARKER}-issue pr={pr} -->")


def file_issues(out, repo, pr, token, call=api):
    items = json.load(open(os.path.join(out, "issues.json"), encoding="utf-8"))
    filed = []
    if items:
        base_sha = open(os.path.join(out, "base_sha.txt"), encoding="utf-8").read().strip()
        existing = {}
        page = 1
        while True:
            got = call("GET", f"https://api.github.com/repos/{repo}/issues?labels={ISSUE_LABEL}&state=all"
                              f"&per_page=100&page={page}", token)
            for i in got:
                existing[i["title"].strip().lower()] = i["number"]
            if len(got) < 100:
                break
            page += 1
        for it in items:
            n = existing.get(it["title"].lower())
            if n is not None:
                filed.append(f"existing #{n}")
                continue
            r = call("POST", f"https://api.github.com/repos/{repo}/issues", token,
                     {"title": it["title"], "body": issue_body(it, repo, base_sha, pr), "labels": [ISSUE_LABEL]})
            filed.append(f"#{r['number']}")
    with open(os.path.join(out, "filed.txt"), "w", encoding="utf-8") as f:
        f.write(", ".join(filed) + ("\n" if filed else ""))
    return filed


def post(out, repo, pr, token):
    body = open(os.path.join(out, "review.md"), encoding="utf-8").read()
    filed_path = os.path.join(out, "filed.txt")
    filed = open(filed_path, encoding="utf-8").read().strip() if os.path.exists(filed_path) else ""
    if filed:
        body = body.replace(f"<!-- {MARKER} ", f"Filed outside this PR: {filed}\n\n<!-- {MARKER} ", 1)
    head = open(os.path.join(out, "head_sha.txt"), encoding="utf-8").read().strip()
    comments = json.load(open(os.path.join(out, "comments.json"), encoding="utf-8"))
    url = f"https://api.github.com/repos/{repo}/pulls/{pr}/reviews"
    ev_path = os.path.join(out, "event.txt")
    event = open(ev_path, encoding="utf-8").read().strip() if os.path.exists(ev_path) else "COMMENT"
    if event not in ("APPROVE", "REQUEST_CHANGES", "COMMENT"):
        raise Refused(65, f"event.txt says {event!r}; only APPROVE, REQUEST_CHANGES or COMMENT is posted")
    payload = {"commit_id": head, "event": event, "body": body, "comments": comments}
    try:
        r = api("POST", url, token, payload)
    except urllib.error.HTTPError as e:
        if e.code != 422 or not comments:
            raise
        payload["comments"] = []
        payload["body"] = body.replace(
            f"<!-- {MARKER} ", "<sub>GitHub refused the inline suggestions; they are left out.</sub>\n\n"
            f"<!-- {MARKER} ", 1)
        r = api("POST", url, token, payload)
    return r.get("html_url", "")


def self_test():
    assert footer("h", "CLEAN", "", "full").endswith("mode=full -->")
    assert footer("h", "CLEAN", "", "full", "0a1b2c").endswith("mode=full ledger=0a1b2c -->")
    assert footer("h", "CLEAN", "", "full", "0a1b2c", "123").endswith("ledger=0a1b2c run=123 -->")
    assert footer("h", "CLEAN", "", "full", None, "1 x").endswith("mode=full -->")
    diff = (
        "diff --git a/crates/x/src/a.rs b/crates/x/src/a.rs\n--- a/crates/x/src/a.rs\n+++ b/crates/x/src/a.rs\n"
        "@@ -10,3 +10,4 @@ fn f() {\n ctx\n-old\n+new1\n+new2\n ctx2\n"
        "diff --git a/tools/n.sh b/tools/n.sh\n--- /dev/null\n+++ b/tools/n.sh\n@@ -0,0 +1,2 @@\n+a\n+b\n"
        "diff --git a/gone.rs b/gone.rs\n--- a/gone.rs\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
    )
    got = added_lines(diff)
    assert got == {"crates/x/src/a.rs": {11, 12}, "tools/n.sh": {1, 2}}, got

    kept, dropped = keep_comments(
        [
            {"path": "crates/x/src/a.rs", "line": 11, "body": "b"},
            {"path": "crates/x/src/a.rs", "line": 12, "start_line": 11, "body": "b"},
            {"path": "crates/x/src/a.rs", "line": 10, "body": "context line, not added"},
            {"path": "crates/x/src/a.rs", "line": 13, "start_line": 11, "body": "runs past the hunk"},
            {"path": "nope.rs", "line": 1, "body": "b"},
            {"path": "tools/n.sh", "line": "1", "body": "line is a string"},
            "not an object",
        ],
        got,
    )
    assert [(c["line"], c.get("start_line")) for c in kept] == [(11, None), (12, 11)], kept
    assert dropped == 5, dropped
    many, over = keep_comments([{"path": "tools/n.sh", "line": 1, "body": "x"}] * 12, got)
    assert len(many) == MAX_COMMENTS and over == 2

    assert secrets_in(["fine"], [("K", "")]) == []
    assert secrets_in(["key sk-ant-api03-abcdefghijklmnopqrst"], []) == ["token-shape"]
    assert secrets_in(["x MYSECRETVALUE y"], [("K", "MYSECRETVALUE")]) == ["env:K"]
    assert secrets_in(["ghp_" + "a" * 36], []) == ["token-shape"]

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out")
        os.mkdir(out)
        dp, pj = os.path.join(d, "diff.patch"), os.path.join(d, "pr.json")
        open(dp, "w").write(diff)
        json.dump({"head_sha": "abc123", "mode": "full"}, open(pj, "w"))

        def write(body, decision, comments=None):  # a fresh review job's output
            if os.path.exists(os.path.join(out, "event.txt")):
                os.remove(os.path.join(out, "event.txt"))
            open(os.path.join(out, "review.md"), "w").write(body)
            open(os.path.join(out, "decision.txt"), "w").write(decision)
            cp = os.path.join(out, "comments.json")
            if comments is None:
                if os.path.exists(cp):
                    os.remove(cp)
            else:
                open(cp, "w").write(comments)

        body = "**Verdict: MUST_FIX**: one.\n\n### Must fix\n#### 1. t\nx\n\n### Non-blocking\n⚠ a\n⚠ b\n"
        write(body, "CLEAN", json.dumps([{"path": "tools/n.sh", "line": 2, "body": "s"},
                                         {"path": "tools/n.sh", "line": 9, "body": "s"}]))
        verdict, n, notes = check(out, dp, pj, [])
        assert (verdict, n) == ("MUST_FIX", 1) and len(notes) == 2, (verdict, n, notes)
        posted = open(os.path.join(out, "review.md")).read()
        assert posted.rstrip().endswith(
            "<!-- bloomery-review head_sha=abc123 verdict=MUST_FIX must_fix=1 warn=2 mode=full -->"), posted
        assert open(os.path.join(out, "decision.txt")).read() == "MUST_FIX\n"
        # A second check (the post job repeats it) keeps one footer and one signature.
        check(out, dp, pj, [])
        again = open(os.path.join(out, "review.md")).read()
        assert again.count("<!-- bloomery-review") == 1 and again.count("🔨 Shingler") == 1, again
        assert open(os.path.join(out, "event.txt")).read() == "REQUEST_CHANGES\n"

        # APPROVE only for a CLEAN verdict the decision agrees with, after the validator, off the guarded paths.
        clean = "**Verdict: CLEAN**: fine.\n\n⚠ a\n<!-- validator: removed=0 downgraded=0 upgraded=0 -->\n"
        fp = os.path.join(d, "files.txt")
        for body_, decision, files, want, why in (
                (clean, "CLEAN", "modified\t+1\t-0\ttools/n.sh\n", "APPROVE", None),
                (clean, "MUST_FIX", "modified\t+1\t-0\ttools/n.sh\n", "COMMENT", "decision.txt disagreed"),
                (clean.split("<!--")[0], "CLEAN", "modified\t+1\t-0\ttools/n.sh\n", "COMMENT", "validator"),
                (clean, "CLEAN", "modified\t+1\t-0\ttools/n.sh\nmodified\t+1\t-0\t.github/workflows/ci.yml\n",
                 "COMMENT", ".github/workflows/ci.yml"),
                (clean, "CLEAN", None, "APPROVE", None)):
            write(body_, decision)
            if files is None:
                os.remove(fp)
            else:
                open(fp, "w").write(files)
            check(out, dp, pj, [], files_path=fp if files is not None else None)
            got = open(os.path.join(out, "event.txt")).read().strip()
            text = open(os.path.join(out, "review.md")).read()
            assert got == want and ("Not approved:" in text) == bool(why) and (not why or why in text), (got, text)
            check(out, dp, pj, [], files_path=fp if files is not None else None)
            assert open(os.path.join(out, "review.md")).read() == text  # a second check changes nothing
        assert pr_paths(None, "--- a/x.rs\n+++ b/x.rs\n--- /dev/null\n+++ b/.github/y\n") == [".github/y", "x.rs"]
        write(body, "CLEAN")
        check(out, dp, pj, [])

        up = os.path.join(d, "reviewer.usage.json")
        json.dump({"role": "reviewer", "num_turns": 7, "cost_usd": 0.51234, "context_max": 90000,
                   "context_sum": 400000, "tokens": {"output": 9000}, "tools": {"Read": 20, "Grep": 9},
                   "denied": 0}, open(up, "w"))
        check(out, dp, pj, [], [up])
        check(out, dp, pj, [])  # the post job has no usage files; usage.txt keeps the marker
        posted = open(os.path.join(out, "review.md")).read()
        assert posted.count("<!-- bloomery-usage reviewer: turns=7 cost_usd=0.5123 ctx_max=90000") == 1, posted
        assert posted.count("<!-- bloomery-review") == 1 and "tools=Read:20,Grep:9" in posted, posted

        base = os.path.join(d, "base")
        os.makedirs(os.path.join(base, "crates"))
        open(os.path.join(base, "crates", "a.rs"), "w").write("x")
        good = {"title": "gguf: a NaN block quantizes to code 0", "body": "b", "evidence": [{"path": "crates/a.rs", "line": 3}]}
        open(os.path.join(out, "issues.json"), "w").write(json.dumps([
            good, {**good, "title": "missing file", "evidence": [{"path": "crates/nope.rs", "line": 1}]},
            {**good, "evidence": [{"path": "../etc/passwd", "line": 1}]}, {**good, "title": ""},
            {**good, "title": "third"}, {**good, "title": "fourth"}]))
        check(out, dp, pj, [], base=base)
        assert [i["title"] for i in json.load(open(os.path.join(out, "issues.json")))] == [good["title"], "third"]

        calls = []

        def fake(method, url, token, payload=None):
            calls.append((method, url, payload))
            if method == "GET":
                return [{"title": "Third", "number": 7}]
            return {"number": 41}

        assert file_issues(out, "o/r", "5", "t", fake) == ["#41", "existing #7"]
        assert open(os.path.join(out, "filed.txt")).read() == "#41, existing #7\n"
        assert calls[1][2]["labels"] == [ISSUE_LABEL] and "<!-- bloomery-review-issue pr=5 -->" in calls[1][2]["body"]
        open(os.path.join(out, "issues.json"), "w").write("[]")
        assert file_issues(out, "o/r", "5", "t", fake) == [] and len(calls) == 2

        # post sends the event check decided, and refuses any other.
        open(os.path.join(out, "head_sha.txt"), "w").write("abc123\n")
        real_api, g = api, globals()
        g["api"] = fake
        try:
            for ev in ("APPROVE", "REQUEST_CHANGES", "COMMENT"):
                open(os.path.join(out, "event.txt"), "w").write(ev + "\n")
                post(out, "o/r", "5", "t")
                assert calls[-1][0] == "POST" and calls[-1][2]["event"] == ev, calls[-1]
            open(os.path.join(out, "event.txt"), "w").write("DISMISS\n")
            try:
                post(out, "o/r", "5", "t")
                raise AssertionError("an unknown event was posted")
            except Refused:
                pass
        finally:
            g["api"] = real_api

        write("**Verdict: CLEAN**: fine.\nsk-ant-api03-abcdefghijklmnopqrstuv\n", "CLEAN")
        try:
            check(out, dp, pj, [])
            raise AssertionError("a token-shaped string was not refused")
        except Refused as e:
            assert e.code == 3

        os.environ["POST_SELFTEST_SECRET"] = "s3cr3t-value-123"
        write("**Verdict: CLEAN**: fine.\n", "CLEAN", '[{"path":"tools/n.sh","line":1,"body":"s3cr3t-value-123"}]')
        try:
            check(out, dp, pj, ["POST_SELFTEST_SECRET"])
            raise AssertionError("a secret's value in comments.json was not refused")
        except Refused as e:
            assert e.code == 3

        write("No verdict line.\n", "CLEAN")
        try:
            check(out, dp, pj, [])
            raise AssertionError("a body without a verdict line was not refused")
        except Refused as e:
            assert e.code == 65

        write("**Verdict: CLEAN**: long.\n" + ("line\n" * 4000), "CLEAN")
        check(out, dp, pj, [])
        assert len(open(os.path.join(out, "review.md")).read()) < MAX_BODY + 400
    print("post.py self-test: ok")


def main(argv):
    if argv[1:] == ["--self-test"]:
        self_test()
        return 0
    if len(argv) < 2 or argv[1] not in ("check", "issues", "post"):
        print(__doc__, file=sys.stderr)
        return 64
    opts, secret_env, usage, rest = {}, [], [], argv[2:]
    while rest:
        k = rest.pop(0)
        if not rest:
            print(f"{k} needs a value", file=sys.stderr)
            return 64
        v = rest.pop(0)
        if k == "--secret-env":
            secret_env.append(v)
        elif k == "--usage":
            usage.append(v)
        else:
            opts[k] = v
    try:
        if argv[1] == "check":
            verdict, n, notes = check(opts["--out"], opts["--diff"], opts["--pr-json"], secret_env, usage,
                                      opts.get("--base"), opts.get("--files"))
            event = open(os.path.join(opts["--out"], "event.txt"), encoding="utf-8").read().strip()
            print(f"check: verdict={verdict} event={event} inline={n}" + "".join(f"\n  note: {x}" for x in notes))
        else:
            token = os.environ.get("GH_TOKEN", "")
            if not token:
                print(f"{argv[1]}: GH_TOKEN is not set", file=sys.stderr)
                return 64
            if argv[1] == "issues":
                print("issues:", ", ".join(file_issues(opts["--out"], opts["--repo"], opts["--pr"], token)) or "none")
            else:
                print("posted:", post(opts["--out"], opts["--repo"], opts["--pr"], token))
    except KeyError as e:
        print(f"missing option {e}", file=sys.stderr)
        return 64
    except Refused as e:
        print(f"{argv[1]}: {e}", file=sys.stderr)
        return e.code
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
