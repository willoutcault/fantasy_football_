import os
import time
from datetime import datetime
from typing import List, Dict

from flask import request, send_file, redirect, url_for

import requests
import hashlib
import pathlib
import subprocess
import shutil
from dotenv import load_dotenv
from flask import Flask, render_template_string, request, send_file, redirect, url_for
from jinja2 import DictLoader

# ----------------- ENV -----------------
load_dotenv()

LEAGUE_ID = int(os.getenv("LEAGUE_ID", "0"))
SEASON = int(os.getenv("SEASON", datetime.now().year))
ESPN_S2 = os.getenv("ESPN_S2", "")
SWID = os.getenv("SWID", "")

# Discord (optional; the section hides automatically if not configured)
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID", "")
DISCORD_VIDEOS_MAX = int(os.getenv("DISCORD_VIDEOS_MAX", "8"))
DISCORD_FETCH_LIMIT = int(os.getenv("DISCORD_FETCH_LIMIT", "75"))      # initial scan size (<=100)
DISCORD_CACHE_SECS = int(os.getenv("DISCORD_CACHE_SECS", "1600"))       # cache TTL

if not LEAGUE_ID or not ESPN_S2 or not SWID:
    raise SystemExit("Please set LEAGUE_ID, ESPN_S2, and SWID in your .env file.")

try:
    from espn_api.football import League
except ImportError as e:
    raise SystemExit("Missing dependency espn-api. Run: pip install espn-api") from e

# ----------------- Thumbnail Setup -----------------
THUMB_DIR = pathlib.Path("/tmp/thumbs")
THUMB_DIR.mkdir(parents=True, exist_ok=True)

def _thumb_path_for(url: str) -> pathlib.Path:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return THUMB_DIR / f"{h}.jpg"

def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None

def _try_render_thumb(vurl: str, out_path: pathlib.Path, ss: str) -> bool:
    # NOTE: Timestamp format must be 00:00:SS.mmm (e.g., 00:00:03.000)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", ss,               # fast seek to avoid black frame at t=0
        "-i", vurl,              # remote URL (ffmpeg follows redirects)
        "-frames:v", "1",        # grab exactly one frame
        "-vf", "thumbnail,scale=640:-1",  # pick a representative frame and resize
        "-q:v", "4",             # quality (1..31); 4 is plenty for a poster
        "-y", str(out_path)
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,    # keep errors for logs
            timeout=10                 # a bit more forgiving than 6s
        )
        return out_path.exists()
    except subprocess.TimeoutExpired:
        print(f"[thumb] ffmpeg timeout at ss={ss}")
    except subprocess.CalledProcessError as e:
        # Include a short snippet of stderr to help debug
        err = (e.stderr or b"").decode("utf-8", errors="ignore")[:300]
        print(f"[thumb] ffmpeg error at ss={ss}: {err}")
    except Exception as e:
        print(f"[thumb] unexpected error at ss={ss}: {e}")
    return False

# ----------------- ESPN helpers -----------------
def get_league(year: int | None = None):
    return League(
        league_id=LEAGUE_ID,
        year=year or SEASON,
        espn_s2=ESPN_S2,
        swid=SWID,
    )

def safe_team_name(team):
    if getattr(team, "team_name", None):
        return team.team_name
    if getattr(team, "owner", None):
        return str(team.owner)
    if getattr(team, "owners", None) and team.owners:
        return ", ".join(team.owners)
    return f"Team {getattr(team, 'team_id', '?')}"

def team_logo(team):
    return getattr(team, "logo_url", None) or getattr(team, "logo", None) or ""

def team_logo_map(league):
    return {safe_team_name(t): team_logo(t) for t in league.teams}

def overall_standings(league):
    rows = []
    for t in league.standings():
        rows.append({
            "team": safe_team_name(t),
            "logo": team_logo(t),
            "wins": t.wins,
            "losses": t.losses,
            "ties": getattr(t, "ties", 0),
            "pf": round(t.points_for, 2),
            "pa": round(t.points_against, 2),
            "streak_len": getattr(t, "streak_length", 0),
            "streak_type": str(getattr(t, "streak_type", "") or "").upper(),  # "WIN"/"LOSS" if available
        })
    return rows

def stories_sorted_teams(league):
    """Return teams sorted by record (wins desc, losses asc), then PF desc."""
    items = []
    for t in league.teams:
        wins = int(getattr(t, "wins", 0))
        losses = int(getattr(t, "losses", 0))
        ties = int(getattr(t, "ties", 0) or 0)
        pf = round(float(getattr(t, "points_for", 0.0) or 0.0), 2)

        # streak
        streak_len = int(getattr(t, "streak_length", 0) or 0)
        stype = str(getattr(t, "streak_type", "") or "").upper()
        if not stype:
            stype = "WIN" if streak_len >= 0 else "LOSS"
            streak_len = abs(streak_len)
        streak = f"W{streak_len}" if stype == "WIN" and streak_len > 0 else (f"L{streak_len}" if stype == "LOSS" and streak_len > 0 else "—")

        items.append({
            "team": safe_team_name(t),
            "logo": team_logo(t),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "pf": pf,
            "streak": streak,
            "record": f"{wins}-{losses}" + (f"-{ties}" if ties else "")
        })

    items.sort(key=lambda x: (-x["wins"], x["losses"], -x["pf"], x["team"].lower()))
    return items

def last_completed_week(league):
    wk = max(1, league.current_week - 1)
    return min(wk, league.settings.reg_season_count)


def stories_sorted_teams(league):
    """Return teams sorted by record (wins desc, losses asc), then PF desc,
       including streak color class and extra stats for tooltips."""
    items = []
    for t in league.teams:
        wins = int(getattr(t, "wins", 0))
        losses = int(getattr(t, "losses", 0))
        ties = int(getattr(t, "ties", 0) or 0)
        pf = round(float(getattr(t, "points_for", 0.0) or 0.0), 2)
        pa = round(float(getattr(t, "points_against", 0.0) or 0.0), 2)

        streak_len = int(getattr(t, "streak_length", 0) or 0)
        stype = str(getattr(t, "streak_type", "") or "").upper()
        if not stype:
            stype = "WIN" if streak_len >= 0 else "LOSS"
            streak_len = abs(streak_len)

        streak_txt = (
            f"W{streak_len}" if stype == "WIN" and streak_len > 0 else
            f"L{streak_len}" if stype == "LOSS" and streak_len > 0 else "—"
        )
        streak_class = (
            "streak-win" if stype == "WIN" and streak_len > 0 else
            "streak-loss" if stype == "LOSS" and streak_len > 0 else
            "streak-none"
        )

        items.append({
            "team": safe_team_name(t),
            "logo": team_logo(t),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "pf": pf,
            "pa": pa,
            "streak": streak_txt,
            "streak_class": streak_class,
            "record": f"{wins}-{losses}" + (f"-{ties}" if ties else ""),
        })

    # Sort: wins ↓, losses ↑, PF ↓, then name ↑
    items.sort(key=lambda x: (-x["wins"], x["losses"], -x["pf"], x["team"].lower()))
    # Add rank after sorting
    for i, it in enumerate(items, 1):
        it["rank"] = i
    return items


# ----------------- Scoreboard -----------------
STARTER_EXCLUDE_SLOTS = {"BE", "BN", "IR", "RES"}

def expected_starter_count(league) -> int:
    """
    Count how many starting slots this league expects (exclude bench/IR/reserve).
    """
    total = 0
    # espn-api exposes a mapping like {'QB':1, 'RB':2, 'WR':2, 'TE':1, 'FLEX':1, 'D/ST':1, 'K':1, 'BE':7, 'IR':2}
    for slot, count in getattr(league.settings, "roster_positions", {}).items():
        if slot in STARTER_EXCLUDE_SLOTS:
            continue
        total += int(count or 0)
    return total

def _has_real_game(p) -> bool:
    opp = getattr(p, "pro_opponent", None)
    if not opp:
        return False
    return str(opp).strip().upper() not in {"BYE", "NONE"}

def _is_game_final(p) -> bool:
    if not _has_real_game(p):
        return False
    try:
        if int(getattr(p, "game_played", 0) or 0) >= 1:
            return True
    except Exception:
        pass
    status = (getattr(p, "game_status", None) or getattr(p, "proGameStatus", None) or "")
    return str(status).lower() in {"final", "finished", "complete", "completed", "post"}

def _any_unfinished_starter(lineup):
    for p in lineup or []:
        slot = getattr(p, "slot_position", "") or ""
        if slot in STARTER_EXCLUDE_SLOTS:
            continue
        if _has_real_game(p) and not _is_game_final(p):
            return True
    return False

def scoreboard_with_status(league):
    wk = league.current_week
    rows = []
    for box in league.box_scores(wk):
        home = box.home_team; away = box.away_team
        home_line = getattr(box, "home_lineup", []) or []
        away_line = getattr(box, "away_lineup", []) or []
        ongoing = _any_unfinished_starter(home_line) or _any_unfinished_starter(away_line)
        rows.append({
            "home": safe_team_name(home),
            "home_logo": team_logo(home),
            "away": safe_team_name(away) if away else "(bye)",
            "away_logo": team_logo(away) if away else "",
            "home_score": round(box.home_score, 2),
            "away_score": round(box.away_score, 2) if away else 0.0,
            "ongoing": ongoing,  # True => 🕒, False => ✅
        })
    return rows

def week_completed(league, week) -> bool:
    for box in league.box_scores(week):
        if _any_unfinished_starter(getattr(box, "home_lineup", []) or []):
            return False
        if box.away_team and _any_unfinished_starter(getattr(box, "away_lineup", []) or []):
            return False
    return True


# ----------------- Smirnoff (per-week + season) -----------------
def _collect_ice_from_lineup(lineup, week, expected_starters: 10):
    """
    Returns (zeros, negatives, offenders)
    offenders: list of {"name", "points", "ices", "week", ["reason"]}

    Rules:
      • negative points => 2 ices
      • zero points     => 1 ice
      • starter on BYE  => 1 ice
      • empty starter   => 1 ice  (computed by comparing filled vs expected)
    """
    zeros = negatives = 0
    offenders = []

    # Count filled starters (excludes BN/IR/RES etc.)
    starters = [p for p in (lineup or []) if (getattr(p, "slot_position", "") or "") not in STARTER_EXCLUDE_SLOTS]

    # 1) Score-based ices, and BYE starters
    for p in starters:
        # BYE starter (no real opponent)
        if not _has_real_game(p):
            offenders.append({
                "name": getattr(p, "name", "Unknown"),
                "points": 0.0,
                "ices": 1,
                "week": week,
                "reason": "BYE starter"
            })
            continue

        # Only count score-based ices if the game is final
        if not _is_game_final(p):
            continue

        pts = float(getattr(p, "points", 0.0) or 0.0)
        if pts < 0:
            negatives += 1
            offenders.append({
                "name": getattr(p, "name", "Unknown"),
                "points": round(pts, 2),
                "ices": 2,
                "week": week,
                "reason": "Negative points"
            })
        elif pts == 0:
            zeros += 1
            offenders.append({
                "name": getattr(p, "name", "Unknown"),
                "points": 0.0,
                "ices": 1,
                "week": week,
                "reason": "Zero points"
            })

    # 2) Empty starter slots (if expected_starters provided)
    if expected_starters is not None:
        filled = len(starters)
        missing = max(0, expected_starters - filled)
        for _ in range(missing):
            offenders.append({
                "name": "Empty Starter",
                "points": 0.0,
                "ices": 1,
                "week": week,
                "reason": "Empty lineup slot"
            })

    return zeros, negatives, offenders

def ice_tracker_for_week(league, week, logo_map):
    rows = []
    expected = 10

    for box in league.box_scores(week):
        # home
        z, n, offenders = _collect_ice_from_lineup(getattr(box, "home_lineup", []) or [], week, expected_starters=expected)
        home_total = sum(o["ices"] for o in offenders)
        h_team_name = safe_team_name(box.home_team)
        rows.append({
            "team": h_team_name,
            "logo": logo_map.get(h_team_name, ""),
            "total_ices": home_total,
            "offenders": offenders
        })

        # away (if present)
        if box.away_team:
            z, n, offenders = _collect_ice_from_lineup(getattr(box, "away_lineup", []) or [], week, expected_starters=expected)
            away_total = sum(o["ices"] for o in offenders)
            a_team_name = safe_team_name(box.away_team)
            rows.append({
                "team": a_team_name,
                "logo": logo_map.get(a_team_name, ""),
                "total_ices": away_total,
                "offenders": offenders
            })

    # combine by team (unchanged)
    combined = {}
    for r in rows:
        t = r["team"]
        if t not in combined:
            combined[t] = {"team": t, "logo": r.get("logo", ""), "total_ices": 0, "offenders": []}
        combined[t]["total_ices"] += r["total_ices"]
        combined[t]["offenders"] += r["offenders"]

    out = list(combined.values())
    out.sort(key=lambda x: (-x["total_ices"], x["team"].lower()))
    return out

def ice_tracker_season_to_date(league):
    logo_map = team_logo_map(league)
    last_wk = last_completed_week(league)
    by_team = {}
    for wk in range(1, last_wk + 1):
        for row in ice_tracker_for_week(league, wk, logo_map):
            t = row["team"]
            if t not in by_team:
                by_team[t] = {"team": t, "logo": row.get("logo",""), "total_ices": 0, "offenders": []}
            by_team[t]["total_ices"] += row["total_ices"]
            by_team[t]["offenders"].extend(row["offenders"])

    result = []
    for rec in by_team.values():
        icons = []
        for o in rec["offenders"]:
            reason = f' • {o["reason"]}' if "reason" in o and o["reason"] else ""
            tip = f'Wk {o["week"]} • {o["name"]} • {o["points"]:+.2f} pts{reason}'
            for _ in range(o["ices"]):
                icons.append({"tip": tip})
        rec["icons"] = icons
        result.append(rec)

    result.sort(key=lambda x: (-x["total_ices"], x["team"].lower()))
    return result


# ----------------- Season Storylines -----------------
def ice_king(league):
    """Return {'team': str, 'logo': str, 'ices': int} for the team with the most ices this season."""
    rows = ice_tracker_season_to_date(league)
    if not rows:
        return None
    top = max(rows, key=lambda r: r["total_ices"])
    return {"team": top["team"], "logo": top.get("logo", ""), "ices": top["total_ices"]}

def streak_tracker(league):
    """Return {'hot': [(team, len)], 'cold': [(team, len)]}."""
    hot, cold = [], []
    for t in league.teams:
        length = int(getattr(t, "streak_length", 0) or 0)
        stype = str(getattr(t, "streak_type", "") or "").upper()  # "WIN"/"LOSS" if present
        if not stype:
            stype = "WIN" if length >= 0 else "LOSS"
            length = abs(length)
        if stype == "WIN" and length >= 2:
            hot.append((safe_team_name(t), length, team_logo(t)))
        if stype == "LOSS" and length >= 2:
            cold.append((safe_team_name(t), length, team_logo(t)))
    hot.sort(key=lambda x: (-x[1], x[0].lower()))
    cold.sort(key=lambda x: (-x[1], x[0].lower()))
    return {"hot": hot, "cold": cold}

def top_dog_last_year():
    """Return {'team': str, 'logo': str} for last season's champ if accessible."""
    try:
        prev = get_league(SEASON - 1)
        champ_team = prev.standings()[0]  # first in standings = champion
        return {"team": safe_team_name(champ_team), "logo": team_logo(champ_team)}
    except Exception:
        return None


# ----------------- Discord (robust fetch: channel + threads + pagination) -----------------
def _discord_get(url, params=None):
    if not DISCORD_BOT_TOKEN:
        raise RuntimeError("Missing DISCORD_BOT_TOKEN")
    headers = {"Authorization": f"Bot {DISCORD_BOT_TOKEN}"}
    return requests.get(url, headers=headers, params=params, timeout=15)

# Accept common video extensions; keep .mov too
_VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".ogg", ".avi", ".mkv"}

def _looks_like_video(att: dict) -> bool:
    """
    True if an attachment is likely a video.
    Some mobile uploads set content_type to 'video/quicktime' (MOV) or omit it.
    We accept if content_type starts with video/ OR filename has a known extension.
    """
    ct = (att.get("content_type") or "").lower()
    fn = (att.get("filename") or "").lower()
    return ct.startswith("video/") or any(fn.endswith(ext) for ext in _VIDEO_EXTS)

def _extract_video_urls_from_message(msg: dict) -> List[dict]:
    """
    Collect BOTH attached videos and embedded videos (e.g., oversized uploads or links
    that Discord turns into playable embeds). Returns a list of dicts with 'url', 'filename',
    'created_at', 'author', and 'where'.
    """
    items = []
    author = (msg.get("author") or {}).get("username", "unknown")
    ts = msg.get("timestamp")

    # 1) Attachments (most reliable)
    for att in msg.get("attachments", []):
        if _looks_like_video(att):
            items.append({
                "url": att.get("url"),
                "filename": att.get("filename") or "attachment-video",
                "created_at": ts,
                "author": author,
                "where": "attachment",
            })

    # 2) Embeds with video (oversized uploads or external providers)
    for emb in msg.get("embeds", []):
        vid = emb.get("video") or {}
        vurl = vid.get("url")
        if vurl:
            items.append({
                "url": vurl,
                "filename": emb.get("title") or "embed-video",
                "created_at": ts,
                "author": author,
                "where": f"embed:{(emb.get('provider') or {}).get('name', 'unknown')}",
            })

    return items


def _paginate_messages(channel_id: str, max_msgs: int = 500) -> List[dict]:
    msgs: List[dict] = []
    before = None
    remaining = max_msgs
    while remaining > 0:
        limit = min(100, remaining)
        params = {"limit": limit}
        if before:
            params["before"] = before
        r = _discord_get(f"https://discord.com/api/v10/channels/{channel_id}/messages", params=params)
        if not r.ok:
            print(f"[discord] messages fetch failed {r.status_code}: {r.text[:200]}")
            break
        batch = r.json()
        if not batch:
            break
        msgs.extend(batch)
        before = batch[-1]["id"]
        remaining -= len(batch)
        if len(batch) < limit:
            break
    return msgs

def _list_thread_ids(parent_channel_id: str, cap: int = 50) -> List[str]:
    ids = set()
    try:
        ra = _discord_get(f"https://discord.com/api/v10/channels/{parent_channel_id}/threads/active")
        if ra.ok:
            for th in ra.json().get("threads", []):
                ids.add(th.get("id"))
        rp = _discord_get(f"https://discord.com/api/v10/channels/{parent_channel_id}/threads/archived/public",
                          params={"limit": cap})
        if rp.ok:
            for th in rp.json().get("threads", []):
                ids.add(th.get("id"))
        rpr = _discord_get(f"https://discord.com/api/v10/channels/{parent_channel_id}/threads/archived/private",
                           params={"limit": cap})
        if rpr.ok:
            for th in rpr.json().get("threads", []):
                ids.add(th.get("id"))
    except Exception as e:
        print(f"[discord] thread listing error: {e}")
    return list(ids)

# simple cache
_discord_cache: Dict[str, Dict] = {"ts": 0, "items": []}

def fetch_discord_videos(max_items: int = None) -> List[dict]:
    """Fetch latest video attachments from a Discord channel + its threads (cached)."""
    if not DISCORD_BOT_TOKEN or not DISCORD_CHANNEL_ID:
        return []
    if max_items is None:
        max_items = DISCORD_VIDEOS_MAX

    now = time.time()
    if (now - _discord_cache["ts"]) < DISCORD_CACHE_SECS and _discord_cache["items"]:
        return _discord_cache["items"][:max_items]

    vids: List[dict] = []

    # 1) Parent channel with pagination
    try:
        parent_msgs = _paginate_messages(DISCORD_CHANNEL_ID, max_msgs=max(500, DISCORD_FETCH_LIMIT))
        print(f"[discord] parent messages: {len(parent_msgs)}")
        for m in parent_msgs:
            vids.extend(_extract_video_urls_from_message(m))
    except Exception as e:
        print(f"[discord] parent fetch error: {e}")

    # 2) Threads under the channel (active + archived)
    try:
        tids = _list_thread_ids(DISCORD_CHANNEL_ID, cap=50)
        print(f"[discord] threads found: {len(tids)}")
        for tid in tids[:30]:  # safety cap
            msgs = _paginate_messages(tid, max_msgs=150)
            for m in msgs:
                vids.extend(_extract_video_urls_from_message(m))
    except Exception as e:
        print(f"[discord] thread fetch error: {e}")

    vids.sort(key=lambda x: (x.get("created_at") or ""), reverse=True)
    print(f"[discord] total video candidates: {len(vids)}")
    _discord_cache["items"] = vids
    _discord_cache["ts"] = now
    return vids[:max_items]


# ----------------- Flask UI -----------------
app = Flask(__name__)

BASE_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>ESPN Fantasy – Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/@picocss/pico@2.0.6/css/pico.min.css" rel="stylesheet">
    <link rel="preload" as="image" href="{{ url_for('static', filename='smirnoff.webp') }}">
    <style>
      :root { --spacing:.9rem; }
      .grid { display:grid; gap:var(--spacing); grid-template-columns:1fr; }
      .card { border:1px solid var(--muted-border-color); border-radius:14px; padding:1rem; }
      header { margin-bottom:1rem; }
      table { width:100%; }
      .center{text-align:center;} .small{color:var(--muted-color); font-size:.9rem;}
      .pill{display:inline-block;padding:.15rem .5rem;border-radius:999px;border:1px solid var(--muted-border-color);font-size:.85rem;}
      .spacer{height:.5rem;}
      .tick{color:green;} .clock{color:#a96b00;}
      /* Storylines bar */
      .story-row { display:grid; gap:.6rem; grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .story { min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
      .badge { display:inline-flex; align-items:center; gap:.35rem; }
      .team-logo { width:20px; height:20px; object-fit:cover; border-radius:50%; vertical-align:-3px; }

      /* Smirnoff Standings (COMPACT) */
      .chart { display:flex; flex-direction:column; gap:.5rem; } /* tighter vertical gap */
      .row { display:grid; grid-template-columns: 240px 1fr; align-items:center; gap:10px; min-height:72px; }
      .team { display:flex; align-items:center; gap:.5rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
      .team-lg { width:20px; height:20px; object-fit:cover; border-radius:50%; }
      .bottles { display:flex; flex-wrap:wrap; gap:6px 6px; min-height:48px; align-items:center; }
      .bottle { position:relative; width:64px; height:64px; display:inline-flex; align-items:center; justify-content:center; cursor:help; }
      .bottle-img { width:100%; height:100%; object-fit:contain; }
      .bottle .tip { position:absolute; bottom:110%; left:50%; transform:translateX(-50%); background:#111; color:#fff; padding:.3rem .45rem; border-radius:8px; font-size:.75rem; white-space:nowrap; opacity:0; pointer-events:none; transition:opacity .12s; z-index:10; }
      .bottle:hover .tip { opacity:1; }

      /* Mobile: bottles appear in a row beneath the player's name */
      @media (max-width: 640px) {
      .row {
        grid-template-columns: 1fr;   /* stack content */
        align-items: start;
        gap: .35rem;
        min-height: unset;
      }
      .team {
       min-width: 0;                 /* allow name truncation */
      }
      .team strong {                  /* ensure name can ellipsize */
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .bottles {
        grid-column: 1 / -1;          /* take full width under the name */
        justify-content: flex-start;
        margin-top: .1rem;
        gap: 6px;                     /* keep a tidy row of bottles */
      }
      .bottle {                        /* slightly smaller bottles on mobile */
        width: 52px;
        height: 52px;
      }
      }
      
    /* Mobile-friendly tables */
    @media (max-width: 640px) {
    table {
        display: block;
        overflow-x: auto;
        white-space: nowrap;   /* prevent text wrapping in cells */
        -webkit-overflow-scrolling: touch; /* smooth scrolling on iOS */
    }
    thead {
        display: none;  /* hide table header if you want a simpler look */
    }
    tbody tr {
        display: block;
        margin-bottom: 1rem;
        border-bottom: 1px solid #ddd;
        padding-bottom: .5rem;
    }
    tbody td {
        display: flex;
        justify-content: space-between;
        padding: .25rem .5rem;
        font-size: .9rem;
    }
    tbody td:before {
        content: attr(data-label);
        font-weight: bold;
        margin-right: .5rem;
        color: #555;
    }
    }


      /* Scoreboard logos */
      .tm { display:flex; align-items:center; justify-content:center; gap:.35rem; }
      .tm img { width:18px; height:18px; object-fit:cover; border-radius:50%; }

      /* Discord video grid (SMALLER tiles, at top of page) */
      .vid-grid { display:grid; gap:8px; grid-template-columns: 1fr 1fr; }
      @media (min-width: 700px) { .vid-grid { grid-template-columns: 1fr 1fr 1fr; } }
      @media (min-width: 1100px) { .vid-grid { grid-template-columns: 1fr 1fr 1fr 1fr; } }
      .vid { display:flex; }
      .vid video {
        width: 100%;
        height: auto;          /* keep native aspect ratio */
        max-height: 220px;     /* keeps tiles small without cropping */
        border-radius: 10px;
        outline: 0;
        object-fit: contain;   /* 🔁 was 'cover' — this avoids top/bottom crop */
        object-position: center center;
        background: #000;      /* letterbox bars look intentional */
        box-shadow: 0 2px 8px rgba(0,0,0,.3);}
    /* Stories strip (top) */
    .stories {
    display: flex;
    flex-wrap: wrap;     /* allow wrapping to multiple rows */
    justify-content: center; /* center the row(s) */
    gap: 14px;
    padding: 6px 2px 8px;
    overflow: hidden;    /* hide scrollbar */
    }

        .story-card {
        position: relative; display: flex; flex-direction: column; align-items: center;
        min-width: 84px; max-width: 112px; text-align: center; scroll-snap-align: start;
        }
        .story-ava {
        width: 58px; height: 58px; border-radius: 50%;
        border: 3px solid #9ca3af; /* default gray */
        object-fit: cover; background: #111;
        }
        .story-ava.streak-win  { border-color: #16a34a; box-shadow: 0 0 0 2px rgba(22,163,74,.18); }
        .story-ava.streak-loss { border-color: #dc2626; box-shadow: 0 0 0 2px rgba(220,38,38,.18); }
        .story-ava.streak-none { border-color: #9ca3af; box-shadow: 0 0 0 2px rgba(156,163,175,.18); }

        .story-name { font-size: .82rem; margin-top: .35rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 112px; }
        .story-lines { margin-top: .15rem; line-height: 1.15; }
        .story-line { font-size: .75rem; color: var(--muted-color); white-space: nowrap; }

        .story-tip {
        position: absolute; bottom: 105%; left: 50%; transform: translateX(-50%);
        background: #111; color:#fff; padding:.45rem .6rem; border-radius:10px;
        font-size:.8rem; white-space: nowrap; opacity:0; pointer-events:none; transition:opacity .12s; z-index: 20;
        }
        .story-card:hover .story-tip { opacity: 1; }

    </style>
  </head>
  <body>
    <main class="container">
      <header>
        <h2>🧊 Ice Tracker 🧊</h2>
        <nav>
          <ul>
            <li><strong>{{ league_name }}</strong></li>
            <li class="small">Season {{ season }}</li>
          </ul>
        </nav>
        <a href='https://www.theknot.com/tk-media/images/e46d4b0f-1ae4-4b82-b046-dcbd9adcb6cd~cr_362.0.2026.1664?ordering=explicit'>Produced by GOAT Commish</a>
      </header>
      {% block body %}{% endblock %}
      <footer class="small">Built with Flask + <code>espn-api</code>. Not affiliated with ESPN.</footer>
    </main>
  </body>
</html>
"""

ONE_PAGE_HTML = """
{% extends "base.html" %}
{% block body %}

  {# --- TOP STORIES BAR (logos + name + record + PF + streak; colored rings; multiline; tooltip) --- #}
  {% if story_teams %}
  <article class="card" style="padding-bottom:.65rem;">
    <div class="stories" aria-label="Teams (sorted by record, PF)">
      {% for t in story_teams %}
        <div class="story-card">
          {% if t.logo %}
            <img class="story-ava {{ t.streak_class }}" src="{{ t.logo }}" alt="{{ t.team }}"/>
          {% else %}
            <div class="story-ava {{ t.streak_class }}"></div>
          {% endif %}
          <div class="story-name">{{ t.team }}</div>
          <div class="story-lines">
            <div class="story-line">{{ t.record }}</div>
            <div class="story-line">PF {{ '%.0f'|format(t.pf) }}</div>
            <div class="story-line">{{ t.streak }}</div>
          </div>
          <div class="story-tip">
            #{{ t.rank }} • {{ t.team }}<br/>
            Record: {{ t.record }}<br/>
            PF: {{ '%.2f'|format(t.pf) }} • PA: {{ '%.2f'|format(t.pa) }}<br/>
            Streak: {{ t.streak }}
          </div>
        </div>
      {% endfor %}
    </div>
  </article>
  {% endif %}

  {# --- DISCORD VIDEO WALL (moved to TOP, small tiles, no meta) --- #}
    {% if discord_videos %}
    <article class="card">
    <h4>Latest Smirnoff Clips</h4>
    <div class="vid-grid">
        {% for v in discord_videos[:4] %}
        <div class="vid">
            <video controls preload="none" playsinline loading="lazy"
                poster="{{ url_for('thumb') }}?url={{ v.url | urlencode }}">
            <source src="{{ v.url }}" type="video/mp4">
            </video>
        </div>
        {% endfor %}
    </div>
    <p class="small" style="margin-top:.5rem;">
        <a href="{{ url_for('videos') }}">View all clips →</a>
    </p>
    </article>
    {% endif %}

  {# --- ICE STANDINGS (now more compact) --- #}
  {% if ice_rows %}
  <article class="card">
    <h4>Ice Standings (Weeks 1–{{ last_week }})</h4>
    <section class="chart">
      {% for r in ice_rows %}
        <div class="row">
          <div class="team">
            {% if r.logo %}<img class="team-lg" src="{{ r.logo }}" alt=""/>{% endif %}
            <strong>{{ r.team }}</strong>
          </div>
          <div class="bottles">
            {% for ic in r.icons %}
              <div class="bottle" aria-label="{{ ic.tip }}">
                <img class="bottle-img"
                src="/static/smirnoff.webp"
                alt="Smirnoff Ice"
                width="64" height="64"
                loading="lazy"
                decoding="async"
                fetchpriority="low" />
                <div class="tip">{{ ic.tip }}</div>
              </div>
            {% endfor %}
          </div>
        </div>
      {% endfor %}
    </section>
  </article>
  {% endif %}

  <article class="card">
    <h4>Current Week Scoreboard (Week {{ current_week }})</h4>
    <p class="small">🕒 ongoing • ✅ completed</p>
    <table>
      <thead><tr><th class="center">Home</th><th class="center">Score</th><th class="center">Away</th><th class="center">Score</th><th class="center">Status</th></tr></thead>
      <tbody>
        {% for m in current_board %}
        <tr>
            <td class="center" data-label="Home">
            <span class="tm">
                {% if m.home_logo %}<img src="{{ m.home_logo }}" alt=""/>{% endif %}
                <span>{{ m.home }}</span>
            </span>
            </td>
            <td class="center" data-label="Score">{{ "%.2f"|format(m.home_score) }}</td>
            <td class="center" data-label="Away">
            <span class="tm">
                {% if m.away_logo %}<img src="{{ m.away_logo }}" alt=""/>{% endif %}
                <span>{{ m.away }}</span>
            </span>
            </td>
            <td class="center" data-label="Score">{{ "%.2f"|format(m.away_score) }}</td>
            <td class="center" data-label="Status">
            {% if m.ongoing %}
                <span class="clock">🕒</span>
            {% else %}
                <span class="tick">✅</span>
            {% endif %}
            </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </article>

  <article class="card">
    <h4>League Standings</h4>
    <table>
        <thead>
            <tr>
            <th>#</th>
            <th>Team</th>
            <th>W-L-T</th>
            <th>PF</th>
            <th>PA</th>
            </tr>
        </thead>
        <tbody>
            {% for row in standings %}
            <tr>
            <td data-label="#">{{ loop.index }}</td>
            <td data-label="Team">
                {% if row.logo %}<img class="team-logo" src="{{ row.logo }}" alt=""/>{% endif %}
                {{ row.team }}
            </td>
            <td data-label="W-L-T">
                {{ row.wins }}-{{ row.losses }}{% if row.ties and row.ties>0 %}-{{ row.ties }}{% endif %}
            </td>
            <td data-label="PF">{{ "%.2f"|format(row.pf) }}</td>
            <td data-label="PA">{{ "%.2f"|format(row.pa) }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
  </article>

{% endblock %}
"""

app = Flask(__name__)
app.jinja_loader = DictLoader({"base.html": BASE_HTML})

@app.route("/")
def dashboard():
    league = get_league()
    last_week = last_completed_week(league)

    # Safe Discord fetch so the route never crashes
    try:
        clips = fetch_discord_videos()
        # Debug preview (optional): print where clips came from
        for i, c in enumerate(clips[:5], 1):
            print(f"[discord] clip{i}: where={c.get('where')} file={c.get('filename')} url={c.get('url')}")
    except Exception as e:
        print(f"[discord] fetch error: {e}")
        clips = []

    print(f"[discord] showing {len(clips)} clip(s) on the page")

    return render_template_string(
        ONE_PAGE_HTML,
        league_name=league.settings.name,
        season=SEASON,
        current_week=league.current_week,
        last_week=last_week,
        standings=overall_standings(league),
        current_board=scoreboard_with_status(league),
        ice_rows=ice_tracker_season_to_date(league),
        iceking=ice_king(league),
        streaks=streak_tracker(league),
        topdog=top_dog_last_year(),
        discord_videos=clips,
        story_teams=stories_sorted_teams(league),  # <-- NEW
    )

@app.route("/videos")
def videos():
    try:
        clips = fetch_discord_videos(max_items=20)  # show more here
    except Exception as e:
        print(f"[discord] fetch error: {e}")
        clips = []

    return render_template_string(
        """
        {% extends "base.html" %}
        {% block body %}
        <article class="card">
          <h4>Smirnoff Video Wall (Full)</h4>
          <div class="vid-grid">
            {% for v in discord_videos %}
              <div class="vid">
                <video controls preload="none" playsinline loading="lazy">
                  <source src="{{ v.url }}" type="video/mp4"/>
                  Your browser can’t play this video.
                </video>
              </div>
            {% endfor %}
          </div>
        </article>
        {% endblock %}
        """,
        discord_videos=clips,
    )

@app.route("/thumb")
def thumb():
    vurl = (request.args.get("url") or "").strip()
    if not vurl:
        return redirect(url_for("static", filename="smirnoff.png"), code=302)

    out_path = _thumb_path_for(vurl)
    if out_path.exists():
        return send_file(out_path, mimetype="image/jpeg", max_age=7*24*3600)

    if not _ffmpeg_available():
        print("[thumb] ffmpeg not found; returning placeholder")
        return redirect(url_for("static", filename="smirnoff.png"), code=302)

    # Try at 3s (avoid black intro), then 1s as a fallback
    if _try_render_thumb(vurl, out_path, "00:00:03.000") or _try_render_thumb(vurl, out_path, "00:00:01.000"):
        return send_file(out_path, mimetype="image/jpeg", max_age=7*24*3600)

    print("[thumb] giving up; returning placeholder")
    return redirect(url_for("static", filename="smirnoff.png"), code=302)

if __name__ == "__main__":
    app.run(debug=True)
