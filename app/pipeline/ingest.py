from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from app.config import DATA_DIR, RUNS_DIR

CACHE_DIR = DATA_DIR / "cache"


@dataclass
class MediaAsset:
    path: Path
    url: str
    platform: str
    kind: str
    duration_s: float


_YOUTUBE_RE = re.compile(r"(youtube\.com|youtu\.be)", re.I)
_SHORT_RE = re.compile(r"youtube\.com/shorts/|/shorts/", re.I)
_IG_RE = re.compile(r"(instagram\.com|instagr\.am)", re.I)
_IG_REEL_PATH = re.compile(r"/(?:share/)?(?:reel|reels)/", re.I)
_IG_STORY_PATH = re.compile(r"/stories/", re.I)
_IG_LIVE_PATH = re.compile(r"/live(?:/|$|\?)", re.I)
_IG_SHORTCODE = re.compile(
    r"(?:instagram\.com|instagr\.am)/(?:share/)?(?:reel|reels|p|tv)/([A-Za-z0-9_-]+)",
    re.I,
)
_YT_LIVE_RE = re.compile(r"youtube\.com/live|/live\b|livestream", re.I)
_LIVE_RE = _YT_LIVE_RE  # used by YouTube download sample-window
_YT_ID = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:shorts/|watch\?v=|embed/|live/))([A-Za-z0-9_-]{11})"
)
_IG_STRIP_QS = frozenset({"igsh", "igshid"})


def is_instagram_url(url: str | None) -> bool:
    return bool(url and _IG_RE.search(url))


def is_live_url(url: str | None) -> bool:
    """YouTube live only. Instagram Live is out of v1 (not the live-sessions surface)."""
    if not url or is_instagram_url(url):
        return False
    return bool(_YT_LIVE_RE.search(url))


def normalize_instagram_url(url: str) -> str:
    """Strip igsh / utm_* tracking; keep a canonical https URL for yt-dlp."""
    raw = (url or "").strip()
    if not raw:
        return raw
    parsed = urlparse(raw)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc or "www.instagram.com"
    query = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() not in _IG_STRIP_QS and not k.lower().startswith("utm_")
    ]
    path = parsed.path or "/"
    return urlunparse((scheme, netloc, path, "", urlencode(query), ""))


def instagram_shortcode(url: str | None) -> str | None:
    if not url:
        return None
    m = _IG_SHORTCODE.search(url)
    return m.group(1) if m else None


def infer_kind(url: str | None, kind_hint: str | None, duration_s: float | None = None) -> str:
    """Short vs VOD from duration. Live only when explicitly hinted (live sessions).

    Public POST /v1/detect never passes kind_hint; kind is inferred after probe.
    Reels are short until duration is known; a 55s `/p/` post is still short after probe.
    """
    if kind_hint == "live":
        return "live"
    if duration_s is not None:
        return "short" if duration_s <= 180 else "vod"
    if url and not is_instagram_url(url) and _YT_LIVE_RE.search(url):
        return "live"
    if url and (_SHORT_RE.search(url) or _IG_REEL_PATH.search(url)):
        return "short"
    return "vod"


def classify_url(url: str | None, kind_hint: str | None, local_path: str | None) -> tuple[str, str]:
    if local_path and not url:
        return "file", infer_kind(None, kind_hint)
    assert url, "url or local_path is required"
    if is_instagram_url(url):
        return "instagram", infer_kind(url, kind_hint)
    if _YOUTUBE_RE.search(url):
        return "youtube", infer_kind(url, kind_hint)
    return "other", infer_kind(url, kind_hint)


def youtube_id(url: str | None) -> str | None:
    if not url:
        return None
    m = _YT_ID.search(url)
    return m.group(1) if m else None


def probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


# ---------------------------------------------------------------------------
# Structured ingest error (API routes catch this and return 422)
# ---------------------------------------------------------------------------

class IngestError(RuntimeError):
    """Raised when media cannot be fetched.

    Attributes:
        code: Machine-readable error code, e.g. ``youtube_bot_check``,
              ``private_video``, ``geo_blocked``, ``unsupported_host``.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def classify_ytdlp_error(output: str) -> str:
    """Map yt-dlp stdout/stderr to a stable ingest error code."""
    low = (output or "").lower()
    if any(
        s in low
        for s in (
            "sign in to confirm",
            "not a bot",
            "http error 429",
            "too many requests",
            "confirm you’re not a bot",
            "confirm you're not a bot",
        )
    ):
        return "youtube_bot_check"
    if any(
        s in low
        for s in (
            "private video",
            "this video is private",
            "members-only",
            "members only",
            "join this channel to get access",
        )
    ):
        return "private_video"
    if any(
        s in low
        for s in (
            "not available in your country",
            "not available in your region",
            "uploader has not made this video available in your country",
        )
    ):
        return "geo_blocked"
    if "video unavailable" in low or "this video is not available" in low:
        return "private_video"
    if "http error 403" in low:
        return "youtube_bot_check"
    return "youtube_bot_check"


def classify_instagram_error(output: str) -> str:
    """Map yt-dlp Instagram failures to a stable ingest code."""
    low = (output or "").lower()
    if any(
        s in low
        for s in (
            "rate-limit",
            "rate limit",
            "http error 429",
            "too many requests",
            "please wait a few minutes",
            "please wait a few seconds",
        )
    ):
        return "rate_limited"
    if any(
        s in low
        for s in (
            "there is no video",
            "no video formats",
            "does not contain a video",
            "only images",
            "image-only",
            "photo post",
            "this post is an image",
        )
    ):
        return "unsupported_media"
    if any(
        s in low
        for s in (
            "private",
            "not available",
            "http error 404",
            "no media found",
            "requested content is not available",
            "deleted",
            "page not found",
        )
    ):
        return "private_or_missing"
    if any(
        s in low
        for s in (
            "login required",
            "login_required",
            "checkpoint",
            "challenge_required",
            "challenge required",
            "not logged in",
            "please log in",
            "http error 401",
            "http error 403",
            "csrf",
        )
    ):
        return "instagram_auth"
    return "instagram_auth"


def is_netscape_cookie_file(path: Path) -> bool:
    """True if *path* looks like a Netscape cookie jar (yt-dlp requirement)."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    head = text[:800]
    if "Netscape HTTP Cookie File" in head or "HTTP Cookie File" in head:
        return True
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if len(line.split("\t")) >= 7:
            return True
        return False
    return False


def ytdlp_available() -> bool:
    exe = _ytdlp_exe()
    return Path(exe).exists() or shutil.which(exe) is not None


def _cookie_status_from_args(args: list[str]) -> dict:
    if not args:
        return {"configured": False, "source": "none"}
    if args[0] == "--cookies":
        return {"configured": True, "source": "file", "path": args[1]}
    if args[0] == "--cookies-from-browser":
        return {"configured": True, "source": "browser", "browser": args[1]}
    return {"configured": False, "source": "none"}


def cookie_status() -> dict:
    """Both jars. YouTube and Instagram cookies are never interchangeable."""
    return {
        "youtube": _cookie_status_from_args(_cookie_args("youtube")),
        "instagram": _cookie_status_from_args(_cookie_args("instagram")),
    }


# ---------------------------------------------------------------------------
# yt-dlp executable resolution
# ---------------------------------------------------------------------------

def _ytdlp_exe() -> str:
    """Return the absolute path to yt-dlp, preferring the one next to the
    current Python interpreter (i.e. inside the active .venv).

    Falls back to PATH lookup, then bare ``"yt-dlp"`` so subprocess can
    raise a clear FileNotFoundError if nothing is found.
    """
    import sys

    scripts_dir = Path(sys.executable).parent
    for name in ("yt-dlp.exe", "yt-dlp"):
        candidate = scripts_dir / name
        if candidate.exists():
            return str(candidate)

    found = shutil.which("yt-dlp")
    return found or "yt-dlp"


# ---------------------------------------------------------------------------
# Cookie helpers
# ---------------------------------------------------------------------------

# Browsers to auto-try when no explicit cookie config is set.
_AUTO_BROWSERS = ["chrome", "edge", "firefox", "chromium"]


def _cookie_args(platform: str = "youtube") -> list[str]:
    """Return yt-dlp cookie flags. YouTube and Instagram jars are separate.

    YouTube priority: YTDLP_COOKIES → YTDLP_COOKIES_FROM_BROWSER → data/youtube.cookies.txt
    Instagram priority: INSTAGRAM_COOKIES / IG_COOKIES → browser env → data/instagram.cookies.txt
    """
    if platform == "instagram":
        cookies = (
            os.getenv("INSTAGRAM_COOKIES", "").strip()
            or os.getenv("IG_COOKIES", "").strip()
        )
        env_browser = (
            os.getenv("IG_COOKIES_FROM_BROWSER", "").strip()
            or os.getenv("YTDLP_COOKIES_FROM_BROWSER", "").strip()
        )
        default_jar = DATA_DIR / "instagram.cookies.txt"
        env_name = "INSTAGRAM_COOKIES"
    else:
        cookies = os.getenv("YTDLP_COOKIES", "").strip()
        env_browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "").strip()
        default_jar = DATA_DIR / "youtube.cookies.txt"
        env_name = "YTDLP_COOKIES"

    if cookies:
        p = Path(cookies)
        if p.exists() and p.stat().st_size > 10 and is_netscape_cookie_file(p):
            return ["--cookies", str(p)]
        print(
            f"WARN: {env_name}={cookies!r} missing, empty, or not a Netscape jar — ignoring"
        )

    if env_browser:
        return ["--cookies-from-browser", env_browser]

    if default_jar.exists() and default_jar.stat().st_size > 10:
        if is_netscape_cookie_file(default_jar):
            return ["--cookies", str(default_jar)]
        print(
            f"WARN: {default_jar} is not a Netscape cookie jar — skipping. "
            "Export cookies from your browser (yt-dlp wiki: Extractors → exporting cookies)."
        )

    return []


# ---------------------------------------------------------------------------
# yt-dlp command builder
# ---------------------------------------------------------------------------

def _ytdlp_base(dest: Path, url: str) -> list[str]:
    is_live = bool(_LIVE_RE.search(url))
    cmd = [
        _ytdlp_exe(),
        "--no-playlist",
        "-f",
        "bv*[height<=720]+ba/b[height<=720]/232+234/bv*+ba/best",
        "--merge-output-format",
        "mp4",
        "--retries",
        "3",
        "--fragment-retries",
        "3",
    ]
    if is_live:
        # For 24/7 continuous live broadcasts, capture a 5-minute (300s) sample window
        cmd += ["--downloader-args", "ffmpeg:-t 300"]
    node_exe = Path(r"C:\Users\venkat\AppData\Local\hermes\node\node.exe")
    if node_exe.exists():
        cmd += ["--js-runtimes", f"node:{node_exe}"]
    cmd += ["-o", str(dest), url]
    return cmd


# ---------------------------------------------------------------------------
# Download with retry strategy
# ---------------------------------------------------------------------------

def _bot_check_message() -> str:
    default_jar = DATA_DIR / "youtube.cookies.txt"
    return (
        "YouTube blocked the download (bot check / 429). "
        "Anonymous yt-dlp on this IP is not an SLA.\n\n"
        "To test detection tonight, supply authenticated cookies using ONE of:\n"
        "  Option A (recommended) — close the browser, then:\n"
        "    $env:YTDLP_COOKIES_FROM_BROWSER = 'chrome'   # or 'edge' / 'firefox'\n\n"
        "  Option B — export a Netscape cookie file:\n"
        f"    Place cookies at: {default_jar}\n"
        "    Guide: https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies\n\n"
        "  Option C — guaranteed path: upload an mp4:\n"
        "    --local /path/to/video.mp4   or   POST /v1/detect (multipart file)\n\n"
        "After one success the video id is cached under data/cache/ and the URL works offline."
    )


def _download_youtube(url: str, dest: Path) -> None:
    """Attempt to download *url* to *dest* using yt-dlp.

    Retry strategy
    --------------
    Phase A — use whatever credentials are configured:
      A1. cookie creds + android client
      A2. cookie creds + ios client
      A3. cookie creds + default client

    Phase B — auto-try browsers (also if Phase A creds were rejected):
      B1. --cookies-from-browser chrome + android
      B2. --cookies-from-browser edge   + android
      B3. --cookies-from-browser firefox + android

    Phase C — anonymous last-resort (almost always bot-checked)

    Raises
    ------
    IngestError with a structured code (youtube_bot_check, private_video, …).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    cookie = _cookie_args()
    last_output = ""

    def _run(extra_flags: list[str], label: str) -> bool:
        """Run yt-dlp with *extra_flags* inserted before the URL. Return True on success."""
        nonlocal last_output
        base = _ytdlp_base(dest, url)
        cmd = base[:-1] + extra_flags + [base[-1]]
        print(f"INFO: yt-dlp attempt [{label}]")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as e:
            raise IngestError(
                "yt_dlp_missing",
                "yt-dlp is not installed or not on PATH.",
            ) from e
        chunks: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            chunks.append(line)
        proc.wait()
        last_output = "".join(chunks)
        if proc.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return True
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False

    # Phase A: configured credentials
    if cookie:
        for client, label in [
            ("android,web", "configured-creds+android"),
            ("ios,web", "configured-creds+ios"),
            ("", "configured-creds+default"),
        ]:
            flags = list(cookie)
            if client:
                flags += ["--extractor-args", f"youtube:player_client={client}"]
            if _run(flags, label):
                return

    # Phase B: browser cookies (always worth trying if Phase A failed or was empty)
    tried_browser = None
    if cookie and cookie[0] == "--cookies-from-browser" and len(cookie) > 1:
        tried_browser = cookie[1]
    for browser in _AUTO_BROWSERS:
        if browser == tried_browser:
            continue
        flags = [
            "--cookies-from-browser",
            browser,
            "--extractor-args",
            "youtube:player_client=android,web",
        ]
        if _run(flags, f"auto-browser={browser}"):
            print(
                f"INFO: YouTube cookie auto-fetched from {browser}. "
                f"Set $env:YTDLP_COOKIES_FROM_BROWSER = '{browser}' to make this explicit."
            )
            return

    # Phase C: anonymous (last resort — will almost certainly fail on bot-protected videos)
    for client, label in [
        ("android,web", "anon+android"),
        ("ios,web", "anon+ios"),
        ("", "anon+default"),
    ]:
        flags = ["--extractor-args", f"youtube:player_client={client}"] if client else []
        if _run(flags, label):
            return

    code = classify_ytdlp_error(last_output)
    if code == "private_video":
        raise IngestError(
            code,
            "This video is private, members-only, or otherwise unavailable. "
            "Pass a local mp4 via --local or POST /v1/detect with multipart file.",
        )
    if code == "geo_blocked":
        raise IngestError(
            code,
            "This video is not available in this region. "
            "Pass a local mp4 via --local or POST /v1/detect with multipart file.",
        )
    raise IngestError(code, _bot_check_message())


def _ig_auth_message() -> str:
    jar = DATA_DIR / "instagram.cookies.txt"
    return (
        "Instagram blocked the download (login wall / checkpoint). "
        "Anonymous yt-dlp is not an SLA for Reels.\n\n"
        "Cookies are the default path (keep them separate from YouTube):\n"
        f"  Place a Netscape Instagram jar at: {jar}\n"
        "  Close the browser, then optionally:\n"
        "    $env:IG_COOKIES_FROM_BROWSER = 'chrome'   # or 'edge'\n\n"
        "Guaranteed path: upload the mp4:\n"
        "    --local /path/to/reel.mp4   or   POST /v1/detect (multipart file)\n\n"
        "After one success the shortcode is cached under data/cache/ig_<id>.mp4."
    )


def _download_instagram(url: str, dest: Path) -> None:
    """yt-dlp Instagram: cookies first, then browser, then anonymous."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cookie = _cookie_args("instagram")
    last_output = ""
    url = normalize_instagram_url(url)

    def _run(extra_flags: list[str], label: str) -> bool:
        nonlocal last_output
        cmd = [
            _ytdlp_exe(),
            "--no-playlist",
            "-f",
            "bv*+ba/b",
            "--merge-output-format",
            "mp4",
            "--retries",
            "3",
            "--fragment-retries",
            "3",
            "-o",
            str(dest),
        ]
        cmd = cmd + extra_flags + [url]
        print(f"INFO: yt-dlp IG attempt [{label}]")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as e:
            raise IngestError("yt_dlp_missing", "yt-dlp is not installed or not on PATH.") from e
        chunks: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            chunks.append(line)
        proc.wait()
        last_output = "".join(chunks)
        if proc.returncode == 0 and dest.exists() and dest.stat().st_size > 10_000:
            return True
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False

    if cookie:
        if _run(list(cookie), "ig-configured-cookies"):
            return

    tried_browser = None
    if cookie and cookie[0] == "--cookies-from-browser" and len(cookie) > 1:
        tried_browser = cookie[1]
    for browser in _AUTO_BROWSERS:
        if browser == tried_browser:
            continue
        if _run(["--cookies-from-browser", browser], f"ig-auto-browser={browser}"):
            print(
                f"INFO: Instagram cookies auto-fetched from {browser}. "
                f"Set $env:IG_COOKIES_FROM_BROWSER = '{browser}' to make this explicit."
            )
            return

    if _run([], "ig-anon"):
        return

    code = classify_instagram_error(last_output)
    if code == "unsupported_media":
        raise IngestError(
            code,
            "This Instagram post has no video (photo/carousel images only). "
            "Upload an mp4 via POST /v1/detect.",
        )
    if code == "private_or_missing":
        raise IngestError(
            code,
            "This Instagram post is private, deleted, or not found. "
            "Upload an mp4 via POST /v1/detect.",
        )
    if code == "rate_limited":
        raise IngestError(
            code,
            "Instagram rate-limited this IP. Wait, or upload the mp4 via POST /v1/detect.",
        )
    raise IngestError(code, _ig_auth_message())


# ---------------------------------------------------------------------------
# Public ingest entry point
# ---------------------------------------------------------------------------

def ingest(
    url: str | None,
    local_path: str | None,
    kind_hint: str | None,
    run_dir: Path,
) -> MediaAsset:
    run_dir.mkdir(parents=True, exist_ok=True)
    platform, kind = classify_url(url, kind_hint, local_path)

    if local_path:
        src = Path(local_path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(src)
        dest = run_dir / f"input{src.suffix or '.mp4'}"
        if dest.resolve() != src:
            dest.write_bytes(src.read_bytes())
        try:
            duration = probe_duration(dest)
        except Exception as e:
            raise IngestError(
                "probe_failed",
                f"ffprobe could not read duration from {dest.name}: {e}",
            ) from e
        kind = infer_kind(url, kind_hint, duration)
        return MediaAsset(dest, url or str(src), platform, kind, duration)

    if platform == "instagram":
        assert url
        if _IG_LIVE_PATH.search(url):
            raise IngestError(
                "unsupported",
                "Instagram Live is out of v1. Upload a recording via POST /v1/detect "
                "(multipart file).",
            )
        if _IG_STORY_PATH.search(url):
            raise IngestError(
                "unsupported",
                "Instagram Stories expire and are out of v1. Upload an mp4 via POST /v1/detect.",
            )
        dest = run_dir / "input.mp4"
        code = instagram_shortcode(url)
        cached = CACHE_DIR / f"ig_{code}.mp4" if code else None
        if cached and cached.exists() and cached.stat().st_size > 10_000:
            print(f"INFO: using cached IG download {cached}")
            shutil.copy2(cached, dest)
        else:
            _download_instagram(url, dest)
            if cached:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dest, cached)
        if not dest.exists() or dest.stat().st_size < 10_000:
            raise IngestError(
                "unsupported_media",
                "Instagram returned no video file. Upload an mp4 via POST /v1/detect.",
            )
        try:
            duration = probe_duration(dest)
        except Exception as e:
            raise IngestError(
                "unsupported_media",
                f"Downloaded Instagram media is not a playable video: {e}. "
                "Photo carousels are unsupported — upload an mp4.",
            ) from e
        kind = infer_kind(url, kind_hint, duration)
        return MediaAsset(dest, normalize_instagram_url(url), "instagram", kind, duration)

    if platform != "youtube":
        raise IngestError(
            "unsupported_host",
            f"Only YouTube URLs or local files are supported (got {platform}). "
            "Upload an mp4 via POST /v1/detect (multipart file).",
        )

    dest = run_dir / "input.mp4"
    vid = youtube_id(url)
    cached = CACHE_DIR / f"{vid}.mp4" if vid else None
    if cached and cached.exists() and cached.stat().st_size > 10_000:
        print(f"INFO: using cached download {cached}")
        shutil.copy2(cached, dest)
    else:
        _download_youtube(url or "", dest)
        if cached:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, cached)

    if not dest.exists():
        found = list(run_dir.glob("input*.mp4"))
        if not found:
            raise IngestError("youtube_bot_check", _bot_check_message())
        dest = found[0]
    try:
        duration = probe_duration(dest)
    except Exception as e:
        raise IngestError(
            "probe_failed",
            f"ffprobe could not read duration from downloaded file: {e}",
        ) from e
    kind = infer_kind(url, kind_hint, duration)
    return MediaAsset(dest, url or "", platform, kind, duration)