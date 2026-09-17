"""Build provenance for the AI Telemetry SDK — the durable repo ↔ app_id binding.

Drop-in module for ``Financial_SDK/financial_rag_project/user-monitor-sdk/ai_telemetry/``.

Everything else in the correlator infers which repository a running application
came from. This makes the application *state* it: the deployed artifact reports
its own repo and commit on every event, so the binding is ground truth instead of
a guess, and a finding can name the exact commit that shipped the defect.

Wiring, in two steps:

1. Stamp the build. In CI, after checkout::

       docker build \
         --build-arg ZENDATA_REPO="$GITHUB_REPOSITORY" \
         --build-arg ZENDATA_COMMIT="$GITHUB_SHA" ...

   and in the Dockerfile::

       ARG ZENDATA_REPO
       ARG ZENDATA_COMMIT
       ENV ZENDATA_REPO=$ZENDATA_REPO ZENDATA_COMMIT=$ZENDATA_COMMIT

2. Attach it in ``ai_telemetry/client.py``. Add the three fields to the END of
   ``TelemetryEvent``::

       repo: Optional[str] = None
       commit_sha: Optional[str] = None
       build_id: Optional[str] = None

   append them to the END of the CSV column list (see WHY below), and in
   ``capture()`` alongside the other always-on signal fields::

       from ai_telemetry.provenance import current_provenance
       if not event.repo:
           event.repo, event.commit_sha, event.build_id = current_provenance()

   ``current_provenance()`` is resolved once and cached, so the per-event cost is
   a tuple lookup.

No new dependencies: the SDK's zero-dependency property is preserved.

────────────────────────────────────────────────────────────────────────────────
THREE RULES THIS MODULE EXISTS TO ENFORCE

The value it produces becomes ``sdk_provenance`` — the STRONGEST binding source
in ``zdcorrelate.binding.SOURCE_PRECEDENCE``, outranked only by a human override.
A wrong answer here is worse than no answer, because it is trusted more.

1. **Never attribute the SDK's own repository to the application.**
   The obvious fallback — ``git -C <this file's directory>`` — is wrong in every
   real deployment. The SDK is installed INTO the app (``pip install -e
   ./user-monitor-sdk``), so this file lives inside the *SDK's* checkout, whose
   ``remote.origin.url`` is the SDK repo. Reading it would confidently report
   that the running application came from the telemetry SDK's repository. Git is
   therefore run from the APPLICATION root (see ``_app_root``), never from
   ``__file__``, and any answer equal to this package's own repo is discarded.

2. **Git is opt-in.** Set ``ZENDATA_PROVENANCE_GIT=1`` to enable the local-git
   fallback. Default off, so anything arriving on an event is a deliberate build
   stamp rather than an accident of where the process happened to start. Absence
   is reported as absence; the correlator already degrades to ``code_literal``
   and labels the binding as inferred.

3. **A dirty tree is not the commit.** Images are commonly built with ``COPY . .``
   from a working tree, not from a pristine checkout of HEAD. Reporting HEAD's
   SHA for a modified tree is a false claim of exactly the kind this system
   exists to prevent, so ``git describe --always --dirty`` is used and the
   ``-dirty`` suffix travels all the way into the report.
"""
from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

# Environment variables, in precedence order. The first group is what Zendata
# asks customers to set; the rest are read automatically from the common CI
# systems so most pipelines need no configuration at all.
_REPO_VARS = (
    "ZENDATA_REPO",
    "GITHUB_REPOSITORY",        # GitHub Actions
    "CI_PROJECT_PATH",          # GitLab CI
    "BITBUCKET_REPO_FULL_NAME",  # Bitbucket Pipelines
    "BUILD_REPOSITORY_NAME",    # Azure Pipelines
)
_COMMIT_VARS = (
    "ZENDATA_COMMIT",
    "GITHUB_SHA",
    "CI_COMMIT_SHA",
    "BITBUCKET_COMMIT",
    "BUILD_SOURCEVERSION",
    "SOURCE_COMMIT",            # Docker Hub autobuild
)
_BUILD_VARS = (
    "ZENDATA_BUILD_ID",
    "GITHUB_RUN_ID",
    "CI_PIPELINE_ID",
    "BITBUCKET_BUILD_NUMBER",
    "BUILD_BUILDID",
)

# Where to look for the APPLICATION's checkout when the git fallback is enabled.
_APP_ROOT_VAR = "ZENDATA_APP_ROOT"
_GIT_ENABLE_VAR = "ZENDATA_PROVENANCE_GIT"


def _first_env(names) -> str:
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def normalize_repo(value: str) -> str:
    """Reduce a repo URL or slug to ``owner/name``.

    Must stay behaviourally identical to ``zdcorrelate.binding.normalize_repo``
    — the two sides of the join have to agree on the key or the binding silently
    fails to match. ``tests/test_binding.py::test_sdk_provenance_matches_the_loader_key``
    pins the two implementations together.
    """
    v = (value or "").strip().rstrip("/")
    if not v:
        return ""
    if v.endswith(".git"):
        v = v[:-4]
    # `break` matters: without it "git+https://host/o/n" strips "git+https://"
    # and then re-tests the remainder against the shorter prefixes.
    for prefix in ("https://", "http://", "ssh://", "git+https://", "git://"):
        if v.startswith(prefix):
            v = v[len(prefix):]
            break
    # scp-style remote: git@host:owner/name  →  host/owner/name
    if "@" in v.split("/", 1)[0]:
        v = v.split("@", 1)[1]
    v = v.replace(":", "/")
    parts = [p for p in v.split("/") if p and p != "."]
    if not parts:
        return ""
    if len(parts) >= 3 and "." in parts[0]:      # host/owner/name
        parts = parts[-2:]
    elif len(parts) >= 2:
        parts = parts[-2:]
    return "/".join(parts).lower()


def _app_root() -> str:
    """The APPLICATION's directory — never this package's.

    Order: an explicit ``ZENDATA_APP_ROOT``, then the directory of the entry-point
    module (``__main__``), then the process working directory. Every one of these
    belongs to the application; none of them is derived from ``__file__``, which
    is the whole point (see rule 1 in the module docstring).
    """
    explicit = (os.environ.get(_APP_ROOT_VAR) or "").strip()
    if explicit:
        return explicit
    main = sys.modules.get("__main__")
    main_file = getattr(main, "__file__", None)
    if main_file:
        try:
            return str(Path(main_file).resolve().parent)
        except OSError:
            pass
    return os.getcwd()


def _git(cwd: str, *args: str) -> str:
    """Run one git command in ``cwd``. Never raises; "" means "could not tell"."""
    try:
        proc = subprocess.run(
            ["git", "-C", cwd, *args],
            capture_output=True, text=True, timeout=3, check=True)
        return proc.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError, OSError, ValueError):
        return ""


@lru_cache(maxsize=1)
def _own_repo() -> str:
    """The repo slug of the checkout this FILE lives in — the SDK's own repo.

    Computed only so it can be REJECTED as an answer for the application.
    """
    try:
        here = str(Path(__file__).resolve().parent)
    except OSError:
        return ""
    return normalize_repo(_git(here, "config", "--get", "remote.origin.url"))


def _git_fallback() -> tuple[str, str]:
    """``(repo, commit)`` read from the application's local checkout, or ``("", "")``.

    Disabled unless ``ZENDATA_PROVENANCE_GIT`` is truthy, and discarded when it
    resolves to this package's own repository.
    """
    if (os.environ.get(_GIT_ENABLE_VAR) or "").strip().lower() not in ("1", "true", "yes", "on"):
        return "", ""

    root = _app_root()
    repo = normalize_repo(_git(root, "config", "--get", "remote.origin.url"))
    if not repo:
        return "", ""
    if repo == _own_repo():
        # The process is running from inside the SDK's own checkout (a test
        # harness, or an app whose cwd is the vendored SDK). Reporting this would
        # name the telemetry SDK as the application's source repository.
        return "", ""
    # --dirty: the image may have been built from a modified tree; see rule 3.
    commit = _git(root, "describe", "--always", "--dirty", "--abbrev=40")
    return repo, commit


@lru_cache(maxsize=1)
def current_provenance() -> tuple[str | None, str | None, str | None]:
    """Return ``(repo, commit_sha, build_id)`` for this deployment.

    Any component may be None. A None repo is not a failure — it means the build
    was not stamped, and the correlator falls back to static app_id detection and
    reports the binding as inferred rather than reported.
    """
    repo = normalize_repo(_first_env(_REPO_VARS))
    commit = _first_env(_COMMIT_VARS)
    build = _first_env(_BUILD_VARS)

    if not repo or not commit:
        git_repo, git_commit = _git_fallback()
        # Only adopt the git repo when the env named none. A stamped repo with an
        # unstamped commit must not silently acquire a commit from a DIFFERENT
        # repository's checkout.
        if not repo and git_repo:
            repo, commit = git_repo, (commit or git_commit)
        elif repo and not commit and git_repo == repo:
            commit = git_commit

    return (repo or None, (commit or None), (build or None))


def is_dirty() -> bool:
    """True when the reported commit came from a modified working tree."""
    _, commit, _ = current_provenance()
    return bool(commit and commit.endswith("-dirty"))


def provenance_dict() -> dict:
    """Provenance as a dict, for merging into an event or a health endpoint."""
    repo, commit, build = current_provenance()
    return {k: v for k, v in
            {"repo": repo, "commit_sha": commit, "build_id": build}.items() if v}


def describe() -> str:
    """One-line human summary — useful in a startup log line and in support."""
    repo, commit, build = current_provenance()
    if not repo:
        return ("build provenance NOT stamped — set ZENDATA_REPO/ZENDATA_COMMIT at "
                "build time so findings can be tied to the repository and commit")
    parts = [repo]
    if commit:
        parts.append(f"@{commit[:12]}")
    if build:
        parts.append(f"(build {build})")
    if is_dirty():
        parts.append("[DIRTY WORKING TREE — the commit named is not exactly what shipped]")
    return "build provenance: " + " ".join(parts)


if __name__ == "__main__":
    print(describe())
    print(provenance_dict())
