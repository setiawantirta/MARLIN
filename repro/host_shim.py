"""A stand-in for the hosted connector, so the study's own parsing code can run.

`stages/legacy/stage01/01_fetch_opentargets_civic.py` and `03_fetch_string_network.py`
each begin with `host.mcp(...)` calls and then continue with a long block of parsing
and seed-selection logic - the logic that actually decided the paper's 186 seed genes.

Reimplementing that parsing in a notebook would risk diverging from what produced the
published table, and the divergence would be silent. Instead this shim answers the
three connector methods those scripts call from data already retrieved over the public
APIs by `repro.fetch_sources`, so the scripts execute unmodified:

    from repro.fetch_sources import fetch_open_targets, fetch_civic, fetch_string
    from repro.host_shim import HostShim, run_legacy

    shim = HostShim(open_targets=ot, civic=cv, string=net)
    run_legacy("stages/legacy/stage01/03_fetch_string_network.py", shim)

The shim refuses any method it was not given data for, rather than returning an empty
result that would quietly shrink the seed set.
"""
from __future__ import annotations

import os
import runpy


class HostShim:
    """Answers host.mcp() for the three methods stage 01 uses, from cached payloads."""

    def __init__(self, open_targets=None, civic=None, string=None):
        self._ot = open_targets
        self._civic = civic
        self._string = string

    def mcp(self, server, method, **kw):
        if method == "open_targets_graphql":
            if self._ot is None:
                raise RuntimeError("HostShim has no Open Targets payload; call "
                                   "fetch_sources.fetch_open_targets() first")
            did = kw.get("variables", {}).get("id")
            if did not in self._ot:
                raise KeyError(f"no cached Open Targets payload for {did!r}; the script asked "
                               f"for a disease that was not fetched (have: {sorted(self._ot)})")
            blk = self._ot[did]
            # Rebuild the GraphQL envelope the script unwraps.
            return {"data": {"disease": {
                "id": did, "name": blk["name"],
                "associatedTargets": {"count": blk["count"], "rows": blk["rows"]}}}}

        if method == "civic_search_evidence":
            if self._civic is None:
                raise RuntimeError("HostShim has no CIViC payload; call "
                                   "fetch_sources.fetch_civic() first")
            dn = kw.get("disease_name")
            if dn not in self._civic:
                raise KeyError(f"no cached CIViC payload for {dn!r} "
                               f"(have: {sorted(self._civic)})")
            return self._civic[dn]

        if method == "get_string_network":
            if self._string is None:
                raise RuntimeError("HostShim has no STRING payload; call "
                                   "fetch_sources.fetch_string() first")
            asked = kw.get("symbols") or []
            have = {n["name"] for n in self._string["nodes"]}
            missing = [s for s in asked if s not in have]
            if missing:
                raise ValueError(
                    f"the cached STRING network covers {len(have)} symbols but the script asked "
                    f"for {len(asked)}, {len(missing)} of which are absent (e.g. {missing[:5]}). "
                    "Re-run fetch_string() with the symbol list the script built.")
            return self._string

        raise NotImplementedError(
            f"HostShim does not implement {server}/{method}. The retrieval stages use only "
            "open_targets_graphql, civic_search_evidence and get_string_network; a new method "
            "here means a stage was changed and this shim needs extending.")


def run_legacy(script, shim, workdir=None):
    """Execute a legacy stage script with `host` bound to the shim.

    The script runs with its own globals, exactly as the stage runner invokes it, so
    its outputs land in `workdir` under the names the next step expects.
    """
    prev = os.getcwd()
    if workdir:
        os.makedirs(workdir, exist_ok=True)
        os.chdir(workdir)
    try:
        return runpy.run_path(script, init_globals={"host": shim}, run_name="__main__")
    finally:
        os.chdir(prev)
