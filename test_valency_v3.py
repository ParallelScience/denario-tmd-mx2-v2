"""Generate paper_v3_citations_valency.tex for Iteration6 using the Valency
backend.

Re-runs Denario's per-section citation step for the Introduction and Methods
of Iteration6, side-by-side with the existing perplexity-derived
paper_v3_citations.tex (which is left untouched).

First run will trigger the Valency OAuth flow in your browser; subsequent
runs reuse ~/.denario/valency_token.json.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

DENARIO = Path("/scratch/scratch-aiscientist/Denario")
ITER = Path("/scratch/scratch-aiscientist/test/denario-tmd-mx2-v2/Iteration6")
PAPER_OUT = ITER / "paper_output"
TEMP = PAPER_OUT / "temp"

# Load env for GOOGLE_API_KEY etc.
for env_path in [
    Path("/scratch/scratch-aiscientist/parallelscience/denario-scientists/.env"),
    DENARIO / ".env",
]:
    if env_path.exists():
        load_dotenv(env_path, override=False)

sys.path.insert(0, str(DENARIO))

from denario.key_manager import KeyManager
from denario.langgraph_agents.citation_backends import get_citation_backend


def _extract_body(tex: str) -> tuple[str, str, str]:
    """Split a standalone .tex into (preamble, body, tail)."""
    begin = tex.find("\\begin{document}")
    end = tex.find("\\end{document}")
    if begin == -1 or end == -1:
        return "", tex, ""
    return (
        tex[: begin + len("\\begin{document}")],
        tex[begin + len("\\begin{document}") : end],
        tex[end:],
    )


def make_state(work_dir: Path, params_yaml: Path) -> dict:
    import yaml
    work_dir.mkdir(parents=True, exist_ok=True)
    keys = KeyManager()
    keys.get_keys_from_env()
    if not keys.GEMINI and not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit(
            "GOOGLE_API_KEY not set — needed by the citation_inserter LLM. "
            "Source a .env that defines GOOGLE_API_KEY before running."
        )
    params = yaml.safe_load(params_yaml.read_text())
    params.setdefault("Citations", {})
    params["Citations"]["backend"] = "valency"
    params["Citations"].setdefault(
        "citation_inserter", {"model": "gemini-2.5-flash", "temperature": 0.1}
    )
    return {
        "system": {
            "keys": keys,
            "params": params,
            "module_path": str(work_dir),
            "f_stream": str(work_dir / "valency_run.log"),
            "LLM_calls": str(work_dir / "LLM_calls.txt"),
            "costs_file": str(work_dir / "costs.txt"),
            "tokens": {"i": 0, "o": 0, "ti": 0, "to": 0},
            "costs": {"i": 0.0, "o": 0.0, "ci": 0.0, "co": 0.0},
        }
    }


def run_section(state, section: str) -> tuple[str, str]:
    src = TEMP / f"{section}.tex"
    out_tex = TEMP / f"{section}_w_citations_valency.tex"
    out_bib = TEMP / f"{section}_valency.bib"
    print(f"[{section}] reading {src.name}")
    text = src.read_text()
    backend = get_citation_backend("valency")
    t0 = time.time()
    result = backend(text, state)
    print(f"[{section}] backend done in {time.time() - t0:.1f}s; "
          f"{len(result.papers)} unique papers cited")
    out_tex.write_text(result.text)
    out_bib.write_text(result.bibtex)
    return result.text, result.bibtex


def main():
    work_dir = PAPER_OUT / "valency_run"
    state = make_state(work_dir, ITER.parent / "params.yaml")
    intro_tex, intro_bib = run_section(state, "Introduction")
    methods_tex, methods_bib = run_section(state, "Methods")

    # Splice into paper_v2_no_citations.tex paragraph-by-paragraph
    # (whitespace between paragraphs differs between per-section temp files
    # and the assembled paper, so body-level replace doesn't match).
    v2 = (PAPER_OUT / "paper_v2_no_citations.tex").read_text()
    intro_src = (TEMP / "Introduction.tex").read_text()
    methods_src = (TEMP / "Methods.tex").read_text()

    def _splice(doc, src_full, annotated_full, label):
        _, src_body, _ = _extract_body(src_full)
        _, ann_body, _ = _extract_body(annotated_full)
        src_paras = [p for p in src_body.strip().split("\n\n") if p.strip()]
        ann_paras = [p for p in ann_body.strip().split("\n\n") if p.strip()]
        if len(src_paras) != len(ann_paras):
            print(f"WARNING [{label}]: paragraph counts differ "
                  f"({len(src_paras)} src vs {len(ann_paras)} annotated); skipping splice")
            return doc, 0
        misses = 0
        for src_p, ann_p in zip(src_paras, ann_paras):
            if src_p in doc:
                doc = doc.replace(src_p, ann_p, 1)
            else:
                misses += 1
        if misses:
            print(f"WARNING [{label}]: {misses}/{len(src_paras)} paragraphs not found verbatim")
        return doc, len(src_paras) - misses

    v2, intro_hits = _splice(v2, intro_src, intro_tex, "Introduction")
    v2, methods_hits = _splice(v2, methods_src, methods_tex, "Methods")
    print(f"spliced paragraphs: Introduction={intro_hits}, Methods={methods_hits}")

    # Point the .tex at the Valency-only bib so we don't clobber the
    # existing perplexity bibliography.bib when compiling.
    v2 = v2.replace("\\bibliography{bibliography}", "\\bibliography{bibliography_valency}")
    out_paper = PAPER_OUT / "paper_v3_citations_valency.tex"
    out_paper.write_text(v2)
    out_bib = PAPER_OUT / "bibliography_valency.bib"
    out_bib.write_text((intro_bib + "\n\n" + methods_bib).strip() + "\n")

    print()
    print(f"wrote {out_paper}")
    print(f"wrote {out_bib}")
    print(f"per-section outputs: {TEMP}/Introduction_w_citations_valency.tex, "
          f"{TEMP}/Methods_w_citations_valency.tex")


if __name__ == "__main__":
    main()
