# Kimi K3 benchmark alignment

Status: `AUDITED — NO EXACT LUCEBOX OVERLAP`

The official Kimi K3 model card and Lucebox's current scored suites do not share
an exact benchmark name.  Claims must therefore keep official-card alignment
separate from Lucebox continuity testing.

## Official K3 card

The card reports these text benchmarks:

- reasoning and knowledge: GPQA Diamond, CritPt, AA-LCR and HLE-Full;
- coding: DeepSWE, ProgramBench, Terminal-Bench 2.1, FrontierSWE,
  SWE-Marathon, PostTrainBench, MLS-Bench-Lite, SciCode and Kimi Code Bench 2.0;
- agentic: BrowseComp, DeepSearchQA, ResearchRubrics, GDPval-AA v2,
  Toolathlon-Verified, MCPMark-Verified, MCP-Atlas, AutomationBench, JobBench,
  AA-Briefcase, Agents' Last Exam, APEX-Agents, OfficeQA Pro,
  SpreadsheetBench 2, OSWorld-Verified, OSWorld 2.0, SaaS-Bench,
  tau3-Banking, Harvey Lab-AA, CorpFin v2, Finance Agent v2 and Legal Research
  Bench.

The card also reports ten vision suites, which the present text-only GGUF path
cannot reproduce.  The Hugging Face repository publishes structured evaluation
metadata only for GPQA Diamond, HLE, DeepSWE and APEX-Agents.

Official K3 scores use maximum reasoning effort and temperature 1.0.  Single
step tasks such as GPQA use top-p 0.95.  This differs intentionally from the
frozen deterministic H23 product-mode gate, which disables thinking.

Source: <https://huggingface.co/moonshotai/Kimi-K3/blob/main/README.md#evaluation-results>

## Lucebox today

Lucebox has reusable scored paths for:

- HumanEval and HumanEval+;
- GSM8K;
- a ten-item Math500 smoke suite.

The SWE-shaped Lucebox fixture is a timing workload, not a patch-correctness
evaluation.  None of these names appears in the official K3 card.

## Chosen bridge

1. **Official-card bridge:** a preregistered 16-question GPQA-Diamond subset,
   native first and then the unchanged candidate, both with thinking enabled.
2. **Lucebox continuity companion:** the existing ten-item Math500 suite, kept
   under its existing scorer and labelled non-official.

GPQA is the smallest suitable official K3 benchmark because it is text-only,
multiple-choice and answer-key scored.  The official dataset is CC-BY-4.0 but
gated by a no-example-disclosure agreement.  Questions, answers, reasoning and
raw model transcripts must remain outside the repository and public gist.  Only
the pinned dataset revision, selection seed/indices, artifact hashes and
aggregate results may be published.

Pinned dataset revision inspected on 2026-08-17:
`633f5ee89ab8ad4522a9f850766b73f62147ffdd`.

The preregistered selection is Python `random.Random(20260817).sample(range(198),
16)`, sorted:

`8, 11, 15, 33, 35, 37, 41, 50, 86, 131, 140, 162, 164, 175, 178, 189`.

Dataset access is currently blocked until the Hugging Face account accepts the
official GPQA terms.  Do not substitute a public mirror, because that would
circumvent the benchmark owner's disclosure gate.

## Interpretation gate

- The 16-item subset is an alignment smoke test, not a reproduction of the
  card's 93.5 full-set score.
- Math500 measures continuity with Lucebox deployments, not official K3 parity.
- Native must pass enough selected items to form a meaningful denominator.
- Candidate quality is the fraction of native-correct items retained; report
  both raw accuracy and native-success retention.
- Do not publish GPQA item text, gold labels or item-level transcripts.
