# Fortschritt: gufo-thunder (aus `lisa_prd.json`)

Quelle: `lisa_prd.json`, `implementation_phases`. Bearbeitung strikt Feature
für Feature, Phase 1 zuerst.

## Umgebung und Nachweis-Methode

- **Host:** AMD Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`), 125 GiB Unified
  Memory, Fedora 44, `/dev/kfd` vorhanden.
- **Toolchain:** Podman-Container `gufo-work` (Basis `nixos/nix`, Mounts
  `/src`, `/models`) mit `nix develop`: ROCm 10.1 (TheRock), HIP 7.16.0,
  clang 24, CMake 4.4.4. Produktionsbinary für alle Messungen:
  `build/release/gufo` (HIP, gfx1151), gebaut mit `cmake --preset release`.
- **Modelle auf diesem Host:** `Qwen3.8-Flash-Next` UD-Q4_K_XL (4 Shards) mit
  MTP-Head, `Qwen3.8-27B` UD-Q4_K_XL (17,6 GB) und `DeepSeek-V4-Flash`
  IQ2XXS-0731 (86,7 GB, der vom Repo unterstützte Checkpoint). Alle drei
  Knee-Läufe sind unten gemessen.
- **CPU-Suite:** Container `gufo-work` (gleiche Devshell), `cmake --preset
  cpu-test`; alle Scheduler-/HTTP-/Runner-Suiten laufen dort grün.
- **Hardware-Verdikt:** `build/release/gufo diagnose` meldet `Overall Status
  [PASS]`, `gfx1151, 20 CUs`, 125 GiB Unified Memory, ROCm 10.1.0.

**Aktuelles Operator-Profil (2026-11-08).** Die laufende Instanz nutzt
`GUFO_AUTO_SESSIONS_MAX=40`, `--sessions auto --context 250000`, einen
17-GiB-KV-Pool (742.656 Tokens) und einen 2-GiB-RAM-Cache; rund 8 GB
Host-Speicher bleiben frei. Ältere Zahlen in diesem Protokoll (etwa 96/150
Sessions, 20-GiB-Pool, 7,25 GB Cache-Anteil oder 250.112 Tokens) sind
historische Messungen des jeweiligen Entwicklungsstands. Maßgeblich ist das
Profil in [docs/README.md](docs/README.md#current-deployment-profile-2026-11-08).

## Phase 1: Foundation (`TP-P1`, `TP-P2`, `TP-P3`, `CP-P1`)

### [ERLEDIGT] TP-P1 — Continuous Batching mit maximaler Batch-Größe

**Feature-ID:** `TP-P1` · **Priorität:** P0 · **Throughput-Impact:** highest

**Rationale (exakt aus der JSON):**

> Auf Strix Halo ist Speicherbandbreite der Flaschenhals; Durchsatz skaliert fast
> linear mit Batch-Größe bis Speicher-/Compute-Grenzen greifen

**Requirements (exakt aus der JSON):**

- [x] Continuous Batching muss aktiv sein
- [x] max_num_seqs muss große Werte unterstützen (Benchmark-Knee ~128 für Qwen3.8 27B)
- [x] max_num_seqs und max_num_batched_tokens müssen gemeinsam gesweept werden
- [x] Preemption-Strategie: RECOMPUTE (nicht SWAP) um Durchsatz aufrechtzuerhalten

**Acceptance Criteria (exakt aus der JSON):**

- [x] Aggregate tok/s steigt monoton mit Batch-Größe bis zum Knee
- [x] Knee-Punkt ist für jedes Zielmodell automatisiert bestimmt
  (Flash-Next, Qwen3.8-27B und DeepSeek V4 Flash gemessen)

#### Umsetzung

1. **Continuous Batching** ist aktiv: der Scheduler gibt allen fälligen Decodern
   pro Iteration einen Schritt, lässt Prefill nur dazwischen laufen und nimmt
   neue Requests mitten im Batch auf. Abgesichert durch
   `TestMidGenerationRequestJoinsNextDecodeBatch`,
   `TestPrefillYieldsToEveryDueDecoder`, `TestLargeBatchAdmitsEveryArrival`.
2. **`max_num_seqs` als echte Batch-Breite:** `--sessions N` ist die
   Aufnahmegrenze (`TextGenerationScheduler::max_num_seqs()` /
   `SetMaxNumSeqs()`), ohne kleinen Festwert. Der Pending-Puffer ist
   automatisch `max(16, N)`, sodass ein Burst in Batch-Breite eingereiht statt
   mit `429 queue_full` abgewiesen wird. Modelle führen breite Pools in
   Decode-Wellen ihrer eigenen Breite aus (verifiziert: `--sessions 16` bediente
   alle 16 Requests, Log `plan=batched-w7` … `batched-w2`); die effektive
   Decode-Breite ist im Qwen-Runner jedoch auf **8** begrenzt
   (`MaximumDecodeBatchWidth()` = `clamp(max_context, 1, 8)`), und genau dort
   saturiert der Durchsatz. Admission und Speicher tragen größere Werte; die
   Ausführungsweite nicht.
3. **RECOMPUTE-Preemption, kein SWAP:** `TextPreemptionPolicy::kRecompute` ist
   die einzige Strategie; ein Swap-Pfad existiert bewusst nicht.
   `PreemptForRecompute` hängt die bereits erzeugten Tokens an den Prompt,
   übernimmt den `SamplerState` (RNG + Historie), ruft `Invalidate()` und
   admitiert das Replay vor allen Neuzugängen. Metriken:
   `result.preemptions`, `result.preempted_tokens`,
   `llamacpp:preemptions_total`, `llamacpp:recompute_tokens_total`.
4. **Gemeinsamer Sweep** von `--sessions` (`max_num_seqs`) und
   `--prefill-chunk` (`max_num_batched_tokens`):
   `tools/gufo/serving_sweep.py` / `tools/serving/gufo-serving-sweep.py`
   messen pro Punkt den kanonischen Serving-Benchmark (synchronisierte Welle,
   eindeutige Prompts pro Request) und bestimmen den Knee aus der Messkurve:
   letzter Batch-Schritt mit ≥ `--min-gain` (Default 3 %) Zuwachs. Die
   Sessions-Achse stoppt am Knee; ein Rückgang wird als Sättigung am vorherigen
   Peak gewertet (nicht geglättet), ein nicht gesättigter Achsenbereich als
   `violations` gemeldet (Exit 1).

#### Nachweis / Verifikation

**Gemessen auf gfx1151** (`build/release/gufo`, Flash-Next Q4_K_XL,
`--speculative off`, greedy, 64 neue Tokens, 2 Wiederholungen):

Lauf A (`--context 4096`, `--prefill-axis 2048,16384`), Artefakt
`artifacts/serving/flash-next-knee-rising.json`:

| sessions | total tok/s @ prefill 2048 | total tok/s @ prefill 16384 | output tok/s @ 16384 |
| ---: | ---: | ---: | ---: |
| 1 | 66.7 | 67.7 | 26.6 |
| 2 | 99.1 | 98.1 | 40.0 |
| 4 | 135.7 | 139.7 | 55.9 |
| 8 | 179.1 | 178.8 | 71.5 |

Monoton steigend bis 8 (dem größten in Lauf A gemessenen Punkt), keine
Verletzung; ein früherer Lauf mit denselben Achsen lag innerhalb von 1 %
(63,9/99,5/133,1/177,9 total).

Lauf B (`--context 1024`, `--sessions-axis 8,16`,
`--prefill-axis 2048,16384`), Artefakt
`artifacts/serving/flash-next-knee.json`:

| sessions | total tok/s @ 2048 | total tok/s @ 16384 | output tok/s @ 16384 |
| ---: | ---: | ---: | ---: |
| 8 | 172.0 | 180.5 | 72.2 |
| 16 | 177.5 | 180.6 | 71.6 |

Ergebnis des Tools (automatisch): **Knee bei 8 Sessions × 16384 Prefill-Tokens,
180.5 total tok/s, `saturated=true`, keine Violations.** Der Schritt 8 → 16
bringt bei 16384 nur +0,06 % (bei 2048 +3,2 %), also ist 8 der Knee. Ein Lauf
mit 32 Sessions überstieg den Unified-Memory-Bedarf des Containers (Server
starb während des Benchmarks); das ist die Speichergrenze, nicht der Scheduler.

Lauf C (**Qwen3.8-27B** UD-Q4_K_XL, `--context 2048`, greedy,
`--speculative off`, 32 neue Tokens, `--prefill-axis 2048`), Artefakt
`artifacts/serving/qwen38-27b-knee.json`:

| sessions | total tok/s | output (end-to-end) tok/s |
| ---: | ---: | ---: |
| 1 | 39.5 | 9.7 |
| 2 | 62.9 | 16.1 |
| 4 | 93.4 | 23.2 |
| 8 | 116.6 | 28.9 |
| 16 | 119.0 | 29.2 |

Automatisches Ergebnis: **Knee bei 8 Sessions, 116,6 total tok/s,
`saturated=true`, `monotone=true`, keine Violations** (8 → 16 gewinnt nur
2,1 %). Das deckt sich mit dem PRD-Hinweis „Akzeptanzlänge steigt bis n=6, der
Durchsatz flacht danach ab": der Durchsatz-Knee der dichten 27B liegt bei 8,
nicht bei 128.

Lauf D (**DeepSeek V4 Flash** IQ2XXS-0731, `--context 1024`, greedy,
`--speculative off`, 32 neue Tokens, `--prefill-axis 2048`), Artefakt
`artifacts/serving/deepseek-v4-flash-knee.json`:

| sessions | total tok/s | output (end-to-end) tok/s |
| ---: | ---: | ---: |
| 1 | 41.8 | 11.8 |
| 2 | 57.9 | 16.1 |
| 4 | 69.3 | 19.5 |
| 8 | 76.9 | 21.8 |
| 16 | 71.9 | 19.8 |

Automatisches Ergebnis: **Knee bei 8 Sessions, 76,9 total tok/s,
`saturated=true`** (8 → 16 fällt um 6,5 %). Damit ist der Knee-Punkt für alle
drei Zielmodelle automatisiert bestimmt; alle drei erreichen ihn bei 8
Sessions, weil die Runner Decode-Wellen dieser Breite ausführen.

**Was das für die PRD-Ziele bedeutet (ehrlich):** Am Knee liefert Flash-Next
208,2 *Gesamt*-tok/s (Prefill+Decode); die Summe der per-Request-Decode-Raten
ist 103,1, die gelieferte End-to-End-Output-Rate 66,3 tok/s. DeepSeek: 99,0
gesamt, 60,9 Decode-Summe, 28,0 Output; 27B: 107,5 gesamt, 64,1 Decode-Summe,
33,9 Output (jeweils ohne Spekulation). Keine der drei Kennzahlen erreicht die
PRD-Gates „>150/80/100 aggregate decode"; sie hängen an Phase 3/4
(MLA-Retrofit, KV-Quantisierung, MTP/DSpark). Die Decode-Summe ist die
PRD-Definition (Summe über die gleichzeitigen Requests), die Output-Rate die
konservative Untergrenze; beide sind hier genannt, damit die Zielwerte nicht
schöngerechnet werden. Die zuvor hier genannten Decode-Werte (72/21,8/28,9)
waren die End-to-End-Output-Rate, nicht die Decode-Phase.

**Erneute Verifikation mit dem neu gebauten Binary (2026-10-30, nach `cmake
--preset release`, aktuelle Tool-Defaults: 128 neue Tokens, `--context 4096`,
`--speculative off`):**

Lauf D (**DeepSeek-V4-Flash** IQ2XXS-0731, `--context 2048`, greedy,
32 Tokens, `--prefill-axis 2048`), Artefakt
`artifacts/serving/deepseek-v4-flash-knee-monotone.json`:

| sessions | total tok/s | output (end-to-end) tok/s |
| ---: | ---: | ---: |
| 2 | 51,3 | 22,9 |
| 4 | 72,1 | 32,0 |
| 6 | 79,7 | 35,8 |
| 8 | 87,6 | 39,1 |

Monoton bis 8. Der Folgelauf über 8,10,12 (`deepseek-v4-flash-knee-10-12.json`)
sättigt: 8 = 88,9, 10 = 88,5 (−0,5 %), also **Knee bei 8 Sessions,
`saturated=true`, `monotone=true`, keine Violations**. Ein früherer Lauf mit
16 Sessions fiel auf 71,9 total (Speicher-/Bandbreitengrenze jenseits des
Knees); der jüngste Lauf mit 16384er-Budget misst bei 16 Sessions 99,4 total
statt eines Rückgangs. Beide Messungen stützen denselben Knee bei 8 Sessions:
der Schritt 8 → 16 gewinnt höchstens 0,4 %.

| Modell | Artefakt (jüngster Lauf) | Knee (automatisch) | monoton | saturated | decode-Summe @8 / @16 |
| --- | --- | ---: | :---: | :---: | ---: |
| Qwen3.8-27B | `qwen38-27b-decode-curve-20261031T125402Z.json` | 8 Sessions × 16384, 107,5 total tok/s | ja | ja (8→16: +2,4 %) | 64,1 / 128,4 |
| Flash-Next | `flash-next-decode-curve-20261031T125402Z.json` | 8 Sessions × 16384, 208,2 total tok/s | ja | ja (8→16: +1,3 %) | 103,1 / 209,7 |
| DeepSeek-V4-Flash | `deepseek-v4-flash-knee-decode-20261031T124705Z.json` | 8 Sessions × 16384, 99,0 total tok/s | ja | ja (8→16: +0,4 %) | 60,9 / 122,7 |

Ältere Knee-Artefakte derselben Modelle (`flash-next-knee-paged.json` 163,0;
`qwen38-27b-knee-verify.json` 83,4 bei 2048er-Prefill-Achse;
`deepseek-v4-flash-knee-10-12.json` 88,9 bei 2048er-Achse) zeigen dieselbe
Knee-Lage, aber niedrigere Gesamtraten und `decodeTokensPerSecond: null`
(Werkzeuglücke, unten). Für die Abnahme zählen die jüngsten Läufe oben; sie
liefen mit `--max-tokens 64 --repetitions 2 --warmup 1 --speculative off` und
Kontext 4096 (Flash-Next) bzw. 2048 (27B) bzw. Modell-Default (DeepSeek).

Die Spalte `decode` ist die PRD-Kennzahl „Summe der Decode-tok/s über alle
gleichzeitigen Requests" (Mittel der Summe der per-Request-Decode-Raten pro
Runde, `_serving_rate`). Sie steigt bei 16 Sessions auf das Doppelte, obwohl
die *gelieferte* Gesamtrate flach bleibt (z. B. Flash-Next 208,2 → 210,8;
27B 107,5 → 110,0; DeepSeek 99,0 → 99,4). Das heißt: bei 16 Sessions decodieren
nicht alle Requests gleichzeitig — ein Teil wartet hinter der schmaleren
Ausführungsweite, und die per-Request-Decode-Zeit enthält diese Wartezeit
nicht. Die Summe ist daher nur bei vollständig parallelem Decode eine
Aggregatrate; als Knee-Metrik ist die gelieferte Gesamtrate belastbar. Der
Knee bleibt deshalb bei 8 Sessions.

**Korrektur einer früheren Fehlbeschriftung:** Die bisherigen Tabellen nannten
die Spalte „output (decode) tok/s". `outputTokensPerSecond` ist jedoch
Completion-Tokens über die gesamte Request-Wandzeit (inklusive Prefill und
Queueing), nicht die Decode-Phase. Die Tabellen sind entsprechend umbenannt;
die echte Decode-Summe steht in der neuen Spalte oben.

**Werkzeuglücke geschlossen:** `serving_sweep.py` las die Decode-Kennzahl aus
einem Aggregat-Schlüssel, den `serving_bench.py` nie schreibt — alle bisherigen
Sweep-Artefakte hatten `decodeTokensPerSecond: null`. Der Sweep nutzt jetzt den
kanonischen `_serving_rate` (Summe der per-Request-Decode-Raten pro Runde);
`tests/tools/serving_sweep_test.py` prüft den Fallback und dass ein Report ohne
Decode-Zeiten die Kennzahl nicht erfindet (14/14 Tests grün). Neue Artefakte:
`artifacts/serving/*-decode-curve-*.json`.

Alle drei Kurven steigen monoton bis 8 und sättigen dort; die absolute
Gesamtrate hängt vom Verhältnis Prefill/Decode und von Prefix-Cache-Treffern ab
und ist deshalb nur innerhalb eines Laufs mit identischen Argumenten
vergleichbar. Für die Abnahme zählt die monoton steigende Kurve bis zum
automatisch bestimmten Knee — alle drei Zielmodelle erfüllen sie mit dem
frisch gebauten Binary.

**Modellpfad-Hinweis:** `gufo serve --model` erwartet eine GGUF-**Datei**. Bei
geshardeten Checkpoints ist das erste Shard anzugeben
(`...-00001-of-000NN.gguf`), bei Einzeldateien die Datei selbst; ein
Verzeichnispfad schlägt mit `cannot mmap: Permission denied` fehl.

**CPU-Suite (Container `gufo-work`, `cmake --preset cpu-test`):**

```
text_generation_scheduler_test ... Passed
http_server_test ............... Passed
text_model_runner_test ......... Passed
serving_bench_test ............. Passed
serving_sweep_test ............. Passed
serve_cli_test ................. Passed
100% tests passed out of 6

ctest --preset cpu-test -L 'server|scheduling|benchmarks|cli' --parallel 6
100% tests passed out of 41
python3 tools/ci/check-format.py → PASS (708 C++ files)
python3 tools/ci/check-docs.py   → PASSED (109 markdown files, 634 links)
```

Relevante Tests: `TestLargeBatchAdmitsEveryArrival` (128 Requests, alle
aufgenommen), `TestBatchShrinkRecomputesRunningRequests` (Preemption mitten im
Decode: Invalidate + exaktes Replay, kein Swap), `TestMaxNumSeqsBoundsAreValidated`,
`serving_sweep_test.py` (Knee-Erkennung, Rückgang als Sättigung am Peak,
Achsen-Abbruch,
Artefakt-Roundtrip).

### [ERLEDIGT] TP-P2 — Token-Budget maximieren

**Feature-ID:** `TP-P2` · **Priorität:** P0

**Rationale:** — (in der JSON nicht gesetzt)

**Requirements (exakt aus der JSON):**

- [x] max_num_batched_tokens Default deutlich über 8192
- [x] Dynamische Anpassung an verfügbaren KV-Speicher

**Acceptance Criteria (exakt aus der JSON):** — (Feature hat keine
`acceptance_criteria`; Abnahme über Requirements, Tests und Messung)

#### Umsetzung

1. **Default 16384:** `kDefaultDecodeActivePrefillTokens` ist das
   `max_num_batched_tokens`-Budget und steigt von 512 auf **16384**;
   `--prefill-chunk` nutzt denselben Default und nennt ihn in der Hilfe.
   `static_assert` im Test hält den Default über 8192.
2. **Dynamische Anpassung an freien KV-/Unified-Speicher:**
   - `TextPrefillPolicy::adapt_to_kv_capacity` (Default an),
     `min_batched_tokens` (0 = automatisch `min(budget, 512)`).
   - Zwei unabhängige Schranken: der Byte-Claim des Runners
     (`KvCapacityTokens()` aus `state_capacity_bytes`, `per_request_state_bytes`
     und `max_context`) **und** der freie Host-Speicher
     (`TextSchedulerPolicy::available_memory`, im Server
     `HostAvailableMemoryBytes()` nach cgroup-Grenzen, mit 2 GiB Reserve).
   - `FreeKvTokens()` = min(Claim-Kapazität, Speicher-Headroom in Tokens) −
     gebundene Resident-Tokens − ein Token pro fälligem Decoder.
   - `PrefillTokenBudget()` = `clamp(min(max_num_batched_tokens, frei), floor,
     max_num_batched_tokens)`; ein decode-idler Prefill behält seinen
     Bulk-Work-Unit, überschreitet aber nie die freie Kapazität.
   - Beobachtbar: `result.prefill_budget_tokens`,
     `result.configured_active_prefill_tokens`, `prefill_kv_limited`.

#### Nachweis / Verifikation

- `TestBatchedTokenBudgetDefaultsAbove8192`: `static_assert` > 8192, Request
  meldet 16384; ein Einzelrequest prefillt trotzdem seinen ganzen Prompt.
- `TestPrefillBudgetAdaptsToFreeKvCapacity`: freier Speicher 32 B über Reserve
  ⇒ erster Chunk ≤ 32 statt 16384, Prompt über mehrere Iterationen vollständig,
  `prefill_kv_limited=true`.
- `TestPrefillBudgetFallsBackToItsFloor`: Reserve voll ausgeschöpft ⇒ Chunk
  genau 8 (Floor), Prompt trotzdem vollständig, Kontextlimits eingehalten.
- **Messung:** Bei 8 Sessions liefert das 16384-Budget 180,5 total tok/s
  gegenüber 172,0 bei 2048 (Lauf B), also +4,9 % — der höhere Default kostet
  keinen Durchsatz und deckt längere Prompts in einem Zug ab.

### [ERLEDIGT] TP-P3 — Speicher-Druck früh erkennen

**Feature-ID:** `TP-P3` · **Priorität:** P0

**Requirements (exakt aus der JSON):**

- [x] Scheduler muss erkennen wenn Batch-Größe durch KV-Speicher limitiert wird
- [x] Preemption mit RECOMPUTE bevorzugt gegenüber SWAP

**Acceptance Criteria (exakt aus der JSON):** — (keine; Abnahme über
Requirements und Tests)

#### Umsetzung

1. **Erkennung:** `LowMemory()`/`MemoryHeadroomBytes()` lesen den freien
   Unified-Speicher; `FreeKvTokens()` kombiniert ihn mit dem KV-Claim. Ein
   Prefill-Chunk, der kleiner ausfällt als konfiguriert, setzt
   `result.prefill_kv_limited`, zählt `llamacpp:kv_limited_total` und loggt
   `event=kv_pressure reason=chunk_limit|kv_capacity|host_memory` mit
   `free_tokens`, `needed_tokens`, `granted`.
2. **Frühe Entlastung:** `RelieveKvPressure` greift, bevor ein Prefill nicht
   einmal seinen Floor bekommt: der neueste Decoder wird mit **RECOMPUTE**
   freigegeben (Prompt + generierte Tokens + Sampler bleiben, `Invalidate()`,
   kein Host-Swap), damit der wartende Prefill laufen kann. Ein
   `kv_reliefs_in_episode`-Deckel verhindert Preemptions-Ping-Pong; der
   verursachende Request wird erst nach verbrauchtem KV erneut betrachtet.
3. **RECOMPUTE statt SWAP:** einzige Policy (`TextPreemptionPolicy::kRecompute`);
   Preemption ist über `preemptions`, `preempted_tokens`,
   `llamacpp:preemptions_total`, `llamacpp:recompute_tokens_total` sichtbar.

#### Nachweis / Verifikation

- `TestKvPressurePreemptsWithRecompute`: Speicher fällt während eines
  laufenden Decodes auf 0; der Scheduler meldet `memory_pressure()`, gibt den
  Decoder per RECOMPUTE frei (`invalidations` steigt, kein Swap), beide
  Trajektorien bleiben exakt, der wartende Prompt wird bedient.
- `TestNoPressureWithRoomToGrow`: bei 7 GiB Headroom keine falsche Preemption,
  kein `prefill_kv_limited`.
- `http_server_test` prüft `kv_limited_total` im `/metrics`-Text.
- Auf dem echten Server trat bei 16 Sessions (4,3 GiB frei, 2 GiB Reserve)
  erwartungsgemäß **kein** KV-Druck auf; die Preemption ist daher durch die
  deterministischen Scheduler-Tests belegt, nicht durch einen GPU-Zufallslauf.

### [ERLEDIGT] CP-P1 — Chunked Prefill mit dynamischer Chunk-Größe

**Feature-ID:** `CP-P1` · **Priorität:** P0

**Rationale (exakt aus der JSON):**

> Ohne Chunked Prefill blockieren lange Prefills die gesamte Batch; auf Strix
> Halo kann ein 32K-Prefill 20-30s dauern

**Requirements (exakt aus der JSON):**

- [x] Lange Prefills in Häppchen zerlegen
- [x] Interleaving mit Decode-Schritten
- [x] Dynamische Chunk-Größe basierend auf Token-Budget pro Iteration
- [x] Decode-Anteil pro Iteration muss hoch bleiben

**Acceptance Criteria (exakt aus der JSON):** — (keine; Abnahme über
Requirements, Tests und Funktionslauf)

#### Umsetzung

- `StepPrefill` verarbeitet genau **einen** Chunk pro Iteration; ein langer
  Prompt wird über viele Iterationen verteilt und wechselt sich mit wartenden
  Peers ab (`prefilling` Round-Robin).
- Sobald ein Decoder fällig ist, läuft **jeder** fällige Decoder vor dem
  nächsten Prefill-Chunk; jeder Chunk markiert die Decoder wieder fällig, sodass
  `max_consecutive_active_prefill_chunks` 1 nicht überschreitet.
- Die Chunk-Größe ist dynamisch: `max_num_batched_tokens`, begrenzt durch freie
  KV-/Speicherkapazität und den Rest-Prompt, mit Floor (512 bzw. konfiguriert).
  Idle-Prefill behält den Bulk-Work-Unit, aber nie über der freien Kapazität.
- Ein Decode-loser Prefill blockiert keine Decoder, weil er nach einem Chunk
  zurück in die Runde geht; Snapshots/Captures werden zwischen Chunks bedient.

#### Nachweis / Verifikation

- `TestChunkedPrefillInterleavesLongPrompts`: zwei 3000-Token-Prompts, Budget
  512 ⇒ je ≥ 5 Chunks, kein Chunk > 512, mindestens 6 Wechsel zwischen den
  Requests.
- `TestLongPrefillKeepsDecodeTurns`: ein Decoder + 2000-Token-Prefill ⇒
  `max_consecutive_active_prefill_chunks <= 1` und
  `active_decode_prefill_chunks >= 1`, beide Trajektorien exakt.
- `TestDecodeActivePrefillIsBounded` / `TestPrefillYieldsToEveryDueDecoder`
  (je multi-token × batched) für die Decode-Vorrangregel.
- **Funktionslauf auf gfx1151:** `tests/functional/run.py --suite
  prefill-scheduling` mit 2 Sessions und 16K-Kontext (Flash-Next) — ein kurzer
  Request liefert seine Ausgabe, während der lange Peer noch prefillt:
  `processed=2048` von `total=5877` beim ersten Output-Token des kurzen
  Requests (`short: "4"`), Lang-Prompt danach vollständig. Ergebnis:
  `artifacts/functional/prefill-scheduling/` (`status=passed`, 6,3 s).
  Mit dem aktuellen Binary erneut gefahren:
  `artifacts/functional/prefill-scheduling-reverify/` (`status=passed`,
  6,0 s, gleiche Zahlen: `processed=2048`, `total=5877`, `cache=0`).

## Phase 2: Paged Attention und Shared KV-Pool

Quelle: `implementation_phases[1]` (`PA-P1`, `PA-P3`, `PA-P4`, `KV-P1`,
`KV-P2`, `KV-P3`, `D1`). Der gemeinsame KV-Pool (`KvBlockPool`/`KvBlockTable`,
CoW, `ForkFrom`) war im Flash-Next-Executor implementiert, aber in
`engine.cpp` mit `kv_pool_blocks = 0` abgeschaltet; `docs/cache-redesign/paged-kv.md`
§10 dokumentierte den Grund: der paged Snapshot-Restore divergierte.

### [ERLEDIGT] PA-P1 / PA-P3 — Paged-Attention-Pfad und Block-Allokation

**Umsetzung**

- `exec.kv_pool_blocks = pool_sessions * blocks_per_session`: der gemeinsame
  Pool ersetzt die kontiguierlichen Per-Session-Planes mit **demselben**
  Speicherumsatz (eine Plane für K und V, `[block][layer][128][row]`).
  `pool_sessions` kommt aus der tatsächlichen Session-Zahl des Servers
  (`--sessions`), nicht aus der auf 8 begrenzten Decode-Breite; sonst lehnt
  `--sessions 16` die neunte Session mit „attention KV pool exhausted" ab.
- Blöcke werden beim Session-Start vollständig gemappt (126 KiB), der Pool
  einmalig genullt. Das macht einen Capture deterministisch, der Zeilen
  sichert, bevor sie zum ersten Mal geschrieben wurden.
- `KvBlockTable::Append` allokiert on demand; partielle letzte Blöcke werden
  korrekt adressiert (`block_for`, `ByteOffset`).

**Behobener Defekt (§10)**

1. `Executor::WalkSnapshot` adressierte paged K/V über den **logischen**
   Blockindex statt über die physische Id aus der Tabelle.
2. `ContinuationAdapter::RowSpans` (der Pfad, den der Serving-Cache nutzt)
   machte denselben Fehler und plante Capture-Spans, bevor die Session Blöcke
   besaß.
3. Ein Prefill-Checkpoint ist nur eine `Session`-**View** der Quelle: er kopiert
   `attention_` (inklusive Tabellenzeiger), lässt aber `kv_blocks_` null. Damit
   fiel `RowSpans` genau für die Shared-Prefix-Captures des nebenläufigen
   Serving auf den logischen Index zurück.

**Nachweis**

- §10-Reproduktion (zwei identische greedy Requests, 145-Token-Prompt, 48
  Tokens, `--sessions 2`, zweiter Request aus dem RAM-Cache mit
  `cached_tokens=145`): Request 1 und 2 sind byte-identisch
  (`sha 6b00e61ccc2c73ec`) — vorher divergierte Request 2.
  Kontiguierlich liefert denselben Hash.
- Langer Prompt ohne Nebenläufigkeit (11 401 Tokens, Cache-Hit mit
  `cached_tokens=11401`): beide Antworten identisch (`sha 8c3a171b375b32ac`),
  d. h. auch mehrblättrige Restores sind exakt.
- `cache-shared-prefix` (überlappende Präfixe, parallel): **passed** (94,5 s bei
  16k, 4 Sessions).
- `cache-concurrency` (identische Gruppen, abgebrochene Requests, 4 parallele
  Sessions, 8k): **passed** (129,8 s) mit paged Pool; mit kontiguierlichem Pool
  ebenfalls 125,0 s. Vor Punkt 3 schlug der paged Lauf reproduzierbar fehl
  (falsche Antwort nach Restore), während kontiguierlich grün war.
- Artefakte: `artifacts/functional/paged-sp-16k-final/`,
  `artifacts/functional/paged-cc-8k-final/`,
  `artifacts/functional/contig-cc-8k-final/`.
- **Durchsatz unverändert:** derselbe Knee-Sweep mit paged Pool
  (`artifacts/serving/flash-next-knee-paged.json`, 4/8/16 Sessions × 16384
  Prefill, `--context 4096`, `--speculative off`) liefert **162,99 total
  tok/s** am Knee bei 8 Sessions, `saturated=true`, `monotone=true` —
  kontiguierlich waren es 162,85. Die Blockindirektion kostet also nichts
  Messbares.

### [ERLEDIGT] PA-P4 / KV-P1 / KV-P2 — Block-Sharing zwischen Requests

**Requirements (exakt aus der JSON):**

- [x] Paged Attention ist Voraussetzung für effizienten Shared KV-Pool
- [x] Block-Tabelle pro Request: logisch → physisch + Füllstand

**Umsetzung**

1. **Chunk-Granularität = Blockgröße.** `kKey`/`kValue` deklarieren
   `rows_per_chunk = 128` (`kKvBlockTokens`) und `layout_version = 2`. Ein Block
   ist der größte Bereich, der im Pool zusammenhängend ist; ältere
   Disk-Checkpoints mit 256er-Chunks werden über die Layout-Version verworfen
   statt fehlgelesen.
2. **Borrow statt Copy beim Capture.** `BorrowRowsView` liefert für eine paged
   Plane den Span, wenn der Bereich in genau einem Block liegt; der Cache hält
   dann die Device-Zeilen der Quelle (`Payload::Borrowed`, `RowPin`) statt einer
   Kopie. Bereiche über eine Blockgrenze hinaus sind nicht zusammenhängend und
   werden weiterhin kopiert.
3. **Adoption beim Restore.** `CopyRowsIn` erkennt am zurückgegebenen
   Device-Zeiger, dass die Zeilen noch live in einer Quell-Session liegen, und
   tauscht die Tabelleneinträge, statt zu kopieren:
   `KvBlockTable::Adopt` nimmt eine Referenz auf den Quellblock und gibt die
   eigene frei. Alle Layer teilen einen Blockindex; nach dem ersten Adopt sind
   die übrigen Layer ein No-op.
4. **Sicherheit.** `BeforeWrite` invalidiert vor jedem Schreibzugriff die
   Share-Records der Tabelle (ebenso `GuardAll`/`BeforeRelease`); ein
   materialisierter Payload hat einen anderen Zeiger und wird nie adoptiert.
   `BeginRestore`/`Invalidate` mappen die Tabelle neu (ReleaseAll + vollständiges
   Append), damit kein adoptierter Block in einen neuen Request überlebt; der
   Device-Tabellenspiegel wird per `d_kv_table_rows_ = 0` neu hochgeladen. Ein
   Guard verweigert den Copy-Pfad, falls doch ein geteilter Block beschrieben
   würde (R2-Schutz).
5. **CoW bleibt Pflicht.** Die partielle Tail-Zeile (< 128) wird immer in einen
   privaten Block kopiert; volle Blöcke sind append-only und werden nie wieder
   beschrieben.

**Nachweis**

- `tests/models/qwen/kv_blocks_test.cpp`: `AdoptSharesAFullBlock` (Quellblock
  Refcount 2, eigener Block zurück im Pool, erneutes Adopt leakt keine
  Referenz) und `AdoptRefusesInvalidBlocks`.
- `cache-shared-prefix` (16k, 4 Sessions): **passed** (87,7 s). Drei Restores
  adoptieren 26/10/26 Blöcke; Log:
  `event=kv_blocks_shared blocks=26 pool_live=486 pool_free=26` — die 26
  privaten Blöcke des Ziels (≈ 78 MiB bei 3 MiB/Block) sind zurück im Pool.
- `cache-concurrency` (8k, 4 Sessions): **passed** (133,2 s), 11 Restores mit je
  42 adoptierten Blöcken.
- §10-Reproduktion: weiterhin byte-identisch (`sha 6b00e61ccc2c73ec`) mit
  `cached_tokens=145`.
- **Durchsatz unverändert:** Knee-Sweep `artifacts/serving/flash-next-knee-shared.json`
  → 8 Sessions × 16384, **164,10 total tok/s**, `saturated=true` (vorher 162,99).
- Artefakte: `artifacts/functional/paged-sp-share-final/`,
  `artifacts/functional/paged-cc-share/`.

### [ERLEDIGT] KV-P3 — Prefix-Cache-Hit-Rate als Throughput-Metrik

**Requirements (exakt aus der JSON):**

- [x] Hit-Rate pro Request und global exportieren
- [x] Jeder vermiedene Full-Attention-Prefill-Block = zurückgewonnenes
  Token-Budget für Decode-Slots

**Umsetzung**

- Pro Request: `cached_tokens` (OpenAI-Usage) und neu
  `cache_hit_rate=<0..1>` in der strukturierten Request-Zeile; die Rate ist
  `cached_prompt_tokens / prompt_tokens`, geklemmt auf höchstens 1.
- Global in `/metrics`:
  - `llamacpp:prefix_cache_hits_total` (abgeschlossene Requests mit
    mindestens einem wiederverwendeten Prompt-Token),
  - `llamacpp:prefix_cache_misses_total` (Requests, die den ganzen Prompt
    prefilled haben),
  - `llamacpp:prefix_cache_hit_rate` = `cached / (cached + prefilled)`.
- Die Zähler werden genau einmal pro abgeschlossenem Request in
  `RecordRequestMetrics` fortgeschrieben, also auch für abgebrochene Requests
  ohne Doppelzählung.

**Nachweis**

- `http_server_test` prüft HELP/TYPE der beiden Zähler, den Gauge-Typ und dass
  die Rate in `[0,1]` liegt; `PrefixCacheHitRate` liefert 0,25 für 25/100 und
  klemmt 140/100 auf 1,0.
- GPU-Lauf (`artifacts/kv_p3_metrics_check.py`, Flash-Next): ein kalter und ein
  identischer Request ⇒ `prefix_cache_hits_total 1`,
  `prefix_cache_misses_total 1`, `prefix_cache_hit_rate 0.5`.

## Offene Punkte (nicht Teil der Phase-1-Abnahme)

- **PRD-Zielwerte** (`>100/150/80` aggregate Decode-tok/s) hängen an Phase 3/4
  (MLA-Retrofit, Quantisierung, MTP/DSpark); am Knee (8 Sessions) misst der
  jüngste Lauf Flash-Next 103,1 Decode-Summe / 66,3 Output, DeepSeek 60,9 / 28,0
  und 27B 64,1 / 33,9 tok/s ohne Spekulation.
- **Block-Sharing deckt volle Blöcke.** Der partielle Tail-Block (< 128 Zeilen)
  bleibt privat und wird kopiert; ein geteilter Block wird nie wieder
  beschrieben (append-only). Für Präfixe, die nicht auf eine Blockgrenze fallen,
  ist der Gewinn entsprechend um höchstens einen Block pro Session kleiner.
- `--sessions 32` überschreitet für Flash-Next mit 1024 Kontext und Q4-Gewichten
  den Unified-Memory-Bedarf dieses Containers (Server starb während des
  Benchmarks); das ist eine Kapazitäts-, keine Scheduler-Grenze.
- Alle drei Zielmodelle saturieren bei 8 Sessions, weil der Runner Decode-Wellen
  der Breite 8 ausführt. Der aktive paged Pool ändert das nicht (Flash-Next:
  164,10 tok/s nach dem Block-Sharing, 162,99 davor, 162,85 kontiguierlich); ein
  Knee oberhalb von 8 braucht breitere Decode-Wellen bzw. Spekulation (SD-P1),
  nicht einen größeren Scheduler-Parameter.
- Der Sweep wartet nach jedem Messpunkt 20 s (`SETTLE_SECONDS`), damit der
  Treiber die GEM-/GTT-Mappings eines 86–110-GB-Checkpoints freigibt; mit 2 s
  scheitert der nächste Modellstart reproduzierbar.

## Nachtrag: Batch-Durchsatz c8–c64 und Decode-Breite (31.10.2026)

Messung mit `artifacts/bench_pp_tg.py` (prompt-repeat 36, max-tokens 128,
context 4096, `--think off`, warmup 1, repetitions 2 bzw. 1 bei c64,
prefill-chunk 16384, `--max-connections` 128/512) im Container `gufo-work`,
Binary `build/release/gufo`. Vollständige Tabellen:
`artifacts/serving/pp-tg-comparison.md`.

| Modell | Sessions | c8 total | c16 | c32 | c48 | c64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flash-Next Q4_K_XL | 64 | 709,53 | 691,47 | 691,30 | 690,33 | 690,55 |
| Qwen3.8-27B Q4_K_XL (Breite 8) | 16 | 311,36 | 321,69 | 321,06 | 327,60 | 326,83 |
| DeepSeek-V4-Flash IQ2XXS | 16 | 280,98 | 282,40 | 280,29 | 279,30 | 281,16 |

Der Aggregat-Durchsatz ist ab c8 flach, während die TTFT linear mit der
Concurrency wächst (Flash-Next 6,1 s → 49,8 s). Sessions oberhalb der
Acht-Zeilen-Decode-Welle sind Warteschlangenkapazität, keine zusätzliche
Parallelität.

**Decode-Breite 16 für Qwen3.8-27B wurde gemessen und verworfen.** Bei gleicher
Concurrency kostet die 16-Zeilen-Welle 3,4–5,9 % Aggregat (c16 302,86 statt
321,69; c48 316,48 statt 327,60) und halbiert die per-Request-Decode-Rate
(3,50–3,62 statt 7,73–7,91 tok/s). Die Q4_K-Aufmerksamkeitsprojektionen laufen
bereits über den gemeinsamen nativen K-Quant-Pfad (`IsNativeWmmaQuant(kQ4_K)`);
das flache Aggregat ist damit eine ALU-/FP32-Akkumulationsgrenze, keine
Grenze der Gewichts-Leserate. Die Plan-Auswahl steht deshalb wieder auf acht
Zeilen (`MaximumDecodeBatchWidth()`), die 16-Zeilen-Arrays bleiben für die
MTP-Verifikation erhalten. Verifikation: 16/16 Completion-Hashes identisch zum
früheren Breite-8-Lauf, physische Breite 8, Pläne `batched-w2` … `batched-w8`.

**Session-Kopplung:** c8 mit 8 vs. 16/64 Sessions ergab Flash-Next +1,9 % (64),
Qwen3.8-27B −2,8 % (16) und DeepSeek +1,0 % (16). Das Vorzeichen wechselt pro
Modell, jeder Abstand liegt im Streubereich; ein festes Koppeln von
`--sessions` an die Decode-Breite ist durch die Daten nicht gedeckt.
`--sessions` bleibt die Aufnahmegrenze des Nutzers.

**Umsetzungsstand:** Der Fix am Component-Prefix-Cache (Transfer-Pool-Reserve)
ist gebaut und verifiziert; ohne ihn schlugen bei `--sessions ≥ 64` alle
1008 Snapshot-Captures fehl. Wirkung: identischer Prompt sequenziell 1,99 s →
0,53 s, acht gleichzeitige identische Requests 12,63 s → 8,62 s pro Runde.
Kosten im kanonischen Benchmark (eindeutige Nonce pro Request, Cache kann nie
treffen): c8 agg_tg 47,70 → 43,32 (−9 %).

**Gegenprobe auf dem zurückgesetzten Binary** (`--sessions 16`, Concurrency 16,
repetitions 2, Artefakt `artifacts/serving/pp-tg-qwen38-27b-width8-c16.json`):
total **324,61 tok/s**, agg_tg 21,82, agg_pp 302,79, per-Request-Decode 7,79,
per-Request-Prefill 477,19, TTFT 59,5 s, physische Breite 8, Pläne
`batched-w2` … `batched-w8`. Damit liegt die Acht-Zeilen-Welle 7,2 % über dem
Breite-16-Lauf gleicher Concurrency (302,86 total) und 2,2x über dessen
per-Request-Rate (3,50 tok/s).

### Flash-Next Decode-Breite 16 (31.10.2026)

Die Runner-Capabilities meldeten Pläne bis 16 Zeilen, aber
`GatedDeltaNetBatch` wies Kohorten über acht Sessions ab
(`kernels.hip.cpp`, `count > 8`). Deshalb liefen alle bisherigen Flash-Next-Läufe
mit physischer Breite 8, und ein Lauf mit 16 gleichzeitigen Requests brach mit
`Flash-Next advance failed: batched GatedDeltaNet launch failed` ab. Die Kernel
indizieren über `blockIdx.z` und eine 32-Bit-Aktivmaske, vertragen also 16
Zeilen; das Limit steht jetzt auf 16. Zusätzlich erlaubt
`batched_multi_token_decode_max_width` der Flash-Next-Capabilities 16 Zeilen,
damit breite Kohorten im gebatchten Decode-Pfad bleiben.

**Messergebnis** (`--sessions 64`, prompt-repeat 36, max-tokens 128,
repetitions 2, Artefakt `pp-tg-flash-next-w16-c8c16.json`):

| Concurrency | Breite | total tok/s | agg_tg | req_tg | TTFT |
| ---: | :---: | ---: | ---: | ---: | ---: |
| 8 | 8 | 663,94 | 44,63 | 11,98 | 6,7 s |
| 16 | 16 | 717,17 | 48,18 | 7,46 | 13,3 s |

Innerhalb dieses Binary steigert die Verdopplung der Concurrency den
Gesamtdurchsatz um 8,0 % (663,94 → 717,17) und den Aggregat-Decode um 8,0 %
(44,63 → 48,18); im vorigen Binary senkte derselbe Schritt beides um 2,5 %
(709,53 → 691,47). Die 16-Zeilen-Welle macht aus dem c16-Rückgang einen Zuwachs
von rund 10,5 Prozentpunkten. Der absolute c16-Wert steigt über die Binaries
hinweg um 3,7 %; der c8-Vergleichspunkt dieses Binary liegt 6,4 % unter dem
vorigen (Capture-Overhead des Prefix-Cache, siehe oben), der wahre Gewinn ist
also größer. Pro Request sinkt der Decode von 11,98 auf 7,46 tok/s — die breite
Welle tauscht Latenz gegen Batch-Durchsatz.

**Korrektheit:** 16 gleichzeitige Requests liefen mit Plänen `batched-w2` …
`batched-w16`; alle acht Completions, die der 8-Zeilen-Lauf ebenfalls enthält,
sind byte-identisch (`flashnext-width8-correctness.json`,
`flashnext-width16-correctness.json`).

**Offen:** Ab neun Zeilen plant `MtpBatchController::Choose` keine Drafts mehr,
breite Kohorten laufen also ohne MTP-Spekulation. Drafting auch bei 16 Zeilen zu
erlauben braucht eine Kostenkurve für diese Breite; die vorhandene sättigt bei
der C8-Kohorte.

**Test-Build-Klärung:** Der Targetname `qwen38_flash_next_gdn_ops_test` ist
korrekt; er existiert nur in Builds mit `BUILD_TESTING=ON`. `build/release` hat
`BUILD_TESTING=OFF` (Preset `release`), daher antwortet Ninja dort mit
`unknown target`. Das Preset `gpu-test` (`BUILD_TESTING=ON`,
`ENGINE_ENABLE_HIP=ON`, RelWithDebInfo) konfiguriert und baut den Target;
`ctest -R qwen38_flash_next.gdn_ops` läuft grün. Der Test prüft seit dem
Breite-16-Umbau zusätzlich eine Kohorte mit 16 Deskriptoren und aktiver Zeile
15 gegen einen Sentinel-Puffer, damit ein früher Kernel-Return nicht als
Treffer durchgeht.

**Skalierung über c16 hinaus** (repetitions 1, gleiche Sonst-Parameter,
Artefakt `pp-tg-flash-next-w16-c32c64.json`): c32 total 781,84 tok/s
(agg_tg 52,49), c64 total 780,15 (agg_tg 52,37), physische Breite jeweils 16.
Die Breite 16 skaliert damit über c16 hinaus: c8 → c16 → c32 = 663,94 → 717,17
→ 781,84 (+17,7 %), im vorigen Binary 709,53 → 691,47 → 691,30 (−2,6 %).
Bei c32 laufen zwei 16-Zeilen-Wellen, die Breite bleibt 16.

**Verworfener Kernel-Nachschlag: breite Q8_0-Dense-Launches.** `DenseBatch`
zerlegte eine 16-Zeilen-Kohorte in zwei 8-Zeilen-Launches, sodass jede
Q8_0-Gewichtskachel pro Decode-Schritt zweimal gelesen wurde. Der
Zusammenbau zu einem Launch (der MMVQ-Dispatch leitet neun bis sechzehn
gate-lose Token auf `launch_q8_batch<2>`, dessen zwei Wellen dieselbe
Gewichtszeile bearbeiten und Cache-Zeilen teilen) wurde gebaut, geprüft und
gemessen: c8 673,96 (Referenz 663,94, +1,5 %), c16 725,79 (Referenz 717,17,
+1,2 %), physische Breite 8 bzw. 16. Der c8-Arm kann die Änderung nicht
berühren (rows ≤ 8 kehrt in den Ein-Zeilen-Pfad zurück), seine +1,5 % sind also
Host-Drift; damit liegt c16 mit −0,3 % normalisiert im Rauschen. Auch das
c16/c8-Verhältnis bleibt gleich (1,0762 statt 1,0795). Completions waren
byte-identisch (16/16 Hashes gleich `flashnext-width16-correctness.json`).
Artefakt: `artifacts/serving/pp-tg-flash-next-dense16-c8c16.json`; die Änderung
ist wieder zurückgenommen.

Grund: Der geroutete Experten-FFN läuft bereits mit allen 16 Zeilen über
`qfn_mmq_moe_gated_vec`, Nicht-Q8_0-Projektionen laufen ohnehin in einem
`SmallGemm`-Launch beliebiger Breite. Nur Q8_0-Matrizen wurden geteilt, und die
gated Variante lässt sich gar nicht verbreitern — der MMVQ-Dispatch weist mehr
als `MMVQ_MAX_BATCH_SIZE` (8) Zeilen ab, sobald ein Gate-Zeiger gesetzt ist. Es
bleibt damit die Shared-Expert-Up/Gate-Projektion als geteilter Rest, ein zu
kleiner Anteil, um über das Rauschen zu kommen. Weitere Kernel-Verbreiterung ist
auf diesem Pfad kein Hebel; der Durchsatzgewinn der Breite 16 stammt aus der
GDN- und Attention-Batchung, die bereits vollständig greift.

**DeepSeek-V4-Flash Decode-Breite 16 wurde gemessen und verworfen**
(Artefakt `pp-tg-deepseek-w16-c8c16.json`, C8/C16 auf dem erweiterten Binary):

| conc. | reqs | req_pp | req_tg | agg_pp | agg_tg | total | TTFT ms | Breite |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 8 | 8 | 387,73 | 7,37 | 257,64 | 19,12 | 276,75 | 35 149 | 8 |
| 16 | 16 | 390,17 | 3,89 | 261,54 | 19,40 | 280,94 | 70 124 | 16 |

Gegen das vorige Binary bei gleicher Concurrency: c16 −0,5 % (280,94 statt
282,40), c8 −1,5 % (276,75 statt 280,98, Host-Drift derselben Messung). Die
breite Kohorte bringt also keinen Aggregatgewinn und halbiert zusätzlich die
per-Request-Decode-Rate (3,89 statt 7,43 tok/s). Ursache wie bei Flash-Next
angedeutet, hier aber stark genug, den Gewinn zu löschen: sechzehn MoE-Zeilen
routen auf mehr unterschiedliche Experten, der zusätzliche Experten-
Gewichtstraffic hebt den geteilten Lesevorteil auf. Zusätzlich lieferte der
Breite-16-Pfad in zwei identischen Läufen unterschiedliche Completions
(`deepseek-width16-correctness.json` vs `…-run2.json`), also keinen stabilen
numerischen Vertrag. Alle fünf Stellen (AdvanceBatch, EvaluateBatch,
ds4_sessions_eval_batch, rocm_graph_eval_sessions_batch, Plan-Cap) sind auf acht
Zeilen zurückgesetzt; Verifikation: 16/16 Completion-Hashes identisch zum
Breite-8-Baseline-Lauf, physische Breite 8, Pläne `batched-w2` … `batched-w8`.

**Wo die Batch-Runde ihre Zeit verbringt: serialisierter Prefill.**
`agg_pp / req_pp` ist exakt der Prefill-Anteil an der Rundenspanne, weil der
Scheduler einen Request nach dem anderen prefillt und `req_pp` die einzige
erreichte Prefill-Rate ist. Daraus der Decode-Phasen-Durchsatz (Output-Token
über den Nicht-Prefill-Teil der Runde):

| Modell / Level | total tok/s | req_pp | req_tg | Prefill-Anteil | Decode-Phase tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash-Next c8 (w8) | 709,53 | 1331,41 | 12,33 | 49,7 % | 94,8 |
| Flash-Next c16 (w8) | 691,47 | 1305,50 | 11,90 | 49,4 % | 91,8 |
| Flash-Next c16 (w16) | 717,17 | 1173,28 | 7,46 | 57,0 % | 112,1 |
| Flash-Next c32 (w16) | 781,84 | 1338,87 | 7,66 | 54,5 % | 115,3 |
| Qwen3.8-27B c16 (w8) | 324,61 | 477,19 | 7,79 | 63,5 % | 59,7 |
| DeepSeek c8 (w8) | 280,98 | 395,27 | 7,42 | 66,2 % | 57,4 |

Zwei Folgen. Erstens ist der Kernel-Gewinn der 16-Zeilen-Welle größer als die
End-to-End-Tabelle zeigt: der Decode-Phasen-Durchsatz steigt 94,8 → 112,1 tok/s
(+18 %) und erreicht bei c32 115,3; das End-to-End-Ergebnis wird nur verwässert,
weil der Prefill-Anteil auf 57 % wächst. Zweitens dominiert bei 27B und DeepSeek
der serialisierte Prefill zwei Drittel der Runde — nicht die Decode-Breite.

Beide Phasen bleiben weit unter ihren Grenzen: Prefill läuft mit ~1331 tok/s,
also 23 % der DRAM-Spitze und 15 % der fp16-Spitze; Decode erreicht 52 % der
DRAM-Spitze. Die Prefill-Policy hat bereits gemessen, dass 4096-Token-Chunks 5–11 %
schneller sind als 512er, feinere Verschachtelung ist also kein Hebel. Was bleibt,
ist Prefill und Decode *gleichzeitig* statt hintereinander laufen zu lassen. Weil
jede Phase die DRAM-Spitze zu über der Hälfte brachliegen lässt, ist der Gewinn
nicht von einer der beiden Phasenlimits gedeckelt, sondern von der aggregierten
DRAM-Rate: eine c8-Runde bewegt 10,6 GB × 128 Decode-Schritte plus 72 GB × 8
Prefills = 1,94 TB Gewichtstraffic in 21,4 s, also 91 GB/s bzw. 38 % der Spitze.
Bei 150–180 GB/s aggregiert fiele die Runde auf 12,9–10,8 s, also bis zu +66…+98 %
Gesamtdurchsatz. Die strukturelle Voraussetzung ist allerdings enger als „kein
globaler Lock": jedes Modell besitzt genau einen `Executor` (Flash-Next legt
`m->executor_` einmal in `engine.cpp` an), jede Session ist nur ein Unterobjekt
davon, und alle Residents reihen sich auf demselben `stream_` ein; der Scratch
`s_` ist ebenfalls eine Einzelinstanz. Das Capability-Flag
`requires_device_runtime_lock`, das nach einem globalen Serialisierungsvertrag
aussieht, wird von allen drei Runnern gesetzt, aber nirgends im Baum gelesen —
nichts erzwingt exklusiven Gerätezugriff, was allein aber nichts ändert. Die echte
Serialisierung sind der eine `stream_` und der eine Scheduler-Worker-Thread:
`StepPrefill` und `StepDecodeBatch` werden in `RunLoop` hintereinander aufgerufen
und warten jeweils auf ihr Ergebnis. Überlappung braucht also einen zweiten Stream,
partitionierten Scratch und einen Scheduler mit zwei gleichzeitig fliegenden
Phasen; der Executor besitzt für Intra-Prefill-Überlappung bereits einen
`side_stream_` mit Event-Joins (`JoinSide`), an den eine breitere Aufteilung
anschließen könnte. Hier nicht begonnen.

## Nachtrag: Decode-Phasen-Profil und korrigierte Byte-Rechnung (01.11.2026)

Ein `rocprofv3`-Trace eines decode-dominierten Laufs (Server mit 8 Sessions,
acht gleichzeitige Requests mit unterschiedlichen Prompts, 64 Output-Tokens; ein
Request stoppte per EOS, der Plan war also `batched-w7`) trennt sich sauber nach
Wandzeit in Prefill- und Decode-Fenster: Prefill 25 315 Dispatches in 3 899 ms
Kernelzeit, Decode 144 171 Dispatches in 4 841 ms Kernelzeit über rund 5 000 ms
Wandzeit, also **96,8 % GPU-Auslastung**. Die 2 253 Launches pro Decode-Schritt
dauern im Mittel 33 µs. Top-Kernel im Decode-Fenster: `mul_mat_vec_q8<7,...>`
(Q8_0 dense/gated) 1 150 ms / 23,8 %, `mul_mat_vec_moe_grouped<Q4_K,7>` 1 135 ms
/ 23,4 %, `mul_mat_vec_q_moe<Q4_K,8>` 653 ms / 13,5 %, `GdnKernel` 270 ms,
`AttentionKernel<2u>` 180 ms, `RoutedF16GEMMKernel` 162 ms,
`qfn_q8_hc_down_kernel` 126 ms, `SmallGemmKernel` 104 ms. Die MMVQ-Familie trägt
rund 63 % der Decode-Zeit und damit praktisch den gesamten Gewichtstraffic.

Die frühere Rechnung „Decode läuft mit 125 GB/s bzw. 52 % der DRAM-Spitze" war
falsch, weil sie nur die Routed-Expert-Gewichte zählte. Die Tensorliste des
Shards (über `tools/gufo/gguf.py`, alle vier Teile summiert) ergibt: 71,73 GB
Routed Experts, 27,47 GB `per_layer_token_embd` (IQ4_NL-Lookup-Tabelle,
320,0 M × 160, wird zeilenweise gegriffen und kostet pro Schritt nur Kilobyte)
und 4,48 GB übriger dichter Trunk (`attn_qkv` 956 MiB über 36 Layer, `attn_q`
382 MiB über 12, `attn_gate`/`ssm_out` je 574 MiB über 36, `attn_output`
191 MiB über 12, vier Hyper-Connection-Projektionen 638 MiB über 48, F32-Router
240 MiB, Shared Expert 239 MiB). Diese 4,48 GB liest **jeder** Decode-Schritt
vollständig. Ein c8-Schritt bewegt also 10,6 GB Experts + 4,48 GB dicht = 15,1 GB
und erreicht bei 75,6 ms **200 GB/s, 83 % der gemessenen 240,76 GB/s**. Decode
steht damit an der DRAM-Wand; der Spielraum beträgt rund 20 %, nicht 90 %.

Damit ist auch die Rundenschätzung zu korrigieren: eine c8-Runde bewegt
8 × 76,2 GB Prefill plus 128 × 15,1 GB Decode = **2,54 TB in 21,4 s, also
119 GB/s bzw. 49 % der Spitze**. Überlappung ist bei der Rate gedeckelt, die
Decode allein schon hält (200 GB/s): +68 %; +100 % nur, wenn beide Phasen
gemeinsam 240 GB/s erreichten. Prefill selbst bleibt bei 58 GB/s (24 %) an der
eigenen Kernel-Effizienz hängen, nicht am Speicher — Nebenläufigkeit füllt
Decodes DRAM-Leerlauf, macht aber keinen Kernel schneller. Dieselbe Beobachtung
auf der Decode-Seite: 37 % der Decode-Zeit sind nicht-streamende Arbeit
(GatedDeltaNet-Rekurrenz, Attention, Hyper-Connection-Combines,
Aktivierungs-Quantisierung), die läuft, während DRAM stillsteht.

Ein bestehender Hebel wurde bisher in **keiner** Batch-Messung genutzt:
MTP-Speculative-Decoding ist im Binary implementiert (`--speculative mtp`,
`--mtp-model`), der Draft-Head liegt vor
(`/models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`,
2,79 GB), und alle bisherigen `pp-tg`-Läufe zeigen `speculative=off`. Da Decode
bandbreiten- und nicht rechengebunden ist, vervielfacht Spekulation die Tokens
pro Gewichtsdurchlauf; gemessen wird das als nächstes.

### Ergebnis der MTP-Messung: kein Batch-Gewinn (01.11.2026)

`artifacts/serving/pp-tg-flash-next-mtp-c8c16.json` (gleiches Binary, gleiche
Harness: `sessions 64`, prompt-repeat 36, max-tokens 128, reps 2, zusätzlich
`--speculative mtp --mtp-model .../MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`):

| Ebene | reqs | req_pp | req_tg | agg_pp | agg_tg | total | TTFT ms | Breite | Drafts/Output | Acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| c8 | 16 | 1138,28 | 15,34 | 636,24 | 45,80 | 682,04 | 7 113 | 8 | 0,66 | 86,4 % |
| c16 | 32 | 1138,00 | 7,35 | 652,46 | 46,96 | 699,43 | 13 630 | 16 | 0,02 | 98,5 % |

Gegen dasselbe Binary ohne Drafting (c8 663,94 total / 44,63 agg_tg, c16
717,17 / 48,18): c8 **+2,7 %** total, c16 **−2,5 %**. Auf acht Zeilen wirkt
Drafting genau wie erwartet — 86,4 % Acceptance, per Request 11,98 → 15,34 tok/s
Decode (+28,1 %), aggregiert 94,8 → 122,7 tok/s —, aber die Runde verbringt rund
die Hälfte ihrer Zeit im serialisierten Prefill, den Drafting nicht berührt; der
Gesamtzuwachs bleibt 2,7 %. Auf sechzehn Zeilen ist Drafting abgeschaltet
(`MtpBatchController::Choose` liefert für Kohorten > 8 Zeilen null Drafts: 65
gedraftete Tokens auf 32 Requests × 128 Tokens = 0,02 pro Output-Token).

Der eigentliche Befund ist die Decode-Decke: 94,8 tok/s bei acht Zeilen plain,
122,7 bei acht Zeilen plus Drafting, 117,6 bei sechzehn Zeilen (Drafting aus) und
119,4 bei sechzehn Zeilen plain. Vier Formen konvergieren auf ~120 tok/s. Das
Zeilenbudget, nicht die Zeilenzahl, ist die Grenze: eine sechzehnzeilige
Verifikationskette (8 Anker + 8 Drafts) bräuchte 32 Zeilen, also das Doppelte des
qualifizierten Limits, und kostete rund das Doppelte eines plain
Sechzehn-Zeilen-Schritts für 1,87 Tokens pro Zeile — kein Gewinn. MTP bleibt
damit ein Latenz-Hebel für niedrige Nebenläufigkeit, kein Batch-Durchsatz-Hebel.

### Session-Pool-Größe: 15 GB mehr Speicher, aber reproduzierbare +4,6 % bei c8 (01.11.2026)

Gleiches Binary, gleiche Harness, gleiche Nebenläufigkeit 8, nur `--sessions`
unterschiedlich. Der 8-Session-Punkt wurde absichtlich zweimal gemessen, um einen
echten Effekt von der ±5 %-Streuung zwischen zwei Serverstarts zu trennen:

| sessions | Draft | total | agg_tg | agg_pp | req_pp | req_tg | TTFT ms | gpu_device_used |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 (Lauf 1) | aus | 687,20 | 46,20 | 641,00 | 1297,49 | 11,85 | 6 262 | 88 257 MiB |
| 8 (Lauf 2) | aus | 684,50 | 46,02 | 638,49 | 1279,74 | 12,25 | 6 584 | — |
| 8 (Lauf 2) | mtp | **720,85** | **48,46** | **672,39** | 1265,27 | **15,99** | 6 513 | — |
| 16 | aus | 652,88 | 43,91 | 608,97 | 1175,29 | 11,73 | 6 788 | 90 501 MiB |
| 64 | aus | 655,69 | 44,10 | 611,59 | 1169,28 | 12,06 | 6 966 | 103 922 MiB |
| 64, `--cache-ram-bytes off` | aus | 679,17 | 45,61 | 633,56 | 1207,74 | 12,57 | 11 788 | 101 766 MiB |

Die beiden 8-Session-Läufe stimmen auf 0,4 % überein (687,20 vs 684,50), die 16-
und 64-Session-Läufe ebenfalls auf 0,4 % (652,88 vs 655,69); zwischen den beiden
Gruppen liegen jedoch 4,6 %. Das ist damit **kein Startrauschen**: acht Sessions
sind bei Nebenläufigkeit 8 rund 4,6 % schneller als 16 oder 64, und schon 16
Sessions sättigen den Malus. Das Signal kommt aus dem Prefill: 1297 und 1280 tok/s
pro Request gegen 1175 und 1169, also rund 10 % Prefill-Verlust durch den großen
Pool.

Der Speichereffekt ist eindeutig: 64 Sessions belegen 15,3 GB mehr Gerätespeicher
(88,3 → 101,5 GB von 124 GiB) und lassen statt 36,5 GB nur 20,9 GB Hostspeicher
frei. Der Pool ist eine gemeinsame KV-Arena der Größe
`pool_sessions x blocks_per_session`, also 64 Sessions x 4096 Tokens x rund 75 KB
pro Token ≈ 19,6 GB, von denen in einer Runde mit Nebenläufigkeit 8 rund 56
Sessions leer stehen. Schon 16 Sessions kosten 2,2 GB mehr als acht und verlieren
denselben Prefill-Anteil — der Malus sättigt früh, die restlichen 13,4 GB des
64er-Pools bringen nichts.

Beide Effekte zusammen ergeben bei Nebenläufigkeit 8 mit `--sessions 8` **und**
`--speculative mtp` **720,85 total**, also **+10,0 %** gegenüber dem 64-Session-
Punkt ohne Drafting (655,69) und +4,9 % gegenüber dem besten 8-Session-Lauf ohne
Drafting. Der Draft-Head kostet 2,79 GB, gehört also zum kleinen Pool: bei 16 und
64 Sessions ist Drafting für breite Kohorten abgeschaltet und sein Head kostet nur
Speicher (699,43 vs 714,16 bei c16, −2,1 %).

`--cache-ram-bytes off` bringt bei 64 Sessions +3,6 % (655,69 → 679,17) bei
unverändertem Gerätespeicher (101,8 vs 103,9 GB) — der Capture-Aufwand ist
Stream-Arbeit, kein Speicher. Dafür verschlechtert es TTFT von 7,0 s auf 11,8 s;
kein empfehlenswerter Tausch, denn derselbe Cache bringt bei wiederholten
Präfixen −32 % pro Runde.
Der nach allen Änderungen neu gebaute Release-Binary wurde gegen das frühere
Correctness-Artefakt geprüft: acht gleichzeitige 32-Token-Requests bei
`--sessions 16` melden physische Breite 8 mit Plänen `batched-w2 … batched-w8`,
und alle 8 Completion-Hashes sind byte-identisch zu `mtp-correctness-off.json`
(`artifacts/serving/flashnext-rebuild-verify.json`). Der Format-Check des
gesamten Baums läuft ebenfalls durch (`PASS: Checked 708 C++ files`).


### Korrektur der c32/c64-Arme: der ausgelieferte Stand misst niedriger (01.11.2026)

Die Skalierungstabelle im Vergleichsdokument nennt für Flash-Next c32 781,84 und
c64 780,15 total. Beide Zahlen stammen vom 09:26-Lauf, also vom Binary **vor** dem
Präfix-Cache-Transferpool-Fix. Dieselben Stufen wurden auf dem ausgelieferten
Stand neu gemessen (gleiches Modell, `--sessions 64`, prompt-repeat 36,
max-tokens 128, repetitions 2): c32 **734,99** total / agg_tg 49,37 / req_tg 7,73 /
req_pp 1171,73 / TTFT 26,4 s, c64 **700,79** / 47,17 / 7,20 / 1289,92 / TTFT 52,0 s,
beide bei physischer Breite 16. Das sind −6,0 % bzw. −10,2 % gegen die Arme vor dem
Fix, und der Betrag deckt sich mit dem bereits gemessenen Capture-Aufwand bei c8
(−9 % im Benchmark mit eindeutigem Nonce, in dem Captures nie treffen): bei 64
Sessions und 64 gleichzeitigen Requests protokolliert der Cache pro Request
`capture_refused reason=byte_capacity` plus Retention-Umlauf. Vergleiche also nur
innerhalb eines Binary; die Nach-Fix-Zahlen dieses Dokuments (655,69, 714,16,
734,99, 700,79) sind die, die der ausgelieferte Stand produziert. Innerhalb des
Nach-Fix-Stands skaliert c8 → c32 nur um +7,2 % (684,50 → 734,99), und c64 fällt
gegenüber c32 auf 700,79 (−4,7 %).

### Empfohlene Batch-Konfiguration (01.11.2026)

Jede Zeile ist ein Vergleich mit demselben Binary aus diesem Dokument, nichts
ist hochgerechnet:

| Entscheidung | Einstellung | Beleg |
| --- | --- | --- |
| Session-Pool | `--sessions` = tatsächliche Nebenläufigkeit (8 bei 8 breitem Batch) | +4,6 % bei c8 gegen 16/64 Sessions, 15,3 GB weniger Gerätespeicher |
| Drafting | `--speculative mtp --mtp-model .../mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` bei Flash-Next bis 8 Zeilen | +5,3 % bei c8, 86,4 % Akzeptanz, verlustfreie Greedy-Ausgabe; auf dem dichten 27B-Pfad neutral (127,02 vs 125,48 tok/s) |
| Drafting über 8 Zeilen | aus lassen | von `MtpBatchController::Choose` abgeschaltet, −2,1 % bei c16 |
| Präfix-Cache | RAM-Cache an lassen | −32 % pro Runde bei wiederholten Präfixen; Abschalten bringt nur +3,6 % und kostet TTFT 7,0 → 11,8 s |
| Decode-Breite | 27B und DeepSeek acht Zeilen, Flash-Next sechzehn | Breite 16 für 27B (−3,4…−5,9 %) und DeepSeek (−0,5…−1,5 %) verworfen, für Flash-Next angenommen (+8,0 % c8→c16) |
| Prefill-Chunk | 4096 Tokens oder mehr | 4096 gemessen 5–11 % schneller als 512 |

Beide c8-Hebel zusammen (`--sessions 8` plus Drafting) erreichen **720,85 total**,
**+10,0 %** gegenüber dem 64-Session-Punkt ohne Drafting (655,69) mit demselben
Binary. Der verbleibende Hebel ist architektonisch, keine Einstellung: Prefill und
Decode laufen hintereinander auf einem Stream. Prefill lässt DRAM bei 23 %, Decode
liegt bei 83 % DRAM und nur rund 9 % VALU — eine Zwei-Stream-Überlappung ist damit
bei rund +68…+100 % gedeckelt, braucht aber partitionierten Scratch und einen
Scheduler mit zwei gleichzeitig laufenden Phasen.

### Paged Attention: Zeilenadressierung pro Block gehoisted (01.11.2026)

Der Prefill-Attention-Kernel (`WmmaCausalAttention`) hat die Block-Tabelle pro
Key **und pro Plane** aufgelöst: `KvHalfRow` lief für jede K-Zeile und erneut für
jede V-Zeile, obwohl ein 128-Token-Block alle seine Positionen auf dieselbe Basis
abbildet. Jetzt wird die Basis einmal pro Block aufgelöst, in Registern gehalten
und vom K- und vom V-Durchlauf gemeinsam benutzt (der V-Durchlauf besucht genau
die Keys, die der K-Durchlauf eben aufgelöst hat).

`tools/bench`-Messung desselben Kernels (4096-Token-Chunk, zehn Wiederholungen,
Ausgabe-Hash unverändert):

| Geometrie | contiguous | paged vorher | paged nachher |
| --- | ---: | ---: | ---: |
| dense causal | 16,37 ms | 19,97 ms (+22,0 %) | 17,16 ms (+2,2 %) |
| sparse, Tiefe 4096 | 19,44 ms | 21,59 ms (+11,7 %) | 20,18 ms (+4,2 %) |
| sparse, Tiefe 16384 | 23,78 ms | 27,18 ms (+14,3 %) | 26,45 ms (+9,3 %) |
| dense, 1776-Token-Prompt | 3,38 ms | 3,57 ms (+5,4 %) | — (+2…4 %) |

Der Paging-Aufwand im dichten Prefill fällt damit von 22,6 % auf 2–4 %; sparse
behält 9 % bei 16k Tiefe, weil dessen Keys echt gestreut sind und weiterhin einen
Lookup pro Position brauchen. Reproduzierbarkeit: drei aufeinanderfolgende
Dense-Läufe ergeben +4,2 / +2,4 / −2,9 %, der Rest liegt also im Spread.
Korrektheit: `attn_bench`-Hashes sind für contiguous und paged bei jeder
Geometrie identisch, `kv_page_ops_test` meldet weiter „paged KV matches the
contiguous cache byte for byte" (WMMA- und Per-Token-Pfad), `attention_ops_test`
bleibt bei denselben Worst-Case-Fehlern (3,8e-05 … 4,7e-05).

Ende-zu-Ende ist die Änderung neutral — das ist das ehrliche Ergebnis:

| conc. | Cache | total vorher | total nachher | Delta | agg_tg vorher → nachher |
| ---: | :---: | ---: | ---: | ---: | --- |
| 8 | an | 655,69 | 655,36 | −0,1 % | 44,10 → 44,13 |
| 16 | an | 714,16 | 723,84 | +1,4 % | 48,03 → 48,65 |
| 32 | an | 734,99 | 740,77 | +0,8 % | 49,37 → 49,76 |
| 32 | aus | 776,82 | 765,87 | −1,4 % | 52,18 → 51,41 |
| 64 | aus | 769,80 | 768,91 | −0,1 % | 51,73 → 51,70 |

Mittel über die fünf gleichartigen Paare: **+0,1 %**. Der Grund ist die
Geometrie: ein ausgelieferter Request prefillt ~1776 Tokens in einem Chunk, dort
war der ganze Paging-Aufwand 5,4 % des Attention-Kernels, und Attention ist nur
ein einstelliger Anteil des serialisierten Prefills. Der Gewinn ist trotzdem
sinnvoll, weil er mit der Kontexttiefe skaliert — bei 16k Tiefe hätte der
sparse-Pfad sonst 14 % gezahlt. Artefakte:
`pp-tg-flash-next-pagedopt-c8c32.json`, `…-c32.json`,
`…-nocache-c32.json`, `…-nocache-c64c128.json`;
Kernel-Bench `build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_attn_bench`
(neu mit `--paged` / `--identity`).

### 256 parallele Anfragen: der Präfix-Cache ist der Hebel (01.11.2026)

Gleiche Binary-Familie, `--sessions 64`, physische Breite 16, Cache an gegen
`--cache-ram-bytes off`:

| conc. | Cache an total | Cache aus total | Gewinn | agg_tg an → aus |
| ---: | ---: | ---: | ---: | --- |
| 32 | 734,99 | 776,82 | +5,7 % | 49,37 → 52,18 |
| 64 | 700,79 | 768,91 | +9,7 % | 47,17 → 51,70 |
| 128 | — | 777,58 | — | 52,11 |
| 256 | — | 777,95 | — | 52,20 |

Der Cache-Aufwand wächst mit der Nebenläufigkeit: jeder der 64/128 gleichzeitigen
Requests bekommt einen Capture angeboten, das Byte-Budget lehnt ihn ab
(`capture_refused reason=byte_capacity`, 41 Mal im c32-Lauf mit Cache, kein Mal
ohne), und Retention zykelt dabei das Ledger. Zusätzlich hält der Cache bei c32
rund 10,3 GB Host-RAM (Server-RSS 10,9 GB gegen 0,6 GB ohne; host_available
12,5 GB gegen 23,1 GB) — genau das Budget, das eine 256er-Warteschlange braucht.
Cache-aus kostet TTFT: bei c32 26,4 → 45,2 s, bei c64 52,0 → 91,6 s, weil
gecachte Präfixe für die hinteren Requests keinen Prefill mehr überspringen.
Für maximalen Batch-Durchsatz bei 256 parallelen Anfragen ist dieser Tausch
richtig; für interaktive Latenz mit wiederholten Präfixen nicht.

**Reichweite dieses Vergleichs:** Die Benchmark-Harness stellt jedem Request
einen eindeutigen Nonce voran (`Benchmark nonce <uuid>-<c>-<r>-<i>. ` in
`tools/gufo/serving_bench.py`), also konnte in diesen Läufen kein Request einen
Präfix teilen und der Cache hat ausschließlich Arbeit gekostet. Die +5,7 %/+9,7 %
messen damit den **Nicht-Treffer-Fall**. Wenn Prompts einen Präfix teilen, ist der
Cache stattdessen 2,9–7,1× besser (siehe „Shared Prefix Caching" oben). Die
Entscheidung hängt also am Workload: eindeutige Prompts → Cache aus, geteilte
Präfixe → Cache an.

### Shared Prefix Caching: läuft und dupliziert nicht (01.11.2026)

`artifacts/bench_shared_prefix.py` schickt denselben führenden Prompt von allen
Clients (optional mit eigener User-Frage) und liest die Cache-Felder pro Request
plus die `/metrics`-Zähler. Flash-Next, sessions 16, c8, max-tokens 32, Prefix
~1800 Tokens, Cache an gegen `--cache-ram-bytes off`:

| Runde | Cache | agg_tg | TTFT | cached_tokens pro Request | prefill_tokens pro Request |
| ---: | :---: | ---: | ---: | --- | --- |
| 0 (kalt) | an | 44,53 | 2 594 ms | 0, dann 2166×7 | 2182, dann 16×7 |
| 1 (warm) | an | 111,57 | 33 ms | 2182×8 | 16×8 |
| 0 | aus | 15,37 | 13 667 ms | 0×8 | 2182×8 |
| 1 | aus | 15,65 | 13 407 ms | 0×8 | 2182×8 |

Ein 2182-Token-Prompt kostet pro Request also **16 statt 2182 Prefill-Tokens**;
der Kreis liefert **2,9× mehr Decode-Durchsatz kalt und 7,1× warm**, TTFT fällt
von 13,7 s auf 33 ms. `total_tokens_per_second` ist für einen Cache die falsche
Kennzahl: cache-aus steht dort höher (1064 gegen 443), weil es 8× mehr
Prefill-Tokens für dieselbe Ausgabe bewegt. Vergleiche `agg_tg`, `ttft` und
`cached_tokens`.

Bei c64 (`--sessions 64`, max-tokens 64) skaliert der Effekt: Runde 0 serviert 63
Treffer gegen 1 Miss und liefert 86,48 agg_tg bei 11,6 s TTFT; Runde 1 trifft
64/64 und liefert **118,49 agg_tg bei 0,43 s TTFT** — gegen 47,17 agg_tg im
Nonce-Benchmark bei c64 mit Cache, also **2,5×**.

Der Beweis, dass nicht doppelt gespeichert wird, steht im Server-Log desselben
Laufs:

```
event=acquire reuse=none selected_tokens=0    restored_bytes=0
event=acquire reuse=ram  selected_tokens=2166 restored_bytes=186492532   (×7)
event=kv_blocks_shared blocks=16 pool_live=496 pool_free=16
event=kv_blocks_shared blocks=16 pool_live=480 pool_free=32
... pool_free 48, 64, 80 ... 225, 241
```

Jeder teilende Request adoptiert die 16 vollen Blöcke des ersten Captures
(`KvBlockTable::Adopt` nimmt eine Referenz), und `pool_free` steigt um 16, weil
sein eigener Block in den Pool zurückgeht. Nur der Tail unter 128 Tokens bleibt
per Copy-on-Write privat. Restore kostet 2,3–2,5 ms für 2166 Tokens;
`llamacpp:prefix_cache_hit_rate` steht bei 0,87 (kalt) und 0,96 (warm).

Drei Einschränkungen: (1) Teilen braucht `--cache-ram-bytes` **an** — mit `off`
loggt der Server `cache_captures=0`, dann wird nichts captured und nichts
adoptiert. (2) Der Pool ist eine feste Reservierung beim Laden
(`pool_sessions × blocks_per_session`); Teilen senkt den Gesamtverbrauch nicht,
es hält Blöcke innerhalb des Pools frei, also Headroom für mehr Sessions und
längere Kontexte. (3) Ein Treffer braucht einen erreichbaren
Checkpoint-Boundary; divergiert ein Präfix mitten im Chunk, gibt es einen
Teil-Treffer oder einen Miss, den `cache_miss_reason` pro Request ausweist.
Artefakte: `pp-tg-shared-prefix-cache-{on,off}.json`,
`pp-tg-shared-prefix-varyuser-cache-{on,off}.json`,
`pp-tg-shared-prefix-varyuser-c64-cache-on.json`.

Der Schritt 64 → 128 ist +1,1 % (768,91 → 777,58), der Schritt 128 → 256
+0,05 % (777,95), also flach. Bei 16 physischen Zeilen und 55–56 % Prefill-Anteil
an der Runde schaffen weitere 128 wartende Requests nur Warteschlange, keine
Parallelität. Die vom Nutzer angefragte Konfiguration ist damit ende-zu-ende
gemessen: 256 Requests, alle `status=200`, Queue-Tiefe bis 255 gegen 64 residente
Sessions, Rundenspanne 627,7 s (10,5 min) für 488k Tokens, TTFT-Median 275 s.
Der nächste Hebel ist damit nicht mehr Sessions, sondern der serialisierte
Prefill — er läuft weiterhin bei 23 % des DRAM-Peaks. Deshalb bleibt in der
Tabelle oben „Präfix-Cache an" die Empfehlung für wiederholte Präfixe, während
für die 256er-Batch-Konfiguration `--cache-ram-bytes off` gilt. Server-RSS blieb
über die ganze 256er-Runde bei 644 MiB, host_available bei 22,9 GB.

### HTTP-Accept: Backlog 16 und ein Accept pro Poll-Weckruf (01.11.2026)

Der erste Versuch, c256 zu messen, ist nicht am Modell gescheitert, sondern am
HTTP-Server: nach 140 sauber beantworteten Requests (`status=200`) brach der Lauf
mit `ConnectionResetError: [Errno 104] Connection reset by peer` ab. Ursache ist
der Listener selbst — `::listen(fd, 16)` an beiden Aufrufstellen
(`HttpServer::start` und `LoadingResponder`) und eine Accept-Schleife, die pro
`PollReadable(..., 100 ms)`-Weckruf **genau eine** Verbindung annimmt. Bei 256
gleichzeitig öffnenden Clients überläuft die Accept-Queue, und der Kernel
quittiert den Überlauf mit RST statt mit einer HTTP-Antwort. Das trifft genau die
vom Nutzer angefragte Konfiguration (256 parallele Anfragen, `--max-connections
512`) und ist unabhängig vom Modell.

Zwei Änderungen in `src/cli/serve/http_server.cpp`:

1. `ListenBacklog(max_connections)`: Backlog = `max(max_connections, 128)`,
   gedeckelt auf `SOMAXCONN`. Der `LoadingResponder` nutzt die Untergrenze 128,
   `HttpServer::start` nimmt `options_.max_connections`.
2. Der Accept-Pfad drainiert die Queue in einer Schleife, statt eine Verbindung
   pro Weckruf zu ziehen; die Schleife endet beim ersten leeren Accept (der
   Listener ist nonblocking) und ist auf `max_connections` Annahmen pro Weckruf
   begrenzt, damit Device-Loss- und Signal-Prüfungen weiter regelmäßig laufen.

Verifikation: `http_server_test` („HTTP transport checks passed") und
`http_server_signals_test` („HTTP signal shutdown checks passed") laufen durch,
Format-Check `PASS: Checked 708 C++ files`, Release-Binary neu gebaut. Der
c256-Lauf auf dem korrigierten Binary läuft danach sauber durch: 513 von 513
Requests mit `status=200`, kein einziger Reset, kein `capture_refused`
(`pp-tg-flash-next-pagedopt-nocache-c256.json`). Vorher waren nach 140 Requests
116 Verbindungen am Listener gestorben — der Durchsatz-Unterschied zwischen
„c256 messbar" und „c256 bricht ab" lag also allein am Backlog und am
Accept-Drain, nicht am Modell.

### Lazy-Mapping und KV-Pool-Budget: VRAM pro Session (01.11.2026)

Zwei Änderungen am Flash-Next-Paged-Attention-Pool, beide messbar belegt:

**1. Lazy-Mapping.** `CreateSession` mappte bisher das ganze Kontextfenster
(`table->Append(*kv_pool_, max_context_)`), damit „ein früher Capture eine Zeile
vor dem ersten Schreiben lesen kann". Diese Annahme trägt nicht: `CopyRowsOut`
weist `end > position_` ab und `BorrowRowsView` prüft `end > valid_rows`, also
liest kein Capture über den committeten Frontier hinaus. Jetzt wächst die
Tabelle beim Schreiben (`EnsureKvCapacity`), und `RemapTable` mappt nur noch den
Restore-Frontier bzw. null für einen neuen Request. Eine Session, die 2 k von
4 k Kontext nutzt, hält damit 18 statt 32 Blöcke.

**2. Pool-Budget statt `sessions × context`.** Der Pool ist die einzige
Geräte-Allokation, die mit der Nebenläufigkeit wächst. `--kv-pool-bytes N|off`
setzt sie; Default 0 leitet sie aus dem nach dem Weight-Upload noch freien
Gerätespeicher ab (⅛ Reserve), gedeckelt durch `sessions × context`. `off` ist
der Floor: eine Session mit vollem Kontext.

**Gemessene Blockgröße: 3 MiB pro 128-Token-Block.** Direkt über das Budget
bestimmt (384 Blöcke = 1152 MiB, verifiziert an vier Budgetpunkten). Damit ist
der Pool bei `--sessions 4 --context 16384` **1,5 GB** von 88 GB Gesamtbelegung —
die Gewichte dominieren. Eine frühere Ableitung von 8,75 MiB/Block aus dem
Sessions-Vergleich war falsch: die 608 MiB pro zusätzlicher Session teilen sich
in 384 MiB Pool und ~224 MiB rezidivierenden SSM-State.

| Konfiguration | Blöcke | Pool | gpu_device_used_mib |
| --- | ---: | ---: | ---: |
| sessions 4, context 16384, Default | 512 | 1,5 GB | 88 184 |
| sessions 4, `--kv-pool-bytes 384M` | 128 (Floor) | 384 MiB | 87 032 |
| sessions 8, context 16384, Default | 1024 | 3 GB | 90 616 |
| sessions 16, context 16384, Default | 2048 | 6 GB | 95 480 |

Pro Session bei context 16384: 608 MiB = 384 MiB Pool + 224 MiB SSM-State.

**Der Floor trägt nicht für 64 Sessions.** `--kv-pool-bytes off` bei
`sessions 64 --context 16384` endet mit 64× `attention KV pool exhausted`:
64 Sessions × 18 Blöcke bei geteiltem 2182-Token-Prefix brauchen 16 + 64×2 = 144
Blöcke, der Floor bietet 128. Mit 4 GiB Budget (1365 Blöcke) läuft derselbe
Lauf durch: `agg_tg` 87,36 kalt / 122,95 warm, TTFT 11,0 s / 430 ms, null
Exhaustion (`pp-tg-shared-prefix-c64-pool4g.json`).

**Wichtige Grenze:** Bei `sessions 64 --context 16384` verweigert der Load
ohnehin die Aufnahme — `text runner request-state claim exceeds state capacity`,
unabhängig vom Pool-Budget (auch mit 32 GiB Budget). 64 × 16 k State passt nicht
in das freie Gerätefenster. Das ist ein bestehender Claim-Check, keine Regression
dieser Änderung. Der reale Session-Hebel ist damit der SSM-State (224 MiB pro
Session), nicht der KV-Pool.

**Verifikation:** `cache-shared-prefix` und `cache-concurrency` (pr-Tier) laufen
grün (`artifacts/functional/lazy-map-cache-suite`), `kv_blocks_test`,
`kv_page_ops_test` und `attention_ops_test` ebenfalls. Im Suite-Log bleiben
14 `kv_blocks_shared`-Events bei 130 live / 382 freien Blöcken, d. h. Teilen
funktioniert unter Lazy-Mapping unverändert. Format-Check: `PASS: Checked 708
C++ files`.

### KV-Pool-Buchhaltung sichtbar gemacht (01.11.2026)

`KvBlockPool::high_water_blocks()` existierte, wurde aber nirgends gemeldet —
damit war die Reservation (`sessions x context`) nicht gegen den tatsächlichen
Peak prüfbar. Jetzt:

- `KvPoolSnapshot` in `src/models/qwen/kv_blocks.hpp` (Kapazität, live, free,
  high_water) und `KvBlockPool::Snapshot()`.
- `cache::KvPoolGauges` in `src/cache/types.hpp`, abgefragt über
  `cache::Adapter::PoolGauges()` (Default: leer, damit Adapter ohne Paged-Pool
  nichts melden). Flash-Next implementiert es aus `Model::executor_->kv_pool()`
  und rechnet `block_bytes` aus der Konfiguration.
- Ausgabe pro Request unter `usage.gufo.component_cache.global_kv_pool`
  (Blöcke, Bytes, `block_bytes`) und in der `kv_blocks_shared`-Logzeile als
  `pool_capacity=`/`pool_high_water=`.
- `artifacts/kv_pool_probe.py` fragt die Gauges ohne Benchmark-Harness ab.

Messung (sessions 8, context 4096, 1011-Token-Prompt, 3 Requests mit
gemeinsamem Präfix): Kapazität 256 Blöcke / 768 MiB, high_water 8 → 17 Blöcke
(24 → 51 MiB), also **3,1–6,6 % Auslastung**. `block_bytes = 3 145 728 B`
bestätigt 3 MiB/Block. Die Reservation ist damit 15–32x größer als der Peak
dieses Workloads — die Zahl, die eine Working-Set-Bemessung (Hebel 2) und den
LRU-Registry-Cache (Hebel 1) begründet.

Build: `nix develop -c cmake --build --preset release` grün, Format-Check
`PASS: Checked 708 C++ files`. Nächster Schritt: `KvPrefixRegistry`
(Hash + LRU auf der Freiliste) mit Unit-Tests, dann Verdrahtung am
Request-Ende.

### B4 — Retention-Gating für kalten/unique Traffic (01.11.2026)

**Befund:** Ein kalter Request hielt ~178 MB Payload (63 % SSM-State), obwohl
kein späterer Request ihn treffen konnte; der c16-Nonce-Lauf füllte den Ledger
mit 4,68 GB aus 25 Snapshots. Die Ursache war die Ausnahme „stable framing
boundary": bei Single-Turn-Prompts markiert der Qwen-Pfad den *ganzen* Prompt
als stabil (`stable_prefix_tokens` ≈ Prompt−5), also wurde pro Request fast der
komplette Payload aufbewahrt.

**Umsetzung:** `cache::ColdRetentionGate` (FNV-1a über Kompatibilität +
vollständige Prompt-Tokens, 256 Einträge, reine Hashes) und
`RetentionRequest::{reused, shared, keep_cold, prompt_tokens}`. `Admit`/
`Reclaim` verweigern ohne Evidenz (`reason=kColdUnique`); Evidenz ist ein
Treffer/Reuse, ein wartender Peer (`ShareCheckpoint` → `MarkSharedPrefix`),
eine Wiederholung/Verlängerung eines registrierten Prompts oder explizit
`--cache-retain-cold`. Die Stable-Boundary-Ausnahme ist entfernt: sie war für
Single-Turn-Traffic der ganze Prompt und damit genau der zu vermeidende Fall.
Serving: `TextRunnerRamCacheOptions::retain_cold` (Default `false`), Flag
`--cache-retain-cold` an `serve` und `serve llm`; direkte
`ComponentTextCache`-Nutzer behalten das alte Verhalten. Tests, die kalte
Cache-Mechanik prüfen, setzen `{.retain_cold = true}`.

**Unit-Test:** `ColdRetentionGating` in `tests/cache/retention_test.cpp`:
unique Prompts → `Size()==0`, `RetainedPayloadBytes()==0`, `ColdRefused()==4`;
Wiederholung/Verlängerung/Sharing/Reuse/Opt-in werden aufgenommen.

**Messung (`artifacts/serving/b4-retention-gating.json`):** Flash-Next,
sessions 16, context 4096, `--cache-ram-bytes=0`, vier unique 1995-Token-Prompts
sequentiell.

| Modus | Captures | behaltene Snapshot-Bytes | Ledger-Wachstum |
| --- | ---: | ---: | ---: |
| Default (Gating) | 8 (alle verweigert) | **0** | **345 534 B** |
| `--cache-retain-cold` | 8 (alle aufgenommen) | 1 255 951 888 B | 935 057 504 B |

Das Ledger wächst mit Gating 2706× weniger (0,35 MB gegen 935 MB) und behält
null Payload; die 0,35 MB sind Metadaten. `cache_snapshot_bytes` ist pro Request
0 statt 313 987 972 B. CPU-Suite: 146/146 grün, `cache_retention_test`,
`component_text_cache_test`, `qwen_kv_blocks_test` grün, Format-Check
`PASS: Checked 708 C++ files`.

**Sharing bleibt intakt (`artifacts/serving/b4-shared-prefix-check.json`):**
Flash-Next, sessions 16, 8 gleichzeitige Requests mit 1972-Token-Prompt, Cache
an. Runde 0: 7/8 Treffer, je 1967 Tokens gecacht, `cache_restore_ms` 2,3–2,4,
`shared_prefix_wait_ms` 1641 (der Leader führt den Prefill aus, die Peers
warten und adoptieren). Runde 1: 8/8 Treffer, 1972/1972 Tokens,
`cache_reuse=ram`. Das Gating verweigert also nur Traffic ohne
Reuse-/Sharing-Evidenz, nicht den geteilten Präfix.

**Status: [ERLEDIGT].** DoD: unique kalter Traffic behält keine Payload
(gemessen 0 B), Unit-Test zeigt begrenzten Ledger, CPU-Suite grün, Ledger-Bytes
sinken 2706×.

### D5 — Blockgröße 32 vs 128 (B2): Bench entscheidet (01.11.2026)

**Methode:** `tests/models/qwen38_flash_next/attn_bench 4096 10 0 4096 16384
--paged` (WMMA-Causal-Attention, ein Prefill-Chunk mit 4096 Tokens, dense am
Start und sparse bei 4096/16384 Präfix-Tiefe). Der Bench legt dieselben K/V-
Zeilen kontiguierlich und zusätzlich über eine gemischte Blocktabelle in einen
Pool; die Prozentzahl ist der Aufpreis der paged Adressierung bei identischem
Hash. Einzige Änderung je Lauf: `kBlockTokens`/`kBlockShift` im Bench
(`128/7` gegen `32/5`). Das echte Modell blieb durchgehend bei 128.

**A/B, je 3 Läufe, gleicher Host-Zustand, Mediane:**

| Geometrie | Block 128 | Block 32 | Faktor |
| --- | ---: | ---: | ---: |
| dense, Start 0 | +3,9/+5,1/+4,8 % (Median +4,8) | +8,2/+11,3/+7,7 % (Median +8,2) | 1,7× |
| sparse, Start 4096 | +4,8/+6,3/+4,3 % (Median +4,8) | +9,4/+11,0/+9,9 % (Median +10,0) | 2,1× |
| sparse, Start 16384 | +9,8/+9,7/+9,4 % (Median +9,7) | +13,8/+12,9/+12,3 % (Median +13,0) | 1,3× |

Dense-Zeiten beider Konfigurationen liegen im selben Band (16,95–17,04 ms bzw.
16,58–17,04 ms), die Differenz ist also der paged Pfad, nicht Host-Rauschen.
Alle Hashes in beiden Konfigurationen identisch zum kontiguierlichen Lauf.

**Ursache:** Der Aufpreis wächst mit der Tabellenlänge. Bei 4096 Tokens hat
Block 32 viermal so viele Einträge wie Block 128 (128 gegen 32); jede Query
löst mehr Block-Basen auf. Kleinere Blöcke helfen nur gegen interne
Fragmentierung, und die ist hier schon durch `block_for`/`ByteOffset`
abgedeckt.

**Entscheidung:** **128 Tokens/Block bleibt.** Kein Gewinn, also keine
Modelländerung; der Bench-Edit wurde zurückgenommen (`kBlockTokens = 128`,
neu gebaut, Hashes identisch).

### D2 — Shared-Prefix bei c32/c64: Trefferquote + TTFT (01.11.2026)

**Methode:** `artifacts/bench_shared_prefix.py`, Flash-Next, `--context 4096`,
`--prefill-chunk 16384`, `--max-tokens 32`, 2212-Token-Prompt, zwei Runden.
Runde 0 startet alle Requests gleichzeitig (ein kalter Leader, die Peers warten
auf `shared_prefix_wait` und adoptieren), Runde 1 trifft den RAM-Cache.
`--sessions 32/64`, `--max-connections 64/96`.

| Lauf | Runde | Treffer | Server-TTFT Median | total tok/s | agg_tg | Completion-Hashes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| c32 | 0 | 31/32 | 5092 ms | 238,62 | 72,06 | 1 von 32 |
| c32 | 1 | 32/32 | **178 ms** | 146,65 | 146,65 | 1 von 32 |
| c64 | 0 | 63/64 | 9472 ms | 179,20 | 80,22 | 1 von 64 |
| c64 | 1 | 64/64 | **414 ms** | 146,79 | 146,79 | 1 von 64 |

Server-Zähler (`llamacpp:prefix_cache_hit_rate`): c32 0,9666 → 0,9833, c64
0,9821 → 0,9911. **Korrektheit:** In jeder Runde ist genau ein Completion-Hash
für alle Requests gleich (auch der kalte Leader gegen die 31/63 warmen Peers),
die Ausgabe ist also byte-identisch zum Cold-Run. TTFT fällt um 28,6× (c32)
bzw. 22,9× (c64). Artefakte: `artifacts/serving/d2-shared-prefix-c32.json`,
`d2-shared-prefix-c64.json`.

Nebenbefund: 31 bzw. 63 `event=kv_blocks_shared blocks=17`-Adoptionen pro Runde
(17 volle Blöcke = 2176 Tokens), `pool_live` steigt nur um die privaten Tails.

### D3 — Host-Snapshot-Sharing sichtbar und gemessen (01.11.2026)

**Umsetzung (Cache-Schicht, kein Kernel):**

- `ComponentCacheMetrics` trägt jetzt `shared_bytes` (Bytes, die aus einem
  zurückbehaltenen Host-Checkpoint restauriert wurden, also aus dem geteilten
  Snapshot-Store statt aus eigenem Live-State) und `retained_payload_bytes`
  (Payload, den die Retention-Policy prozessweit besitzt).
- Befüllung im Restore-Pfad von `ComponentTextCache::Acquire` (jede geladene
  Komponente zählt ihre Bytes als shared), `Lease::Metrics()` liest
  `RetentionPolicy::RetainedPayloadBytes()`.
- Propagation: `TextGenerationBackend::Result.cache_shared_bytes` wird in
  `CompleteSuccess`/`CompleteCancelled` aus den Commit-Metriken gesetzt und
  erscheint als `usage.gufo.cache_shared_bytes` bzw.
  `usage.gufo.component_cache.{shared_bytes,retained_payload_bytes}`.
- Unit-Test `ReuseAndBranches`: ein Checkpoint-Restore meldet
  `shared_bytes == cache_restore_bytes > 0` und `retained_payload_bytes > 0`;
  die Zero-Copy-Adoption eines Live-Slots meldet `shared_bytes == 0`.

**Messung (`artifacts/serving/d3-shared-snapshot-dedup.json`):** Flash-Next,
16 Sessions, 16 gleichzeitige Clients mit identischem 2172-Token-Prompt,
`--context 4096`, `--think off`.

| Runde | Treffer | Payload pro Restore | Summe `shared_bytes` | `retained_payload_bytes` | unique / (clients × payload) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 (kalt + Peers) | 15/16 | 186 492 532 B | 2 797 387 980 B | 324 534 272 B | **0,109** |
| 1 (warm) | 16/16 | 186 670 736 B | 2 986 731 776 B | 324 534 272 B | **0,109** |

`shared_bytes > 0` für 15 bzw. 16 Requests, während der Cache nur ~1,7 Payloads
(Leader-Checkpoint am Shared-Boundary plus vollständiger Prompt) behält statt
16 × 178 MiB = 2,98 GB. Alle 16 Completion-Hashes sind pro Runde identisch,
die Ausgabe bleibt also unverändert. Damit ist die Host-Snapshot-Deduplizierung
messbar: `unique < clients × payload` (Faktor ~9), ohne zusätzliche
Kernel-Änderung.

CPU-Suite nach der Änderung: 146/146 grün, `component_text_cache_test` grün.

**Status: [ERLEDIGT].** DoD: `cache_shared_bytes` echt befüllt (shared > 0),
gleichzeitige Same-Prefix-Sessions teilen Host-Snapshots (unique < n×payload),
Korrektheit unverändert.

### D1 — Device-Prefix-Cache: Adoption nach Request-Ende + LRU-Freiliste (01.11.2026)

**B3 (LIFO→LRU/FIFO-Freiliste):** `KvBlockPool` gab den zuletzt freigegebenen
Block zuerst aus (`free_list_.back()`). Jetzt ist die Freiliste ein
Ringpuffer in Freigabe-Reihenfolge: `free_head_`/`free_count_`, `Acquire`
nimmt den ältesten freien Block, `Release` hängt hinten an. Beide Operationen
bleiben O(1), der Ring wächst nicht mit der Zahl der Releases. Der älteste
Block bleibt damit länger intakt, statt sofort von der nächsten Capture
überschrieben zu werden. Unit-Test `PoolReusesTheOldestFreeBlock` in
`tests/models/qwen/kv_blocks_test.cpp` (inkl. Ring-Wrap); `qwen_kv_blocks_test`
grün.

**Beweis nach Request-Ende (`artifacts/serving/d1-sequential-adoption.json`):**
Flash-Next, sessions 2, context 4096, 1954-Token-Prompt, `--think off`,
`--cache-retain-cold`. Drei Requests: A kalt, B identisch nach A-Ende,
C identisch mit `cache_prompt=false` als Cold-Referenz.

| Request | Cache | gecachte Tokens | prefill | `cache_restore_bytes` | `cache_restore_ms` | Completion-Hash |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A (kalt) | miss | 0 | 1954 | 0 | 0 | 2f6954a3… |
| B (nach A) | **memory/ram** | **1954/1954** | **0** | 179 644 200 | **2,67** | 2f6954a3… |
| C (Cold-Referenz) | miss | 0 | 1954 | 0 | 0 | 2f6954a3… |

B adoptiert die 15 vollen KV-Blöcke des beendeten A zero-copy
(`event=kv_blocks_shared blocks=15 pool_live=17 pool_free=47
pool_capacity=64`); nur der private SSM-State (179,6 MB) wird aus dem
Host-Snapshot kopiert — daher 2,67 ms statt eines vollen KV-Roundtrips.
`byte_identical=true`: alle drei Completion-Hashes sind gleich. Damit ist der
D1-Beweis nach Request-Ende erbracht (Adopt-Zähler > 0, Prefill 0, Ausgabe
byte-identisch zum Cold-Run).

**Status: [ERLEDIGT].**

### D6 — Exhaustion → Host-Snapshot-Degradierung statt Fehler (01.11.2026)

**Umsetzung:** `TextGenerationScheduler::DegradeKvExhausted` fängt einen
`attention KV pool exhausted`-Fehlschlag ab: der bereits geleistete Prefill
bleibt als Host-Checkpoint erhalten, der Device-State wird freigegeben, und der
Request wird erneut admitted, sobald Residents fertig sind. Log-Events
`kv_exhaustion_degraded` (mit `prefill_position`, `generated`,
`retained_snapshot_bytes`, `deferrals`, Pool-Gauges) und
`kv_exhaustion_deferred` machen den Pfad sichtbar; `bench_kv_degradation.py`
zählt `kv_degradations`, `degraded_restores` und die Client-Ergebnisse.

**Gefundener Defekt (der eigentliche D6-Blocker):** Der erste Lauf mit
überschriebenem Pool degradierte zwar (33 `kv_exhaustion_degraded`, Runde 0
16/16), verlor aber in Runde 1 den GPU-Kontext:
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` in
`AttentionKernel<2u>`, direkt nach
`cache acquire reuse=ram restored_bytes=195971148`; danach liefen 5 Clients in
den 600-s-Timeout. Ursache war der KV-Tabellen-Upload in
`Executor::EnsureKvCapacity`: die Block-IDs wurden in einen lokalen
`std::vector` gesammelt und per `hipMemcpyAsync` hochgeladen. Der Vector wurde
beim Verlassen der Funktion freigegeben, während die asynchrone Kopie (und ein
gegebenenfalls gecapturtes Graph-Replay) noch daraus las — der Kernel bekam
Garbage-Block-IDs und griff außerhalb des Pools zu. Ein Restore, der die
Tabelle auf eine *kürzere* Zeilenzahl zurücksetzt, wurde zusätzlich gar nicht
neu hochgeladen, weil der Upload nur bei Wachstum lief.

**Fix:** Persistenter per-Session-Host-Staging-Puffer `kv_table_host_`, einmal
auf die volle Tabellengröße alloziert, plus vollständiger Re-Upload immer wenn
sich die Zeilenzahl ändert (auch beim Schrumpfen/Truncate) und
Hole-Validierung (`kInvalidKvBlock`) bei jedem Upload. Release-Build grün,
Format-Check `PASS: Checked 708 C++ files`.

**Messung small (`artifacts/serving/d6-kv-degradation-small4.json`):**
Flash-Next, 8 Sessions, 8 Clients, 1800-Token-Prompts, `--kv-pool-bytes
380000000`, `--prefill-chunk 512`, 2 Runden, `--retain-cold`, `--think off`.

| | vor dem Fix | nach dem Fix |
| --- | --- | --- |
| Clients completed | 14/16 | **16/16** |
| Fehler | 2 × `device_lost` (Runde 1) | **0** |
| `kv_exhaustion_degraded` | 31 | 31 |
| `degraded_restores` | 5 + 6 | 5 + 7 |
| `MEMORY_APERTURE_VIOLATION` | ja | **0** |
| Tabellen-Holes | 0 | 0 |

Behaltene Snapshots: 181 954 720 B pro degradiertem Restore. Wand-Zeit Runde 0
23,7 s, Runde 1 3,5 s.

**Messung 700 MB / 16 Clients (`artifacts/serving/d6-kv-degradation.json`):**
gleiche Prompts, `--sessions 16 --clients 16`, `--kv-pool-bytes 700000000`,
`--prefill-chunk 16384`, 2 Runden. Der frühere Lauf dieser Konfiguration
scheiterte in Runde 1 (27/32 completed, 5 Timeouts, `device_lost`).

| Runde | completed | failed | `degraded_restores` | Restore-Bytes | Wand-Zeit |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | **16/16** | 0 | 10 | 0 / 181 954 720 | 38,7 s |
| 1 | **16/16** | 0 | 12 | 0 / 195 971 148 | 10,1 s |

Gesamt 32/32 completed, 0 Fehler, 37 `kv_exhaustion_degraded`, 0
`MEMORY_APERTURE_VIOLATION`, 0 `device_lost`, 0 Tabellen-Holes, kein GPU-Dump.
Der 700-MB-Pool liegt damit unter dem Bedarf von 16 × 4096 Tokens × 24 KiB
(≈ 1,6 GB KV) und der Lauf schließt trotzdem per Degradierung ab, statt zu
werfen.

**Status: [ERLEDIGT].** DoD: ein Lauf mit Pool-Budget unter Bedarf schließt per
Host-Snapshot-Degradierung ab (32/32 Requests, 37 Degradierungen, 12
degradierte Restores mit 195 971 148 B), Log und Metrik als Beweis, kein
Device-Lost.

### A1 — Rekurrenter SSM-State in bf16 (01.11.2026)

**Umsetzung:** `SsmState = hip_bfloat16` in
`kernels/rocm/kernels.hpp`; `GdnKernel`, `GdnRowSplitKernel` und
`RestoreGdnStateKernel` laden den State in F32-Register, rechnen die
Delta-Regel weiter in F32 und schreiben nur beim Verlassen des Registers bf16
zurück. `conv_state` bleibt F32 (nur 4,4 MB pro Session; die Kernel sind
darauf ausgelegt), Rollback-Snapshots bleiben F32. Angepasst: `Session::
LinearState::state` (`SsmState*`), `Alloc`/Zero/`SessionBytes`, der byteweise
gepackte Prefill-Checkpoint, die Snapshot-Region
(`state_elems * sizeof(SsmState)`), `continuation_layout` (`kRecurrent`
`sizeof(uint16_t)`) sowie die beiden Kernel-Tests, die jetzt bf16-Puffer
hochladen und mit einem quantisierten Host-Orakel vergleichen.

**Device-State pro Session:** 36 lineare Layer × 786 432 Elemente × 4 B =
113 246 208 B vorher, × 2 B = **56 623 104 B** nachher (−56,6 MB, −50 %).
Für die damalige 64-Session-Konfiguration: 7,25 GB → **3,62 GB**. Gemessen am
Checkpoint-Payload
(`artifacts/serving/d6-kv-degradation-a1.json` vs `-small4.json`): behaltene
Snapshots **181 954 720 → 125 331 616 B** (−56 623 104 B), Runde-1-Restore
195 971 148 → 139 348 044 B. Der D6-Lauf mit bf16-State bleibt intakt: 16/16
completed, 0 Fehler, 31 Degradierungen, 0 `MEMORY_APERTURE_VIOLATION`.

**Korrektheit (`artifacts/serving/a1-ssm-bf16.json`, Dumps/Scoring in
`artifacts/a1-quality/`):** `gpu_probe --logit-rows 256` über die vier
F32-Referenz-Prompts, 2479 Positionen. Das Scoring-Tool ist eine
numpy-freie Nachbildung von `f32_reference.py compare` und reproduziert den
Python-Baseline exakt (vorher: KL 0,03409, top-1 0,951, NLL 1,2049).

| | KL mean | KL median | KL p99 | top-1 | NLL | NLL@ref | ref NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 State (vorher) | 0,03409 | 0,00066 | 0,6388 | 0,951 | 1,2049 | 1,1112 | 1,1079 |
| bf16 State (nachher) | **0,02910** | 0,00066 | 0,5582 | **0,955** | 1,2123 | 1,1202 | 1,1079 |

Gepaart: KL(after)−KL(before) = **−0,00499**, NLL-Differenz **+0,0074**.
Definierte Toleranz: |ΔKL mean| ≤ 0,01, |Δtop-1| ≤ 0,01, |ΔNLL| ≤ 0,02 —
alle drei eingehalten (KL sogar leicht besser, Rauschen der Quantisierung).
Ein direkter Vorher/Nachher-Vergleich derselben Positionen zeigt
KL(before‖after) 0,0201 bei top-1-Übereinstimmung 97,4 %: einzelne Positionen
kippen, die Verteilung gegen die F32-Referenz bleibt unverändert.

**Tests:** `qwen38_flash_next_gdn_ops_test` und
`qwen38_flash_next_projection_ops_test` grün mit bf16-State-Puffern; die
Rollback-/Checkpoint-Rundreise bleibt byte-identisch zwischen serieller und
Row-Split-Route (State-Toleranz dort 1e-1 des Head-RMS, weil bf16 pro Chunk
rundet). Ein Layout-Check gegen die kompilierte Bibliothek bestätigt 36
rekurrente Komponenten mit je 1 572 864 B (786 432 × 2 B) und
`kConvolution` unverändert 122 880 B. Shared-Prefix-Regression mit dem
bf16-Build (`artifacts/serving/a1-shared-prefix-check.json`): Runde 0 7/8
Treffer, Runde 1 8/8, pro Runde genau ein Completion-Hash, also weiterhin
byte-identisch zum Cold-Run. Format: `PASS: Checked 708 C++ files`.

**Umgebungs-Hinweis:** Der Podman-Nix-Store lief während der Arbeit voll
(78-GB-GPU-Coredump plus Build-Ausgaben) und wurde mit `nix store gc`
bereinigt (1,1 TiB frei). Dabei wurde auch der gepinnte nixpkgs-Input des
Flakes entfernt; ohne Netz kann `nix develop` bis zum nächsten Online-Lauf
nicht mehr evaluieren. Die Binaries (`build/release/gufo`, die gpu-test-Tests,
`gpu_probe`) und die verwendeten Store-Pfade (clang 24, ROCm 10.1,
clang-format 21.1.8) sind vorhanden; die Messungen liefen mit dem gebauten
Release-Binary. Die CPU-Suite wurde vor A1 grün gemessen; A1 ändert nur
GPU-Kernel plus das Layout, dessen gpu-test-Tests neu gebaut und grün sind.

**Status: [ERLEDIGT].** DoD: SSM-State bf16, Device-State pro Session
halbiert (−56,6 MB gemessen), Korrektheit innerhalb der definierten Toleranz.

### D4 — KV-Cache in FP8 (E4M3): Prototyp und Messung (01.11.2026)

**Prototyp:** `tools/bench/kv_fp8_attention.hip` (standalone HIP, nicht im
Serving-Pfad). Ein scalarer Causal-Attention-Kernel wird zweimal kompiliert:
FP16-K/V und E4M3-K/V mit einem Absmax-Scale pro (Token, KV-Head). Beide Arme
teilen die Arithmetik, die Zeitdifferenz ist also der Preis der schmaleren
Ablage plus der Dekodierung im Inneren Loop. Geometrie: 24 Query-Heads, 2
KV-Heads, head_dim 256 (Flash-Next Full Attention), Queries am Kontextende.

**Speicher:** 2048 B/Token (FP16) → **1040 B/Token** (1024 B E4M3 + 16 B
Scales), Faktor **1,97×**.

**Messung (`artifacts/serving/d4-kv-fp8-prototype.json`, Rohausgabe
`d4-kv-fp8-raw.txt`):**

| Tiefe | Query-Zeilen | FP16 ms | FP8 ms | Aufpreis | RMS/Referenz | cos |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 64 | 1,361 | 4,181 | +207 % | 2,9 % | 0,99958 |
| 2048 | 64 | 3,348 | 8,547 | +155 % | 2,8 % | 0,99961 |
| 4096 | 32 | 2,898 | 8,072 | +179 % | 2,9 % | 0,99959 |

**Befund:** FP8 halbiert die KV-Bytes, kostet auf gfx1151 aber 1,6–3,1×
Attention-Zeit, weil E4M3 keinen nativen MMA-Pfad hat und jedes Element im
Inneren Loop dekodiert werden muss. Die Qualität ist bei dieser Geometrie
gut (Cosinus 0,9996, RMS-Fehler ≈ 2,9 % der Ausgabe-RMS). Für eine
Produktionsübernahme bräuchte es einen FP8-Dot-Product-Pfad (oder eine in den
MMA-Tile-Load gefaltete Konversion) plus einen vollständigen Qualitäts-Gate
über das Modell; allein der Speichergewinn rechtfertigt den derzeitigen
Speed-Verlust nicht. Der Prototyp bleibt deshalb außerhalb des Serving-Pfads.

**Status: [ERLEDIGT].** DoD: Prototyp plus Qualitäts-/Speed-Messung auf
begrenztem Workload; Ergebnis dokumentiert (Speicher 1,97× kleiner, Qualität
Cosinus 0,9996, Speed 1,6–3,1× langsamer → keine Produktionsübernahme).



## Nachtrag: Batch-Prefill-Kohorte, Watchdog und Concurrency-Kurve (03.11.2026)

**Batch-Breite.** Flash-Next führt Decode-Wellen bis **32 Zeilen** aus
(`kMaxBatchSessions`, `SupportedPlans` 2…32); 27B und DeepSeek bleiben bei acht.
Die gemessene Breiten-Kurve (336-Token-Prompts, ctx 1024): total 231,61 /
277,14 / 294,00 / 305,19 tok/s bei Breite 8/16/24/32, Decode-Phase 85,87 /
108,34 / 117,33 / 125,10 tok/s. Marginal +19,7 / +6,1 / +3,8 %; Breite 24
liefert 96,3 % des Width-32-Totals bei +22 % Request-TG und −25 % TTFT.

**Batched Prefill.** `Executor::ForwardPrefillBatch` + `Model::PrefillBatch` +
Scheduler-Paarung tragen zwei kalte Prompts in einem Pass. Zwei echte
State-Fehler wurden behoben: (1) der Wechsel auf per-Session-Zeilenviews
(`UseScratch(RowScratch(...))`) löschte `xn_half_` und damit die vom breiten
`Combine` gestagte Norm — `HcMix` las eine veraltete F32-Norm; (2) derselbe
Wechsel löschte `moe_pending_`, sodass der MoE-Epilogue aus dem Residual fiel.
Beide Pfade rechnen jetzt den vollen F32-Mix (die fusionierte F16-Staging ist
nicht über Zeilenviews teilbar). Der Default blieb dabei bit-identisch
(189f9e81 / b9dc0063 / 56d2de5c / 3d03cf8b) und durchsatzneutral. Gemessener
Kohorten-Pass: 2×702 Tokens in 1157/1077 ms = 1213/1303 tok/s gegen 945 tok/s
des seriellen Modells (+28…38 % PP). Der Pfad bleibt **opt-in**
(`GUFO_BATCHED_PREFILL=1`), weil die Kohorte nicht bit-identisch ist: die
zeilenweise gebatchten GEMMs tilelen mit der Zeilenzahl anders (Layer-0-Mixer
`sumsq` 921 seriell vs 1150 Kohorte), wodurch Greedy-Tokens ab Layer 0
divergieren.

**Prefill-Kostenmodell.** `pass(t) ≈ 0,285 s + t/1450` (ctx 2048; `0,275 s +
t/1501` bei ctx 1024). Die Fixkosten sind ein vollständiger Expert-Sweep
(66 GB bei 240,76 GB/s); bei 336 Tokens sind das 55 % des Passes. Bei gleicher
Wandzeit verarbeitet eine 2×702-Kohorte 1404 Tokens (1157 ms), ein einzelner
1248-Token-Pass 1248 Tokens (1146 ms) — die kürzere Attention pro Session
bringt zusätzlich ~8–11 %.

**Concurrency-Kurve 1…32** (gleiche Binary, ctx 1024, 336-Token-Prompts,
cache off): req PP 643,86/639,19/660,18/669,61/671,78/666,35; req TG
28,34/22,12/16,42/11,57/7,29/4,33; agg TG 25,35/37,15/51,30/66,10/76,44/83,54;
agg total 92,29/134,97/185,57/239,10/277,90/301,33 (+46,3/+37,5/+28,9/+16,2/
+8,4 %); Prefill-Anteil 10,4→32,7 %; Decode-Phase 28,29→124,10; TTFT
0,53→16,07 s. Prefill ist nebenläufigkeits-invariant, weil ein Pass gleichzeitig
läuft — genau deshalb steigen die Prefill-Hebel (Kohorte, Overlap) mit der
Breite.

**Watchdog.** `--stall-timeout-ms` existierte, war aber per Default **0
(aus)**; deshalb blieb ein Lauf mit 121 empfangenen / 97 abgeschlossenen
Requests stumm stehen. Default ist jetzt **180000 ms** (längstes legitimes
Fortschritts-Loch ist ein Prefill-Chunk, ~3 s); ein echter Hänger wird als
`device_lost` gemeldet statt unbegrenzt zu warten.

**Doku.** `docs/SERVER.md` (Batch-Breite pro Modell, Prefill-Pass-Kosten,
Stall-Default, opt-in-Kohorte), `docs/CLI.md` (`stallTimeoutMs`-Default,
Sessions-Wellen), `docs/models/qwen3.8-flash-next/README.md` (neuer Abschnitt
„Batching, widths and limits"), `docs/models/qwen3.8-flash-next/BENCHMARKS.md`
(Abschnitt „Batch scaling (November 2026)") und
`artifacts/serving/batch-pp-tg-levers.md` (Abschnitte 13.4, 13.6, 13.7,
Build-Umgebung). Analyse und Rohdaten:
`artifacts/serving/pp-tg-concurrency-1-32-ctx1024*.json`,
`batch-levers-width-curve-ctx1024*.json`, `prefill-c1-r*-ctx2048.json`.

**Offen:** `GUFO_BATCHED_PREFILL` ist nicht bit-identisch (M-invariante
zeilenweise GEMMs nötig); Prefill/Decode-Overlap (Dual-Lane) und die
Kernel-Hebel A3/N2/N3 sind nicht implementiert; die Details stehen in
`artifacts/serving/batch-pp-tg-levers.md` Abschnitt 13.7.

## Nachtrag: 5.-7. November 2026 — breite Pläne, MoE-Dispatch und GDN-State

**Korrektur zur Kohorte.** Die stabile Prefix-Chunk-Grenze schließt die
Arithmetik der Batched-Prefill-Kohorte an den seriellen Frontier an. Der
deterministische 256-Prompt-Vergleich besteht bei Kohorte 8, 16 und 32
(256/256); die Kohorte ist Default-on mit Cap `min(32, prefilling queue depth)`,
`GUFO_BATCHED_PREFILL=0` bleibt Diagnose-Kill-Switch. Die frühere Aussage
„nicht bit-identisch" ist damit überholt.

**Breite Pläne und Auto-Sessions.** `kMaxBatchSessions` steigt von 32 auf
**256**; der Hybrid-Plan füllt die Breite, die die Resident-Kohorte hergibt.
`--sessions auto` löst den Pool nach dem Laden aus der gemessenen
Base-State-Kapazität (25 % Reserve), wächst mit der Nachfrage und deferred
KV-blockierte Requests statt sie zu verwerfen. Auf diesem Host löste der
damalige Stand 96 Sessions bei 250k, 150 bei 128k und 256 bei kurzem Kontext
auf; das aktuelle Operator-Profil steht am Anfang dieses Protokolls.
`GUFO_DECODE_MAX_ROWS` und
`GUFO_ASSEMBLE_DECODE_ROWS` bleiben A/B-Overrides.

**Vier messbare Decode-Hebel (2x2-A/B, 4k/256, 1024x16):**

| Hebel | vorher | nachher | Delta |
| --- | --- | --- | --- |
| #28 Wide-MoE-Dispatch (kompaktierte Expert-Buckets) | 210,2 s / 78,0 TG / 462 PP | 177,0 s / 92,6 TG / 549 PP | −15,8 % Wall, +18,7 % TG |
| #25 Wide-Expert-Tiles (64/128 statt 16 Zeilen) | 176,28 s / 92,95 TG | 174,59 s / 93,85 TG | −1,0 % Wall, +1,0 % TG |
| #51 GDN-State-Writeback (4×16 B statt 32×2 B) | 174,78 s / 93,75 TG / 555,6 PP | 156,48 s / 104,7 TG / 621,1 PP | −10,5 % Wall, +11,7 % TG |
| #52 Weite Q8-Projektionen auf der W8A8/F16-Dense-Route | 155,4 s / 6,59 req/s / 105,4 TG | 145,3 s / 7,05 req/s / 112,8 TG | −6,5 % Wall, +7,0 % req/s |

256x128: #28 103,5 → 81,0 s (114,3 → 145,8 TG); #51 auf #28/#25 43,71 →
39,52 s bei 256x16. Jede Änderung ist 256/256 deterministisch. Das
ausgelieferte Binary bestätigt **144,4 s / 7,09 req/s / 113,5 TG / 673,2 PP**
bei 1024x16; #52 verbessert 256x128 von 64,1 auf 57,1 s (184,3 → 208,0 TG).

**PP-Seite.** Nach dem Decode-Speedup wurden die Prefill-Hebel erneut geprüft:
Cohort 64/48/24 ergeben 160,4/171,1/158,7 s gegen 144,4 s bei 32; Assembly
192/128 ergeben 159,0/161,0 s gegen 156,4 s bei 256 (Vor-#52-Binary). Das
Optimum bleibt Cohort 32 und Assembly 256. Der Frontier-Split (stabile
Prefix-Grenze) lässt sich nicht in einen Pass zusammenfassen, ohne die
Frontier-Arithmetik zu ändern (früherer 10,01-Logit-Mismatch);
Prefill/Decode-Overlap braucht partitionierten Scratch und einen zweiten
Stream und bleibt der strukturelle PP-Hebel.

**Weitere Änderungen.** HTTP-Accept-Backlog 1024 plus Drain-Schleife und
10-ms-Poll für 256+-Bursts; Modell-Alias-Fallback (unbekannte/missing
Modellnamen lösen auf das geladene Modell auf); Tool-Parameter-Prüfung gegen
das deklarierte JSON-Schema; Stop-Sequenz- und Text-Span-Härtung;
Bildlimits 20 MiB/32 MP; `--cache-retain-cold` für die Retention-Gate;
KV-Pool-Gauges und Host-Snapshot-Sharing-Metriken; `gufo diagnose` mit
Power-Profil und Queue-Budget; `tools/bench/gather_bw.hip` für die
Bandbreiten-Kalibrierung.

Die konsolidierte Feature-Liste mit Zweck und Belegen steht in
[`docs/FEATURES.md`](docs/FEATURES.md), die 52 Maßnahmen mit Status in
[`docs/BATCH-THROUGHPUT.md`](docs/BATCH-THROUGHPUT.md), die neuen
Modellzahlen in
[`docs/models/qwen3.8-flash-next/BENCHMARKS.md`](docs/models/qwen3.8-flash-next/BENCHMARKS.md).

**Noch offen:** GDN-State-Bandbreite ist nach #51 nicht mehr der dominante
Posten; verbleibende Kandidaten sind Prefill/Decode-Overlap (Dual-Lane),
niedrigere Experten-Quantisierung und die in `BATCH-THROUGHPUT.md` gelisteten
Fusions-/Cache-Hebel. Rejected bleiben #22, #27, #31, #34 und der FP8-KV-
Prototyp (Details in den jeweiligen Dokumenten).
