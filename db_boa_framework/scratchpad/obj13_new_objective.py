class _ADTCNObjective:
    """
    Wraps ADTCN evaluation as a callable fitness function for DB-BOA.

    Uses the same _Conv1dClassifier architecture as the final model, trained on
    a stratified subsample of `_SURROGATE_ROWS` rows for _SURROGATE_EPOCHS
    epochs.  DB-BOA minimises, so we return −Obf2.

    Epoch count is NOT part of the search space: the surrogate hard-caps at 5
    epochs regardless of what DB-BOA proposes, making that dimension flat.

    Hyperparameter mapping
    ----------------------
    params[0]  →  n_filters          (Conv1d channel count)
    params[1]  →  steps_per_epoch    (gradient steps per epoch → batch_size)
    epoch count is fixed at _SURROGATE_EPOCHS for the surrogate; the final
    model uses ADTCN_CONFIG["epoch_count"].

    OBJ-13 — what was wrong with this objective, and what changed
    -------------------------------------------------------------
    `objective_noise_audit.py` established three defects, all of which made the
    *search* meaningless while leaving the final training perfectly
    deterministic (that was OBJ-2's separate finding, and it still holds):

    1.  **Fitness was a random function of its input.**  `__call__` redrew both
        the 70/30 split permutation and the torch seed from `self.rng` on every
        call, and `self.rng` advances, so evaluating the *same* candidate twice
        gave different scores.  A fixed config re-evaluated 25× on ULB spanned
        Obf2 3.4497–5.0000 and touched the 5.0000 ceiling once **with no search
        involved** — which is why all five optimisers "found" exactly 5.0000.
    2.  **The validation split was not stratified.**  `_MIN_FRAUD_ROWS = 30`
        fires on both datasets, so the surrogate holds exactly 30 fraud rows and
        an unstratified 70/30 leaves ≈9 of them in validation — varying, and in
        principle zero.  Obf2 = 5.0000 means classifying about nine rows.
    3.  **One search axis was arithmetically dead.**  `batch_size = max(32,
        n_train // spe)` with n_train = 1400 and spe ∈ [50, 250] gives 28 … 5,
        every one of them below the 32 floor — so batch_size was **32 for every
        spe in the search range** and the advertised 2-D search was 1-D.

    The repair
    ----------
    `eval_mode` selects between the three behaviours, and running the first two
    side by side **is the result**: the gap between them measures how much of
    the original DB-BOA behaviour was noise-chasing rather than optimisation.

    ``"legacy"``
        The pre-repair behaviour, bit-for-bit.  Kept so the old numbers remain
        reproducible and the comparison is against a live baseline rather than
        a remembered one.  Never the default.
    ``"deterministic"``
        Split permutation and torch seed are fixed at construction, so fitness
        is a **pure function of the candidate**.  1× cost, ranking becomes
        reproducible — but it optimises one arbitrary draw, and that limitation
        is the reason "averaged" exists rather than something to hide.
    ``"averaged"``
        `k_repeats` draws, fixed at construction and **shared by every
        candidate** (common random numbers), returning the mean.  Optimising an
        expectation kills the best-of-N ceiling artefact: a lucky draw can no
        longer win, because every candidate is scored on the same draws.  k×
        cost.

    Two details that matter for honesty:

    *   The draws are pre-drawn in `__init__` from a dedicated RandomState, so
        candidate *i* and candidate *j* are compared on identical data and
        identical initialisation.  Re-drawing per candidate would leave the
        search comparing a good config on an easy draw against a good config on
        a hard one, which is the original defect wearing a mean.
    *   Both repaired modes stratify the 70/30 split, so fraud appears in both
        halves at every surrogate size.  Without this, "more rows" and "fraud
        actually present in validation" would move together and the size knee
        below would not be attributable.

    **Pre-registered expectation (rule 4, written before running).**  A repaired
    surrogate is a *better-behaved* DB-BOA, not necessarily a winning one.  The
    five-optimiser tie already survives on BankSim, where the objective's mean is
    at least unbiased.  If DB-BOA still ties the hand-set default after the
    repair, that is the honest outcome and rule 3 applies — **this is diagnosis
    of a negative result, not a rescue attempt.**
    """

    _SURROGATE_ROWS   = 2_000   # rows per evaluation (speed vs. fidelity trade-off)
    _SURROGATE_EPOCHS = 5       # surrogate training epochs (not searched)
    _MIN_FRAUD_ROWS   = 30      # minimum fraud samples; guards against near-zero count at 0.17% rate

    EVAL_MODES = ("legacy", "deterministic", "averaged")

    def __init__(self, X_opt, y_opt, random_state: int = 42,
                 architecture: str = "cnn", n_raw: int = None,
                 eval_mode: str = "deterministic", k_repeats: int = 3,
                 surrogate_rows: int = None):
        if eval_mode not in self.EVAL_MODES:
            raise ValueError(f"eval_mode must be one of {self.EVAL_MODES}, got {eval_mode!r}")
        self._eval_mode = eval_mode
        self._k = 1 if eval_mode in ("legacy", "deterministic") else max(1, int(k_repeats))
        n_rows = int(surrogate_rows or self._SURROGATE_ROWS)
        self._n_rows = n_rows

        rng   = np.random.RandomState(random_state)
        # `n_raw` must match what the FINAL model will see, or the search tunes
        # a different problem from the one it hands over.  The default caps at
        # 33, which is right for ULB (the tail of its matrix is the PTC/NTC
        # block the CNN never consumes) and wrong for any dataset whose columns
        # are all real features — BankSim's 79 would be cut to 33.  Callers on
        # such datasets pass the loader's `raw_feature_count`.
        n_raw = int(n_raw) if n_raw else min(X_opt.shape[1], N_RAW_FEATURES + 3)
        n_raw = min(n_raw, X_opt.shape[1])
        self._n_raw = n_raw
        self._architecture = architecture

        # Subsample preserving the real class distribution
        fraud_idx  = np.where(y_opt == 1)[0]
        normal_idx = np.where(y_opt == 0)[0]
        total      = len(fraud_idx) + len(normal_idx)
        fraud_rate = len(fraud_idx) / total
        # Minimum of _MIN_FRAUD_ROWS (30) ensures the CNN receives enough
        # positive examples for stable gradient estimates even when the input
        # has the real 0.17% fraud rate.  When the fraud pool is smaller than
        # n_f, we sample with replacement so the selection doesn't raise.
        # NOTE (OBJ-13): this floor fires on BOTH datasets, so the surrogate
        # holds exactly 30 fraud rows either way — the fraud *rate* is not what
        # separates ULB from BankSim here; separability at n≈9 validation
        # positives is.
        n_f = max(self._MIN_FRAUD_ROWS, int(n_rows * fraud_rate))
        n_n = min(len(normal_idx), n_rows - n_f)
        replace_f = len(fraud_idx) < n_f   # allow repetition when pool is too small
        idx = np.concatenate([
            rng.choice(fraud_idx,  n_f, replace=replace_f),
            rng.choice(normal_idx, n_n, replace=False),
        ])
        rng.shuffle(idx)
        X_sub = X_opt[idx, :n_raw].astype(np.float32)
        y_sub = y_opt[idx]

        # Pre-build sequences once so __call__ only trains the CNN
        pad   = np.repeat(X_sub[:1], SEQ_LEN - 1, axis=0)
        X_pad = np.vstack([pad, X_sub])
        self.X_seq = np.stack(
            [X_pad[i : i + SEQ_LEN] for i in range(len(X_sub))], axis=0
        ).astype(np.float32)   # (n_sub, SEQ_LEN, n_raw)
        self.y   = y_sub
        self.rng = rng
        self._call_count = 0
        self.surrogate_fraud_rows = int(y_sub.sum())

        # Pre-draw the evaluation draws ONCE.  Every candidate is then scored on
        # the same splits and the same initialisation — common random numbers —
        # so a difference in fitness is a difference in the candidate.
        self._draws = []
        if eval_mode != "legacy":
            drng = np.random.RandomState(random_state + 1)
            for _ in range(self._k):
                tr, vl = self._stratified_split(drng)
                self._draws.append((tr, vl, int(drng.randint(0, 2 ** 31))))
        self.last_scores = None

    # ── draws ────────────────────────────────────────────────────────────────

    def _stratified_split(self, rs):
        """
        70/30 split that keeps the class ratio in BOTH halves.

        The pre-repair split was a plain permutation over ~2,000 rows holding
        exactly 30 fraud, so validation received ≈9 positives — a count that
        varied per call and could in principle be zero, at which point MCC is
        undefined and Obf2 is meaningless.  Stratifying removes that as a source
        of between-call variance, which is a precondition for the size sweep
        below meaning anything.
        """
        out_tr, out_vl = [], []
        for cls in (0, 1):
            idx = np.where(self.y == cls)[0]
            idx = idx[rs.permutation(len(idx))]
            cut = int(round(0.7 * len(idx)))
            # Guarantee at least one row of each class on each side where the
            # class exists at all, so the metric is always defined.
            cut = min(max(cut, 1), len(idx) - 1) if len(idx) >= 2 else len(idx)
            out_tr.append(idx[:cut])
            out_vl.append(idx[cut:])
        tr = np.concatenate(out_tr)
        vl = np.concatenate(out_vl)
        return tr[rs.permutation(len(tr))], vl[rs.permutation(len(vl))]

    def _legacy_draw(self):
        """The pre-repair draw: a fresh unstratified permutation every call."""
        n       = len(self.X_seq)
        n_train = int(0.7 * n)
        idx     = self.rng.permutation(n)
        return idx[:n_train], idx[n_train:], int(self.rng.randint(0, 2 ** 31))

    # ── scoring ──────────────────────────────────────────────────────────────

    def _score_once(self, n_filters, spe, tr_idx, vl_idx, torch_seed):
        """Train one surrogate on one draw and return Obf2 (higher is better)."""
        X_tr_t = torch.tensor(self.X_seq[tr_idx], dtype=torch.float32)
        y_tr_t = torch.tensor(self.y[tr_idx],     dtype=torch.long)
        X_vl_t = torch.tensor(self.X_seq[vl_idx], dtype=torch.float32)
        y_vl   = self.y[vl_idx]

        n_train = len(tr_idx)
        if self._eval_mode == "legacy":
            # The dead axis, preserved exactly: with n_train = 1400 and
            # spe ∈ [50, 250] this is 32 for every spe in the search range.
            batch_size = max(32, n_train // spe)
        else:
            # `steps_per_epoch` means what it says: spe gradient steps per
            # epoch, so the axis is alive across the whole search range
            # (spe=50 → batch 28, spe=250 → batch 6).  Removing the floor is
            # the fix; the floor WAS the bug.
            batch_size = int(max(1, math.ceil(n_train / max(1, spe))))

        n_fraud  = int(y_tr_t.numpy().sum())
        n_normal = len(y_tr_t) - n_fraud
        w_fraud  = n_normal / max(n_fraud, 1)
        cw       = torch.tensor([1.0, w_fraud], dtype=torch.float32)

        try:
            torch.manual_seed(torch_seed)
            net       = make_temporal_model(n_features=self._n_raw,
                                            n_filters=n_filters,
                                            architecture=self._architecture)
            criterion = nn.CrossEntropyLoss(weight=cw)
            optimizer = optim.Adam(net.parameters(), lr=1e-3)
            loader    = DataLoader(
                TensorDataset(X_tr_t, y_tr_t),
                batch_size=batch_size, shuffle=True,
            )
            net.train()
            for _ in range(self._SURROGATE_EPOCHS):
                for Xb, yb in loader:
                    optimizer.zero_grad()
                    criterion(net(Xb), yb).backward()
                    optimizer.step()
            net.eval()
            with torch.no_grad():
                y_pred = net(X_vl_t).argmax(dim=1).numpy()
        except Exception:
            return None

        # Bounded fitness  Obf2 = 2*MCC + Spec + Pre + NPV, defined once in
        # utils.metrics.obf2_value (which also records why the paper's unbounded
        # 1/FPR form was replaced).
        return obf2_value(compute_all_metrics(y_vl, y_pred))

    def __call__(self, params: np.ndarray) -> float:
        if np.any(np.isnan(params)) or np.any(np.isinf(params)):
            return 10.0

        n_filters = max(8,  int(round(float(params[0]))))
        spe       = max(10, int(round(float(params[1]))))

        draws = [self._legacy_draw()] if self._eval_mode == "legacy" else self._draws

        scores = []
        for tr, vl, seed in draws:
            s = self._score_once(n_filters, spe, tr, vl, seed)
            if s is None:
                return 10.0          # a failed train is a failed candidate
            scores.append(s)

        self.last_scores = scores
        self._call_count += 1
        # DB-BOA minimises, so return the negation of the mean Obf2.
        return -float(np.mean(scores))
