from __future__ import annotations

import csv
import re
import signal
import time
import traceback
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd

from src.llm.gigachat_client import build_llm_client_from_env

from .state import AgentState, FeatureRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_separator(file_path: Path) -> str:
    try:
        sample = file_path.read_text(encoding="utf-8", errors="ignore")[:4096]
        if not sample.strip():
            return ","
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except Exception:
        return ","


def _read_csv_auto(file_path: Path) -> pd.DataFrame:
    sep = _detect_separator(file_path)
    try:
        return pd.read_csv(file_path, sep=sep)
    except Exception:
        return pd.read_csv(file_path)


def _format_preview(df: pd.DataFrame, rows: int = 5) -> str:
    preview = df.head(rows)
    try:
        return preview.to_markdown(index=False)
    except Exception:
        return preview.to_string(index=False)


def _infer_target_col(train_df: pd.DataFrame) -> str:
    if "target" in train_df.columns:
        return "target"
    for col in train_df.columns:
        unique_vals = set(train_df[col].dropna().unique().tolist())
        if unique_vals.issubset({0, 1}) and len(unique_vals) > 0:
            return col
    return ""


def _infer_id_col(train_df: pd.DataFrame, target_col: str) -> str:
    preferred = ["row_id", "id", "sample_id", "record_id"]
    for col in preferred:
        if col in train_df.columns:
            return col
    candidates = []
    for col in train_df.columns:
        if col == target_col:
            continue
        if "id" in col.lower():
            ratio = train_df[col].nunique(dropna=True) / max(len(train_df), 1)
            candidates.append((ratio, col))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    for col in train_df.columns:
        if col != target_col:
            return col
    return ""


def _build_table_profile(table_name: str, df: pd.DataFrame) -> str:
    lines = [f"Table: {table_name}  |  Shape: {df.shape}"]
    for col in df.columns:
        dtype = df[col].dtype
        nan_pct = round(df[col].isna().mean() * 100, 1)
        samples = df[col].dropna().head(10).tolist()
        lines.append(
            f"  {col} ({dtype}): NaN-freq [{nan_pct}%], Samples {samples}"
        )
    lines.append("")
    return "\n".join(lines)


def _compute_cv_score(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    iterations: int = 150,
    random_state: int = 42,
    thread_count: int = 4,
) -> float:
    """5-fold CV ROC-AUC using CatBoost."""
    from catboost import CatBoostClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    if len(X) == 0 or X.shape[1] == 0:
        return 0.0

    X_prep = X.copy()
    cat_cols = [c for c in X_prep.columns if str(X_prep[c].dtype) in {"object", "category", "bool"}]
    for col in X_prep.columns:
        if col in cat_cols:
            X_prep[col] = X_prep[col].fillna("__nan__").astype(str)
        else:
            X_prep[col] = pd.to_numeric(X_prep[col], errors="coerce")
            X_prep[col] = X_prep[col].replace([np.inf, -np.inf], np.nan)
            X_prep[col] = X_prep[col].fillna(X_prep[col].median())

    cat_indices = [X_prep.columns.get_loc(c) for c in cat_cols]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, val_idx in skf.split(X_prep, y):
        X_tr, X_val = X_prep.iloc[train_idx], X_prep.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = CatBoostClassifier(
            verbose=False,
            iterations=iterations,
            random_state=random_state,
            thread_count=thread_count,
            auto_class_weights="Balanced",
        )
        model.fit(X_tr, y_tr, cat_features=cat_indices)
        preds = model.predict_proba(X_val)[:, 1]
        try:
            scores.append(roc_auc_score(y_val, preds))
        except Exception:
            scores.append(0.5)
    return float(np.mean(scores))


def _time_remaining(state: AgentState) -> float:
    elapsed = time.time() - state.get("start_time", time.time())
    return max(0, 540 - elapsed)  # 540s budget (60s safety margin from 600s)


# ---------------------------------------------------------------------------
# Node 1: Profiler
# ---------------------------------------------------------------------------

def data_profiler_node(state: AgentState) -> AgentState:
    state["start_time"] = time.time()

    data_dir = Path(str(state.get("metadata", {}).get("data_dir", "data")))
    readme_path = data_dir / "readme.txt"

    if not data_dir.exists():
        state["execution_error"] = f"Data directory not found: {data_dir}"
        state["status"] = "error"
        return state

    csv_paths = sorted(data_dir.glob("*.csv"))
    if not csv_paths:
        state["execution_error"] = f"No CSV files found in {data_dir}"
        state["status"] = "error"
        return state

    loaded_tables: dict[str, pd.DataFrame] = {}
    for csv_path in csv_paths:
        loaded_tables[csv_path.name] = _read_csv_auto(csv_path)

    train_df = loaded_tables.get("train.csv")
    test_df = loaded_tables.get("test.csv")
    if train_df is None or test_df is None:
        state["execution_error"] = "Both train.csv and test.csv must exist in data/"
        state["status"] = "error"
        return state

    target_col = _infer_target_col(train_df)
    id_col = _infer_id_col(train_df, target_col)

    additional_dfs: dict[str, pd.DataFrame] = {
        name: df for name, df in loaded_tables.items()
        if name not in {"train.csv", "test.csv"}
    }

    # Auto-merge: if train has very few feature columns (just id + target),
    # merge additional tables that share the id column to bring features in
    feature_cols = [c for c in train_df.columns if c not in {id_col, target_col}]
    if len(feature_cols) <= 1 and additional_dfs:
        for table_name, extra_df in list(additional_dfs.items()):
            if table_name == "data_dictionary.csv":
                continue
            shared = set(train_df.columns) & set(extra_df.columns) - {target_col}
            if not shared:
                continue
            join_col = id_col if id_col in shared else sorted(shared)[0]
            try:
                # Align types for join
                train_df[join_col] = train_df[join_col].astype(str)
                test_df[join_col] = test_df[join_col].astype(str)
                extra_df = extra_df.copy()
                extra_df[join_col] = extra_df[join_col].astype(str)
                # Drop unnamed index columns
                drop_cols = [c for c in extra_df.columns if c.startswith("Unnamed")]
                if drop_cols:
                    extra_df = extra_df.drop(columns=drop_cols)
                new_cols = [c for c in extra_df.columns if c not in train_df.columns]
                if not new_cols:
                    continue
                merge_cols = [join_col] + new_cols
                train_df = train_df.merge(extra_df[merge_cols], on=join_col, how="left")
                test_df = test_df.merge(extra_df[merge_cols], on=join_col, how="left")
                print(f"[profiler] auto-merged {table_name} via '{join_col}': +{len(new_cols)} columns")
            except Exception as exc:
                print(f"[profiler] auto-merge {table_name} failed: {exc}")

    readme_text = ""
    if readme_path.exists():
        readme_text = readme_path.read_text(encoding="utf-8", errors="ignore")

    # Build dataset profile with sample values (CAAFE-style)
    profile_sections = ["=== DATASET DESCRIPTION ===", readme_text or "No readme.txt", ""]
    profile_sections.append(f"id_col={id_col}, target_col={target_col}")
    profile_sections.append(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    profile_sections.append("")

    for table_name, df in loaded_tables.items():
        profile_sections.append(_build_table_profile(table_name, df))

    # Compute baseline score (no engineered features)
    metadata = state.get("metadata", {})
    random_state = int(metadata.get("random_state", 42))
    cv_iterations = int(metadata.get("evaluator_iterations", 150))
    thread_count = int(metadata.get("evaluator_thread_count", 4))
    sample_rows = int(metadata.get("evaluator_sample_rows", 50000))

    base_features = [c for c in train_df.columns if c not in {id_col, target_col}]
    X_base = train_df[base_features].copy() if base_features else pd.DataFrame()
    y = train_df[target_col] if target_col else pd.Series()

    if sample_rows > 0 and sample_rows < len(X_base):
        idx = X_base.sample(n=sample_rows, random_state=random_state).index
        X_base = X_base.loc[idx]
        y = y.loc[idx]

    base_score = 0.5
    if len(X_base) > 0 and len(base_features) > 0:
        try:
            base_score = _compute_cv_score(
                X_base, y,
                iterations=cv_iterations,
                random_state=random_state,
                thread_count=thread_count,
            )
        except Exception:
            base_score = 0.5

    profile_sections.append(f"\nBaseline ROC-AUC (original features): {base_score:.4f}")

    state["dataset_profile"] = "\n".join(profile_sections)
    state["train_df"] = train_df
    state["test_df"] = test_df
    state["original_train_df"] = train_df.copy()
    state["original_test_df"] = test_df.copy()
    state["additional_dfs"] = additional_dfs
    state["id_col"] = id_col
    state["target_col"] = target_col
    state["base_score"] = base_score
    state["current_score"] = base_score
    state["execution_error"] = ""
    state["metadata"] = {
        **metadata,
        "data_dir": str(data_dir),
        "tables": list(loaded_tables.keys()),
        "base_train_columns": train_df.columns.tolist(),
        "base_test_columns": test_df.columns.tolist(),
    }
    state["status"] = "profiled"
    print(f"[profiler] dataset={data_dir}, base_score={base_score:.4f}, tables={list(loaded_tables.keys())}")
    return state


# ---------------------------------------------------------------------------
# Node 2: Featuretools Generator (automatic baseline features)
# ---------------------------------------------------------------------------

def _build_entitysets(train_for_ft, test_for_ft, train_index, additional_dfs):
    """Build featuretools EntitySets with correct relationship directions."""
    import featuretools as ft

    MAX_TABLE_ROWS = 500_000

    csv_tables = {
        k: v for k, v in additional_dfs.items()
        if k.endswith(".csv") and k not in {"data_dictionary.csv", "train.csv", "test.csv"}
    }

    # Sample large tables to avoid memory/time blowup
    for name, df in list(csv_tables.items()):
        if len(df) > MAX_TABLE_ROWS:
            print(f"[ft_generator] sampling {name}: {len(df)} -> {MAX_TABLE_ROWS} rows")
            csv_tables[name] = df.sample(n=MAX_TABLE_ROWS, random_state=42).reset_index(drop=True)

    entitysets = []
    for label, main_df in [("train_es", train_for_ft), ("test_es", test_for_ft)]:
        es = ft.EntitySet(id=label)
        main_copy = main_df.copy()
        # Cast all potential join columns to str for consistency
        for col in main_copy.columns:
            if "id" in col.lower():
                main_copy[col] = main_copy[col].astype(str)
        es = es.add_dataframe(dataframe_name="main", dataframe=main_copy, index=train_index)

        main_cols = set(main_copy.columns)

        for table_name, df in csv_tables.items():
            clean_name = table_name.replace(".csv", "").replace(".", "_")
            df_copy = df.copy()

            # Find shared columns with main
            shared = main_cols & set(df_copy.columns) - {train_index}
            if not shared:
                continue

            # Pick best join key
            join_col = None
            for col in sorted(shared):
                if "id" in col.lower():
                    join_col = col
                    break
            if not join_col:
                join_col = sorted(shared)[0]

            # Cast join col to str
            df_copy[join_col] = df_copy[join_col].astype(str)

            # Choose index: prefer join_col if unique (makes it a valid parent key)
            if df_copy[join_col].nunique() == len(df_copy):
                idx = join_col
            else:
                # Look for another unique id column
                natural_index = None
                for col in df_copy.columns:
                    if col == join_col:
                        continue
                    if "id" in col.lower() and df_copy[col].nunique() == len(df_copy):
                        natural_index = col
                        break
                if natural_index:
                    df_copy[natural_index] = df_copy[natural_index].astype(str)
                    idx = natural_index
                else:
                    idx = f"__{clean_name}_idx__"
                    df_copy[idx] = range(len(df_copy))

            try:
                es = es.add_dataframe(dataframe_name=clean_name, dataframe=df_copy, index=idx)

                # Determine relationship direction:
                # If table has unique join_col values -> it's a PARENT (lookup)
                # If table has duplicate join_col values -> it's a CHILD (aggregation)
                if df_copy[join_col].nunique() == len(df_copy):
                    # Parent table: clean_name.join_col is unique -> main looks up
                    es = es.add_relationship(clean_name, join_col, "main", join_col)
                elif main_copy[join_col].nunique() == len(main_copy):
                    # Main has unique join_col -> main is parent, table is child
                    es = es.add_relationship("main", join_col, clean_name, join_col)
                else:
                    # Neither side is unique — skip direct link,
                    # inter-table relationships may still connect them
                    pass
            except Exception:
                pass

        # Build inter-table relationships for depth=2 aggregation
        # Find tables that can be parents (unique key) and link children to them
        added_tables = {df_name for df_name in es.dataframe_dict if df_name != "main"}
        for t1 in list(added_tables):
            for t2 in list(added_tables):
                if t1 == t2:
                    continue
                shared = set(es[t1].columns) & set(es[t2].columns)
                for col in shared:
                    if "id" not in col.lower():
                        continue
                    # t1 has unique col -> t1 is parent of t2
                    try:
                        t1_df = es[t1]
                        if t1_df[col].nunique() == len(t1_df):
                            es = es.add_relationship(t1, col, t2, col)
                    except Exception:
                        pass

        entitysets.append(es)
    return entitysets


def ft_generator_node(state: AgentState) -> AgentState:
    """Generate baseline features using featuretools DFS."""
    try:
        import featuretools as ft
    except ImportError:
        print("[ft_generator] featuretools not installed, skipping")
        state["status"] = "ft_done"
        return state

    import warnings
    warnings.filterwarnings("ignore")

    if _time_remaining(state) < 300:
        print("[ft_generator] TIMEOUT: skipping featuretools")
        state["status"] = "ft_done"
        return state

    train_df = state.get("train_df")
    test_df = state.get("test_df")
    additional_dfs = state.get("additional_dfs", {})
    id_col = state.get("id_col", "")
    target_col = state.get("target_col", "")

    if train_df is None or test_df is None:
        state["status"] = "ft_done"
        return state

    try:
        start = time.time()

        # Prepare main dataframes
        train_copy = train_df.copy()
        test_copy = test_df.copy()

        if id_col and id_col in train_copy.columns:
            train_copy[id_col] = train_copy[id_col].astype(str)
            test_copy[id_col] = test_copy[id_col].astype(str)
            train_index = id_col
        else:
            train_copy["__ft_index__"] = range(len(train_copy))
            test_copy["__ft_index__"] = range(len(test_copy))
            train_index = "__ft_index__"

        train_for_ft = train_copy.drop(columns=[target_col], errors="ignore")
        test_for_ft = test_copy.drop(columns=[target_col], errors="ignore")

        es_train, es_test = _build_entitysets(
            train_for_ft, test_for_ft, train_index, additional_dfs,
        )

        print(f"[ft_generator] EntitySet: {len(es_train.dataframe_dict)} tables, "
              f"{len(es_train.relationships)} relationships")
        for rel in es_train.relationships:
            print(f"  relationship: {rel}")
        for df_name in es_train.dataframe_dict:
            print(f"  table '{df_name}': {len(es_train[df_name])} rows, "
                  f"cols={list(es_train[df_name].columns)[:8]}...")

        # Run DFS with depth=2 for cross-table aggregations
        agg_primitives = ["mean", "std", "max", "min", "count", "num_unique"]
        depth = 2 if len(es_train.relationships) > 0 else 1

        ft_train, feature_defs = ft.dfs(
            entityset=es_train,
            target_dataframe_name="main",
            agg_primitives=agg_primitives,
            trans_primitives=[],
            max_depth=depth,
            max_features=30,
            verbose=False,
        )

        ft_test = ft.calculate_feature_matrix(
            features=feature_defs,
            entityset=es_test,
            verbose=False,
        )

        # Filter: only new numeric features, no NaN-heavy or constant
        good_features = []
        for col in ft_train.columns:
            if col in train_df.columns or col in {id_col, target_col, "__ft_index__"}:
                continue
            if not pd.api.types.is_numeric_dtype(ft_train[col]):
                continue
            if ft_train[col].isna().mean() > 0.95:
                continue
            if ft_train[col].nunique(dropna=True) <= 1:
                continue
            good_features.append(col)

        # Merge into train/test
        if good_features:
            ft_train_subset = ft_train[good_features].copy()
            ft_test_subset = ft_test[good_features].copy()
            ft_train_subset.index = train_df.index
            ft_test_subset.index = test_df.index

            for col in good_features:
                train_df[col] = ft_train_subset[col].values
                test_df[col] = ft_test_subset[col].values

            state["train_df"] = train_df
            state["test_df"] = test_df

            ft_accepted = list(state.get("accepted_features", []))
            for col in good_features:
                ft_accepted.append(FeatureRecord(
                    name=col, code=f"# featuretools: {col}",
                    score_delta=0.0, iteration=-1,
                ))
            state["accepted_features"] = ft_accepted

        elapsed = time.time() - start
        print(f"[ft_generator] generated {len(good_features)} features in {elapsed:.1f}s")
        if good_features:
            print(f"[ft_generator] features: {good_features[:10]}{'...' if len(good_features) > 10 else ''}")

    except Exception as exc:
        print(f"[ft_generator] ERROR: {exc}")
        traceback.print_exc()

    state["status"] = "ft_done"
    return state


# ---------------------------------------------------------------------------
# Node 3: Planner (hypothesis generation before coding)
# ---------------------------------------------------------------------------

def _build_planner_prompt(state: AgentState) -> str:
    """Build prompt for the planner node — generates feature hypotheses."""
    profile = state.get("dataset_profile", "")
    max_profile_len = 6000
    if len(profile) > max_profile_len:
        profile = profile[:max_profile_len] + "\n... [truncated]"

    dynamic_schema = _build_dynamic_schema(state)

    accepted = state.get("accepted_features", [])
    rejected = state.get("rejected_features", [])
    iteration = state.get("iteration", 0)

    accepted_summary = ""
    if accepted:
        names = [f["name"] for f in accepted]
        accepted_summary = f"Already accepted features (DO NOT repeat): {names}"

    rejected_summary = ""
    if rejected:
        names = [f["name"] for f in rejected[-10:]]
        rejected_summary = f"Previously rejected features (avoid similar): {names}"

    score_info = (
        f"Baseline ROC-AUC: {state.get('base_score', 0.5):.4f}\n"
        f"Current ROC-AUC:  {state.get('current_score', 0.5):.4f}\n"
        f"Accepted features so far: {len(accepted)}/5\n"
        f"Iteration: {iteration + 1}/{state.get('max_iterations', 8)}"
    )

    # Build error history so planner doesn't repeat failed approaches
    previous_error = state.get("execution_error", "")
    error_block = ""
    if previous_error:
        # Truncate long tracebacks
        err_short = previous_error[:500] if len(previous_error) > 500 else previous_error
        error_block = f"\nОШИБКА ПРЕДЫДУЩЕЙ ИТЕРАЦИИ:\n{err_short}\nНЕ повторяй подход, который привёл к этой ошибке!\n"

    # Previous plan that failed — so planner doesn't repeat it
    prev_plan = state.get("feature_plan", "")
    prev_plan_block = ""
    if prev_plan and (previous_error or rejected):
        prev_plan_block = f"\nПРЕДЫДУЩИЙ ПЛАН (НЕ ПОВТОРЯЙ эти же гипотезы!):\n{prev_plan[:800]}\n"

    prompt = dedent(f"""
        Ты проектируешь гипотезы для feature engineering в агентской системе
        для автоматической генерации признаков для задачи бинарной классификации.

        {score_info}

        {accepted_summary}

        {rejected_summary}

        {error_block}

        {prev_plan_block}

        {dynamic_schema}

        Твоя задача — сформулировать НОВЫЕ гипотезы для создания признаков, которые
        увеличат ROC-AUC. Верни только краткий план, без кода.

        Формат ответа:
        ПЛАН:
        - Гипотеза 1: [семейство признаков]
            Почему полезно для таргета: ...
            Ключевые колонки и путь джойна: ...
            Проверки на leakage/устойчивость: ...
        - Гипотеза 2: ...
        - Гипотеза 3: ...
        - Приоритизация: [какие 2 идеи реализовать первыми и почему]

        Требования:
        - Предложи от 3 до 5 РАЗНЫХ гипотез (не повторяй предыдущие планы!).
        - Опирайся СТРОГО на реальные колонки из TABLE SCHEMAS выше.
        - train/test — базовые таблицы; additional_dfs — опциональный реляционный контекст.
        - Избегай target leakage.
        - Если есть несколько таблиц, используй переменную `order_details` — это pre-joined таблица всех доп. таблиц. Она уже содержит все нужные колонки.
        - Если есть связи one-to-many, сначала pre-aggregation, потом merge в train/test.
        - Для каждой гипотезы указывай ТОЧНЫЕ имена колонок из схемы.
        - Если схема — single-table, фокусируйся на ЧИСЛОВЫХ трансформациях: ratios, products, differences, polynomial interactions.
        - МОДЕЛЬ — CatBoost. Она обрабатывает категориальные признаки НАТИВНО. НИКОГДА не предлагай one-hot encoding, label encoding, target encoding, WoE, ordinal encoding — это УХУДШИТ результат!
        - Все признаки должны быть ЧИСЛОВЫМИ (int или float). Нельзя создавать строковые или категориальные фичи.
        - Хорошие идеи: col1 / (col2 + 1), col1 * col2, np.log1p(col), col1 - col2, group-by агрегаты.
        - Если прошлые итерации были неудачными, предлагай ПРИНЦИПИАЛЬНО ДРУГИЕ идеи.

        === DATASET PROFILE ===
        {profile}
    """).strip()

    return prompt


def planner_node(state: AgentState) -> AgentState:
    """Generate feature hypotheses before coding."""
    if _time_remaining(state) < 120:
        print(f"[planner] TIMEOUT: only {_time_remaining(state):.0f}s remaining, stopping")
        state["generated_code"] = ""
        state["new_columns"] = []
        state["status"] = "timeout"
        return state

    time.sleep(1)

    try:
        client = build_llm_client_from_env()
        prompt = _build_planner_prompt(state)
        system_prompt = (
            "Ты эксперт по feature engineering для бинарной классификации. "
            "Верни только план гипотез в указанном формате. Без кода."
        )
        plan = client.complete(prompt=prompt, system_prompt=system_prompt)
        state["feature_plan"] = plan.strip()
        state["status"] = "planned"
        # Print first 3 lines of plan for visibility
        plan_preview = "\n".join(plan.strip().split("\n")[:5])
        print(f"[planner] iter={state.get('iteration', 0)} plan generated ({len(plan)} chars)")
        print(f"[planner] preview:\n{plan_preview}")
    except Exception as exc:
        print(f"[planner] ERROR: {exc}")
        state["feature_plan"] = ""
        state["status"] = "planned"  # continue without plan

    return state


# ---------------------------------------------------------------------------
# Node 3: Iterative Coder (CAAFE-style)
# ---------------------------------------------------------------------------

def _build_dynamic_schema(state: AgentState) -> str:
    """Build table schemas and join hints dynamically from actual data."""
    train_df = state.get("train_df")
    test_df = state.get("test_df")
    additional_dfs = state.get("additional_dfs", {})

    lines = ["=== TABLE SCHEMAS ==="]

    # Train/test schema
    if train_df is not None:
        lines.append(f"train/test columns: {list(train_df.columns)}")

    # Additional tables
    csv_tables = {}
    for name, df in additional_dfs.items():
        if ".csv" not in name:
            continue
        if name == "data_dictionary.csv":
            continue
        lines.append(f"{name} columns ({len(df)} rows): {list(df.columns)}")
        csv_tables[name] = df

    # Auto-detect join keys
    lines.append("\n=== JOIN HINTS ===")
    train_cols = set(train_df.columns) if train_df is not None else set()
    target_col = state.get("target_col", "")
    join_cols_train = train_cols - {target_col}

    for name, df in csv_tables.items():
        shared = join_cols_train & set(df.columns)
        if shared:
            lines.append(f"train/test <-> {name}: join on {sorted(shared)}")

    # Inter-table joins
    table_names = list(csv_tables.keys())
    for i, t1 in enumerate(table_names):
        for t2 in table_names[i+1:]:
            shared = set(csv_tables[t1].columns) & set(csv_tables[t2].columns)
            if shared:
                lines.append(f"{t1} <-> {t2}: join on {sorted(shared)}")

    # Detect multi-table schema (need transitive joins)
    if len(csv_tables) >= 2:
        lines.append("\n=== PRE-JOINED TABLE: order_details ===")
        lines.append("Variable `order_details` is a pre-joined DataFrame of ALL additional tables.")
        lines.append("Use it for ANY cross-table aggregation instead of manual joins!")
        # Show what columns order_details will have
        all_cols = set()
        for df in csv_tables.values():
            all_cols.update(df.columns)
        lines.append(f"order_details contains columns from all tables: {sorted(all_cols)}")

    # Rules section
    lines.append("\n=== CRITICAL RULES ===")
    lines.append("- NEVER assume a column exists — CHECK the schema above!")
    lines.append("- Always verify join column exists in BOTH tables before merge.")
    lines.append("- The model is CatBoost which handles categorical features NATIVELY.")
    lines.append("- NEVER do one-hot encoding, label encoding, target encoding, WoE encoding, or ordinal encoding — CatBoost does this internally and your encoding will HURT performance!")
    lines.append("- Focus on creating NEW NUMERICAL features: ratios, products, differences, aggregations, polynomial interactions.")
    lines.append("- All generated features MUST be numeric (int or float). Do NOT create string or categorical features.")
    if len(csv_tables) >= 2:
        lines.append("- For cross-table features, ALWAYS use `order_details` variable (pre-joined). Do NOT manually merge individual tables.")
        lines.append("- `order_details` already contains columns from ALL tables. Groupby on order_details directly.")
        lines.append("- Example: agg = order_details.groupby(['key1','key2'])['col'].mean().reset_index()")
        lines.append("  then: train = train.merge(agg, on=['key1','key2'], how='left')")
        lines.append("- NEVER access additional_dfs for cross-table joins — use order_details!")
    else:
        lines.append("- This is a single-table dataset. All features are already in train/test.")
        lines.append("- Focus on numerical transformations: ratios between columns, products, log/sqrt transforms, differences, binning numeric cols into numeric bins (use integers, not labels).")
        lines.append("- Example good features: col1 / (col2 + 1), col1 * col2, np.log1p(col), col1 - col2")

    return "\n".join(lines)


def _build_iterative_prompt(state: AgentState) -> str:
    previous_error = state.get("execution_error", "")
    retry_mode = state.get("retry_count", 0) > 0 and bool(previous_error)
    iteration = state.get("iteration", 0)

    accepted = state.get("accepted_features", [])
    rejected = state.get("rejected_features", [])

    accepted_summary = ""
    if accepted:
        names = [f['name'] for f in accepted]
        lines = []
        for f in accepted:
            lines.append(f"  - {f['name']}: score_delta=+{f['score_delta']:.4f}")
        accepted_summary = (
            f"ALREADY EXISTING COLUMNS in train/test (DO NOT create these again): {names}\n"
            + "\n".join(lines)
            + "\n\nYou MUST generate features with DIFFERENT names."
        )

    rejected_summary = ""
    if rejected:
        names = [f['name'] for f in rejected[-10:]]
        rejected_summary = f"Previously rejected features (avoid similar): {names}"

    score_info = (
        f"Baseline ROC-AUC: {state.get('base_score', 0.5):.4f}\n"
        f"Current ROC-AUC:  {state.get('current_score', 0.5):.4f}\n"
        f"Accepted features so far: {len(accepted)}/5\n"
        f"Iteration: {iteration + 1}/{state.get('max_iterations', 8)}"
    )

    dynamic_schema = _build_dynamic_schema(state)

    # Plan from planner node
    feature_plan = state.get("feature_plan", "")
    plan_block = ""
    if feature_plan:
        plan_block = f"=== PLAN FROM PLANNER (implement these hypotheses) ===\n{feature_plan}"

    if retry_mode:
        fix_block = dedent(f"""
            YOUR PREVIOUS CODE FAILED with this error:
            {previous_error}

            Fix the code. Common issues:
            - Missing columns after merge (check column exists before using)
            - Row count change after merge (use left merge + pre-aggregation)
            - Type errors (ensure numeric before arithmetic)
            """).strip()
    else:
        fix_block = ""

    prompt = dedent(f"""
        Ты эксперт по feature engineering. Реализуй 2-4 НОВЫХ признака для задачи бинарной классификации.
        Следуй плану от planner'а ниже.

        {score_info}

        {accepted_summary}

        {rejected_summary}

        {fix_block}

        {plan_block}

        {dynamic_schema}

        === ПРАВИЛА ===
        1. Доступные переменные: train, test, additional_dfs (dict of DataFrames), pd, np, order_details (может быть None).
        2. Если переменная order_details существует (not None) — это pre-joined DataFrame всех доп. таблиц. Используй её для cross-table агрегаций.
        3. Добавляй новые колонки НАПРЯМУЮ в train и test.
        4. НЕ определяй функции. Только top-level pandas код.
        5. НЕ меняй существующие колонки и количество строк.
        6. НЕ используй целевую переменную (target) для генерации признаков — это leakage.
        7. Перед каждым merge/groupby ПРОВЕРЯЙ что нужные колонки существуют в DataFrame.
        8. Паттерн: agg = df.groupby(key)[col].agg(func).reset_index(); agg.columns = [key, 'new_name']; train = train.merge(agg, on=key, how='left')
        9. Всегда используй left merge чтобы не менять количество строк.
        10. Используй ТОЛЬКО pandas и numpy. НЕ выводи на экран и НЕ читай файлы.
        11. ВАЖНО: переименовывай агрегированные колонки чтобы избежать конфликтов имён!
        12. ВАЖНО: используй ТОЛЬКО колонки, которые реально существуют в схеме выше! Не галлюцинируй имена колонок.
        13. МОДЕЛЬ — CatBoost. Она обрабатывает категориальные признаки НАТИВНО. ЗАПРЕЩЕНО: one-hot encoding, label encoding, target encoding, WoE, get_dummies. Это УХУДШИТ результат!
        14. ВСЕ новые признаки ДОЛЖНЫ быть числовыми (int/float). Никаких строковых или Categorical фичей!
        15. Хорошие подходы: ratios (col1/(col2+1)), products (col1*col2), differences (col1-col2), log-transforms (np.log1p), group-by агрегаты на числовых колонках.

        === ФОРМАТ ВЫВОДА (один блок на признак) ===
        ```python
        # ('feature_name', 'Short description')
        # Usefulness: Why this feature helps predict the target
        <pandas code adding the column to train and test>
        ```
        """).strip()

    return prompt


def iterative_coder_node(state: AgentState) -> AgentState:
    if _time_remaining(state) < 120:
        print(f"[coder] TIMEOUT: only {_time_remaining(state):.0f}s remaining, stopping")
        state["generated_code"] = ""
        state["new_columns"] = []
        state["status"] = "timeout"
        return state

    # Rate-limit protection: wait between LLM calls
    time.sleep(2)

    try:
        client = build_llm_client_from_env()
        prompt = _build_iterative_prompt(state)
        system_prompt = (
            "Return ONLY valid Python code inside ```python``` fences. "
            "No prose outside code fences. No function definitions. "
            "Top-level pandas code operating on train, test, additional_dfs variables."
        )
        raw_response = client.complete(prompt=prompt, system_prompt=system_prompt)

        # Extract code from fences
        fenced = re.findall(r"```python\s*(.*?)```", raw_response, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            code = "\n\n".join(f.strip() for f in fenced)
        else:
            generic = re.findall(r"```\s*(.*?)```", raw_response, flags=re.DOTALL)
            code = "\n\n".join(g.strip() for g in generic) if generic else raw_response.strip()

        # Clean up: remove non-code lines that GigaChat might add
        cleaned_lines = []
        for line in code.split("\n"):
            stripped = line.strip()
            # Skip empty markdown/prose lines that aren't valid Python
            if stripped.startswith("```"):
                continue
            cleaned_lines.append(line)
        code = "\n".join(cleaned_lines).strip()

        # Validate syntax before sending to executor
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as se:
            print(f"[coder] SyntaxError in generated code: {se}")
            print(f"[coder] First 5 lines:\n" + "\n".join(f"  {i+1}: {l}" for i, l in enumerate(code.split("\n")[:5])))
            state["generated_code"] = ""
            state["execution_error"] = f"Generated code has SyntaxError: {se}"
            state["status"] = "retry"
            return state

        if not code.strip():
            state["generated_code"] = ""
            state["execution_error"] = "LLM returned empty code"
            state["status"] = "retry"
            return state

        state["generated_code"] = code
        state["execution_error"] = ""
        state["status"] = "coded"
        print(f"[coder] iter={state.get('iteration',0)} generated {len(code)} chars of code")
    except Exception as exc:
        state["generated_code"] = ""
        state["execution_error"] = f"LLM coder error: {exc}"
        state["status"] = "retry"
        print(f"[coder] iter={state.get('iteration',0)} LLM ERROR: {exc}")

    return state


# ---------------------------------------------------------------------------
# Node 3: Executor
# ---------------------------------------------------------------------------

def sandbox_executor_node(state: AgentState) -> AgentState:
    if state.get("status") == "timeout":
        return state

    state["retry_count"] = state.get("retry_count", 0) + 1

    if not state.get("generated_code"):
        if not state.get("execution_error"):
            state["execution_error"] = "No generated code to execute"
        state["status"] = "retry"
        return state

    train_df = state.get("train_df")
    test_df = state.get("test_df")
    additional_dfs = state.get("additional_dfs", {})
    if train_df is None or test_df is None:
        state["execution_error"] = "Missing train_df/test_df"
        state["status"] = "retry"
        return state

    cols_before_train = set(train_df.columns)
    cols_before_test = set(test_df.columns)
    original_train_len = len(train_df)
    original_test_len = len(test_df)

    try:
        # Add aliases without .csv extension so LLM can use either key form
        # Also align id_col types to str for consistency
        id_col = state.get("id_col", "")
        additional_dfs_aliased = {}
        for key, df in additional_dfs.items():
            df_copy = df.copy()
            if id_col and id_col in df_copy.columns:
                df_copy[id_col] = df_copy[id_col].astype(str)
            additional_dfs_aliased[key] = df_copy
            name_without_ext = key.replace(".csv", "")
            if name_without_ext not in additional_dfs_aliased:
                additional_dfs_aliased[name_without_ext] = df_copy

        # Auto-build joined tables: try to chain-join additional tables
        # that share columns, creating a wide "details" table for cross-table features
        order_details = None
        csv_tables = {k: v for k, v in additional_dfs.items() if k.endswith(".csv") and k not in {"data_dictionary.csv"}}
        if len(csv_tables) >= 2:
            try:
                # Sort by size (smallest first) to build joins incrementally
                sorted_tables = sorted(csv_tables.items(), key=lambda kv: len(kv[1]))
                # Start with the largest table, try to join others
                joined = sorted_tables[-1][1].copy()
                used = {sorted_tables[-1][0]}
                changed = True
                while changed:
                    changed = False
                    for name, df in sorted_tables:
                        if name in used:
                            continue
                        shared = set(joined.columns) & set(df.columns)
                        if shared and len(joined) > 0:
                            try:
                                joined = joined.merge(df, on=list(shared), how="left")
                                used.add(name)
                                changed = True
                            except Exception:
                                pass
                if len(used) >= 2:
                    order_details = joined
            except Exception:
                order_details = None

        if order_details is not None:
            additional_dfs_aliased["order_details"] = order_details
            print(f"[executor] order_details: {len(order_details)} rows, "
                  f"cols={list(order_details.columns)}")

        import warnings
        warnings.filterwarnings("ignore")

        train_copy = train_df.copy()
        test_copy = test_df.copy()
        if id_col and id_col in train_copy.columns:
            train_copy[id_col] = train_copy[id_col].astype(str)
            test_copy[id_col] = test_copy[id_col].astype(str)

        namespace = {
            "pd": pd,
            "np": np,
            "train": train_copy,
            "test": test_copy,
            "additional_dfs": additional_dfs_aliased,
            "order_details": order_details,
        }

        # Execute with timeout (60s max per code block)
        def _timeout_handler(signum, frame):
            raise TimeoutError("Code execution exceeded 60s timeout")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(60)
        try:
            exec(state["generated_code"], namespace)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        train_out = namespace["train"]
        test_out = namespace["test"]

        if not isinstance(train_out, pd.DataFrame) or not isinstance(test_out, pd.DataFrame):
            raise TypeError("train and test must remain pandas DataFrames")
        if len(train_out) != original_train_len:
            raise ValueError(f"train row count changed: {original_train_len} -> {len(train_out)}")
        if len(test_out) != original_test_len:
            raise ValueError(f"test row count changed: {original_test_len} -> {len(test_out)}")

        new_train_cols = [c for c in train_out.columns if c not in cols_before_train]
        new_test_cols = [c for c in test_out.columns if c not in cols_before_test]
        # Only keep columns that exist in BOTH train and test
        new_cols = [c for c in new_train_cols if c in new_test_cols]

        # Log duplicates for debugging
        dup_cols = [c for c in new_train_cols if c not in new_test_cols]
        existing_cols = [c for c in train_out.columns if c in cols_before_train and c not in cols_before_test]
        if not new_cols and new_train_cols:
            print(f"[executor] WARNING: code added columns to train but not test: {new_train_cols}")
        if not new_cols and not new_train_cols:
            print(f"[executor] WARNING: code added 0 new columns (probably duplicated existing ones)")

        state["train_df"] = train_out
        state["test_df"] = test_out
        state["new_columns"] = new_cols
        state["execution_error"] = ""
        state["status"] = "executed"
        print(f"[executor] new columns: {new_cols}")

    except Exception as exc:
        state["execution_error"] = traceback.format_exc()
        state["new_columns"] = []
        state["status"] = "retry"
        # Show the actual error and traceback for debugging
        tb = traceback.format_exc()
        # Extract just the last few lines of traceback for the relevant line
        tb_lines = tb.strip().split("\n")
        short_tb = "\n".join(tb_lines[-4:])
        code_lines = state.get("generated_code", "").split("\n")[:6]
        code_preview = "\n".join(f"  {i+1}: {line}" for i, line in enumerate(code_lines))
        print(f"[executor] ERROR: {type(exc).__name__}: {exc}")
        print(f"[executor] Traceback:\n{short_tb}")
        print(f"[executor] Code preview:\n{code_preview}")

    return state


# ---------------------------------------------------------------------------
# Node 4: Feature Judge (CAAFE-style incremental evaluation)
# ---------------------------------------------------------------------------

def feature_judge_node(state: AgentState) -> AgentState:
    # If timeout, skip evaluation and go straight to writer
    if state.get("status") == "timeout":
        return state

    train_df = state.get("train_df")
    test_df = state.get("test_df")
    if train_df is None or test_df is None:
        state["status"] = "judged"
        return state

    target_col = state.get("target_col", "")
    id_col = state.get("id_col", "")
    new_columns = state.get("new_columns", [])

    if not new_columns or not target_col:
        # No new columns to evaluate — skip
        state["iteration"] = state.get("iteration", 0) + 1
        state["retry_count"] = 0
        state["execution_error"] = ""
        state["status"] = "judged"
        return state

    metadata = state.get("metadata", {})
    random_state = int(metadata.get("random_state", 42))
    # Judge uses fast CV (100 iterations) for quick screening
    cv_iterations = 100
    thread_count = int(metadata.get("evaluator_thread_count", 4))
    sample_rows = int(metadata.get("evaluator_sample_rows", 50000))

    y = train_df[target_col]

    # Existing accepted feature columns
    accepted_col_names = [f["name"] for f in state.get("accepted_features", [])]

    # Base feature set = original columns (excluding id/target) + already accepted
    base_cols = [
        c for c in state.get("metadata", {}).get("base_train_columns", [])
        if c not in {id_col, target_col}
    ]
    current_feature_cols = base_cols + [c for c in accepted_col_names if c in train_df.columns]

    # Sample for speed
    if sample_rows > 0 and sample_rows < len(train_df):
        idx = train_df.sample(n=sample_rows, random_state=random_state).index
    else:
        idx = train_df.index

    y_sample = y.loc[idx]

    current_score = state.get("current_score", 0.5)
    accepted = list(state.get("accepted_features", []))
    rejected = list(state.get("rejected_features", []))
    code = state.get("generated_code", "")
    iteration = state.get("iteration", 0)

    for col in new_columns:
        # Skip duplicate column names (causes DataFrame instead of Series)
        if train_df.columns.tolist().count(col) > 1:
            print(f"[judge] REJECT {col}: duplicate column name")
            continue
        s = train_df[col]
        if not isinstance(s, pd.Series):
            print(f"[judge] REJECT {col}: not a Series (likely duplicate column)")
            continue
        # Filter: too many NaN or constant
        if s.isna().mean() > 0.95 or s.nunique(dropna=True) <= 1:
            rejected.append(FeatureRecord(name=col, code=code, score_delta=0.0, iteration=iteration))
            print(f"[judge] REJECT {col}: unstable (NaN>{95}% or constant)")
            continue

        # Evaluate: current features + this new one
        test_cols = current_feature_cols + [col]
        test_cols = [c for c in test_cols if c in train_df.columns]
        X_test = train_df.loc[idx, test_cols]

        try:
            new_score = _compute_cv_score(
                X_test, y_sample,
                iterations=cv_iterations,
                random_state=random_state,
                thread_count=thread_count,
            )
        except Exception as e:
            print(f"[judge] REJECT {col}: CV error: {e}")
            rejected.append(FeatureRecord(name=col, code=code, score_delta=0.0, iteration=iteration))
            continue

        delta = new_score - current_score
        print(f"[judge] {col}: score={new_score:.4f}, delta={delta:+.4f}")

        if delta > 0.001:  # threshold: meaningful improvement
            accepted.append(FeatureRecord(name=col, code=code, score_delta=delta, iteration=iteration))
            current_score = new_score
            current_feature_cols.append(col)
            print(f"[judge] ACCEPT {col} (delta=+{delta:.4f}, total accepted={len(accepted)})")
        else:
            rejected.append(FeatureRecord(name=col, code=code, score_delta=delta, iteration=iteration))
            print(f"[judge] REJECT {col} (delta={delta:+.4f})")

    state["accepted_features"] = accepted
    state["rejected_features"] = rejected
    state["current_score"] = current_score
    state["iteration"] = iteration + 1
    state["retry_count"] = 0
    state["execution_error"] = ""
    state["status"] = "judged"

    print(f"[judge] iteration {iteration+1} done. accepted={len(accepted)}, score={current_score:.4f}")
    return state


# ---------------------------------------------------------------------------
# Node 5: Writer
# ---------------------------------------------------------------------------

def writer_node(state: AgentState) -> AgentState:
    train_df = state.get("train_df")
    test_df = state.get("test_df")
    if train_df is None or test_df is None:
        state["execution_error"] = "Writer requires train_df and test_df"
        state["status"] = "error"
        return state

    output_dir = Path(str(state.get("metadata", {}).get("output_dir", "output")))
    output_dir.mkdir(parents=True, exist_ok=True)

    id_col = state.get("id_col", "")
    target_col = state.get("target_col", "")

    # Select best 5 features using greedy forward selection with CV
    accepted = state.get("accepted_features", [])
    # Validate columns exist
    candidates = [
        f for f in accepted
        if f["name"] in train_df.columns and f["name"] in test_df.columns
    ]
    candidates_sorted = sorted(candidates, key=lambda f: f["score_delta"], reverse=True)

    metadata = state.get("metadata", {})
    random_state = int(metadata.get("random_state", 42))
    cv_iterations = int(metadata.get("evaluator_iterations", 300))
    thread_count = int(metadata.get("evaluator_thread_count", 4))
    sample_rows = int(metadata.get("evaluator_sample_rows", 50000))

    base_cols = [
        c for c in metadata.get("base_train_columns", [])
        if c not in {id_col, target_col}
    ]
    y = train_df[target_col]
    if sample_rows > 0 and sample_rows < len(train_df):
        idx = train_df.sample(n=sample_rows, random_state=random_state).index
    else:
        idx = train_df.index

    # Select best 5 features: greedy forward selection if time allows, else top-5 by delta
    max_features = 5
    selected = []
    remaining = [f["name"] for f in candidates_sorted]
    best_score = state.get("base_score", 0.5)

    time_left = _time_remaining(state)
    if time_left < 60 or len(remaining) == 0:
        # Not enough time for greedy CV — take top-5 by score_delta
        selected = remaining[:max_features]
        print(f"[writer] fast selection (time_left={time_left:.0f}s): {selected}")
    else:
        print(f"[writer] {len(remaining)} candidates for greedy selection (max {max_features})")
        for _ in range(min(max_features, len(remaining))):
            if _time_remaining(state) < 30:
                # Running out of time — take best remaining by delta
                slots = max_features - len(selected)
                selected.extend(remaining[:slots])
                print(f"[writer] time limit hit, fast-filled remaining {slots} slots")
                break
            best_feat = None
            best_feat_score = best_score
            for feat in remaining:
                test_cols = base_cols + selected + [feat]
                test_cols = [c for c in test_cols if c in train_df.columns]
                X = train_df.loc[idx, test_cols]
                try:
                    score = _compute_cv_score(
                        X, y.loc[idx],
                        iterations=cv_iterations,
                        random_state=random_state,
                        thread_count=thread_count,
                    )
                except Exception:
                    continue
                if score > best_feat_score:
                    best_feat_score = score
                    best_feat = feat
            if best_feat:
                selected.append(best_feat)
                remaining.remove(best_feat)
                best_score = best_feat_score
                print(f"[writer] greedy select {len(selected)}/{max_features}: {best_feat} -> score={best_feat_score:.4f}")
            else:
                break
        print(f"[writer] greedy selection done: {selected}, final_score={best_score:.4f}")
    state["current_score"] = best_score

    if not id_col or id_col not in train_df.columns:
        id_col = train_df.columns[0]

    # check_submission requires ALL original input columns to be present
    metadata = state.get("metadata", {})
    base_train_cols = metadata.get("base_train_columns", [id_col, target_col])
    base_test_cols = metadata.get("base_test_columns", [id_col])

    train_columns = [c for c in base_train_cols if c in train_df.columns]
    train_columns.extend([c for c in selected if c not in train_columns])

    test_columns = [c for c in base_test_cols if c in test_df.columns]
    test_columns.extend([c for c in selected if c not in test_columns])

    train_out = train_df[train_columns].copy()
    test_out = test_df[test_columns].copy()

    train_out.to_csv(output_dir / "train.csv", index=False)
    test_out.to_csv(output_dir / "test.csv", index=False)

    state["selected_features"] = selected
    state["execution_error"] = ""
    state["status"] = "done"
    state["metadata"] = {
        **state.get("metadata", {}),
        "output_train_path": str(output_dir / "train.csv"),
        "output_test_path": str(output_dir / "test.csv"),
        "final_score": state.get("current_score", 0.0),
        "score_improvement": state.get("current_score", 0.0) - state.get("base_score", 0.0),
    }

    print(f"[writer] selected features: {selected}")
    print(f"[writer] base_score={state.get('base_score', 0):.4f} -> final_score={state.get('current_score', 0):.4f}")
    return state
