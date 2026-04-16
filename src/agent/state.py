from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd


class FeatureRecord(TypedDict):
    name: str
    code: str
    score_delta: float
    iteration: int


class AgentState(TypedDict):
    # --- profiler ---
    dataset_profile: str
    id_col: str
    target_col: str
    train_df: pd.DataFrame | None
    test_df: pd.DataFrame | None
    original_train_df: pd.DataFrame | None
    original_test_df: pd.DataFrame | None
    additional_dfs: dict[str, pd.DataFrame]

    # --- iterative loop ---
    iteration: int
    max_iterations: int
    accepted_features: list[FeatureRecord]
    rejected_features: list[FeatureRecord]
    base_score: float
    current_score: float

    # --- planner ---
    feature_plan: str

    # --- coder / executor ---
    generated_code: str
    execution_error: str
    retry_count: int
    max_retries: int
    new_columns: list[str]

    # --- general ---
    selected_features: list[str]
    status: str
    metadata: dict[str, Any]
    start_time: float


def build_initial_state(max_retries: int = 2, max_iterations: int = 8) -> AgentState:
    return AgentState(
        dataset_profile="",
        id_col="",
        target_col="",
        train_df=None,
        test_df=None,
        original_train_df=None,
        original_test_df=None,
        additional_dfs={},
        iteration=0,
        max_iterations=max_iterations,
        accepted_features=[],
        rejected_features=[],
        base_score=0.0,
        current_score=0.0,
        feature_plan="",
        generated_code="",
        execution_error="",
        retry_count=0,
        max_retries=max_retries,
        new_columns=[],
        selected_features=[],
        status="init",
        metadata={},
        start_time=0.0,
    )
