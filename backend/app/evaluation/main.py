"""Entry point for running batch policy evaluation on the held-out test split."""

from app.data.generator import DEFAULT_DATASET_SEED, DEFAULT_RECORD_COUNT, generate_failed_payment_cases
from app.data.split import DEFAULT_SPLIT_SEED, split_cases
from app.decision.models import PolicyConfig
from app.evaluation.policy_evaluation import format_report, run_batch_evaluation
from app.model.training import DEFAULT_MODEL_SEED, compare_model_variants


def main() -> None:
    # 1. Generate data deterministically
    cases = generate_failed_payment_cases(
        record_count=DEFAULT_RECORD_COUNT, seed=DEFAULT_DATASET_SEED
    )
    
    # 2. Split with the same seed
    partitions = split_cases(cases, seed=DEFAULT_SPLIT_SEED)
    
    # 3. Re-train/select the model exactly as the training module does
    # This avoids depending on the pickled artifact, guaranteeing deterministic reproducibility
    _, _, final_selection = compare_model_variants(
        partitions.train, partitions.validation, seed=DEFAULT_MODEL_SEED
    )
    
    # 4. Evaluate batch policy on the held-out TEST partition
    config = PolicyConfig()
    report = run_batch_evaluation(final_selection.pipeline, partitions.test, config)
    
    # 5. Output report
    print(format_report(report))


if __name__ == "__main__":
    main()
