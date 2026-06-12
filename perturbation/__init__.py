"""
Perturbation module for applying text transformations.
"""

from perturbation.application import (
    PerturbationType,
    calculate_similarity,
    create_perturbation_transformer,
    apply_perturbation,
    apply_perturbations_to_benchmark,
    main as apply_perturbations
)

__all__ = [
    'PerturbationType',
    'calculate_similarity',
    'create_perturbation_transformer',
    'apply_perturbation',
    'apply_perturbations_to_benchmark',
    'apply_perturbations',
]

