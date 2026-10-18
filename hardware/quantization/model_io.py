"""Load persistent trained weights without importing the training pipeline."""
import torch


def load_persistent_model_state_dict(model, state_dict):
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [key for key in incompatible.missing_keys if not key.endswith('.v_mem')]
    if unexpected or missing:
        raise RuntimeError(f'checkpoint mismatch: missing_non_membrane={missing}, unexpected={unexpected}')
