"""Strict human episode routing and fixed normalization for opt-in TTT runs.

No outcome is inferred from actions, videos, losses, or missing annotations.
The source-ID manifest form requires an explicit complete provenance mapping.
"""

import json
import math
from pathlib import Path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_unique_object)


def config_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f"Expected a boolean or 'true'/'false', got {value!r}")


def _episode_id(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer, got {value!r}")
    return value


class TTTEpisodeRouting:
    """Validate once at dataset construction, then attach labels after transforms."""

    def __init__(self, dataset_name, episode_ids, manifest_path=None):
        self.episode_ids = {int(value) for value in episode_ids}
        self.eligible_episode_ids = set()
        self.provenance = {}
        if not manifest_path:
            return
        manifest = read_json(manifest_path)
        required = {"schema_version", "dataset_name", "id_space", "eligible_episode_ids"}
        optional = {"episode_provenance"}
        if not isinstance(manifest, dict) or not required <= manifest.keys() or manifest.keys() - required - optional:
            raise ValueError(f"TTT manifest requires {sorted(required)}; only optional key is episode_provenance")
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
            raise ValueError("Unsupported TTT manifest schema_version; expected integer 1")
        if manifest["dataset_name"] != dataset_name:
            raise ValueError(f"TTT manifest dataset_name {manifest['dataset_name']!r} does not match {dataset_name!r}")
        id_space = manifest["id_space"]
        if id_space not in ("episode_index", "source_episode_index"):
            raise ValueError("TTT manifest id_space must be episode_index or source_episode_index")
        eligible = manifest["eligible_episode_ids"]
        if not isinstance(eligible, list):
            raise ValueError("eligible_episode_ids must be an explicit list")
        eligible = [_episode_id(value, "eligible_episode_ids entry") for value in eligible]
        if len(eligible) != len(set(eligible)):
            raise ValueError("Duplicate eligible_episode_ids")

        provenance = manifest.get("episode_provenance", [])
        if not isinstance(provenance, list):
            raise ValueError("episode_provenance must be a list")
        source_phases = set()
        for row in provenance:
            if not isinstance(row, dict) or set(row) != {"episode_index", "source_episode_index", "phase"}:
                raise ValueError("Each provenance row requires exactly episode_index, source_episode_index, phase")
            row = {key: _episode_id(value, key) for key, value in row.items()}
            ep = row["episode_index"]
            pair = (row["source_episode_index"], row["phase"])
            if ep not in self.episode_ids or ep in self.provenance or pair in source_phases:
                raise ValueError(f"Unknown or duplicate provenance episode / source-phase: {row}")
            self.provenance[ep] = row
            source_phases.add(pair)
        if id_space == "source_episode_index":
            if set(self.provenance) != self.episode_ids:
                raise ValueError("source_episode_index manifests require complete episode_provenance for this dataset")
            known = {row["source_episode_index"] for row in self.provenance.values()}
        else:
            known = self.episode_ids
        unknown = set(eligible) - known
        if unknown:
            raise ValueError(f"Unknown eligible {id_space} IDs: {sorted(unknown)}")
        if id_space == "source_episode_index":
            self.eligible_episode_ids = {
                ep for ep, row in self.provenance.items() if row["source_episode_index"] in eligible
            }
        else:
            self.eligible_episode_ids = set(eligible)

    def sample_fields(self, episode_index, step_index):
        episode_index, step_index = int(episode_index), int(step_index)
        if episode_index not in self.episode_ids:
            raise ValueError(f"Sample has unknown TTT episode_index {episode_index}")
        fields = {
            "ttt_action_eligible": episode_index in self.eligible_episode_ids,
            "ttt_episode_index": episode_index,
            "ttt_step_index": step_index,
        }
        if episode_index in self.provenance:
            row = self.provenance[episode_index]
            fields.update(ttt_source_episode_index=row["source_episode_index"], ttt_phase=row["phase"])
        return fields


def fixed_statistics_metadata(metadata, modality_keys, source, source_tag):
    """Replace statistics using saved VLA concatenation order, preserving modality schema.

    ``source`` is a saved dataset_statistics.json (not raw LeRobot stats.json).
    It must use the same action/state dimension order as the current robot config.
    Dimensions, action mask, and all statistical arrays are validated before use.
    """
    if source_tag not in source or not isinstance(source[source_tag], dict):
        raise ValueError(f"Fixed TTT normalization tag {source_tag!r} is missing")
    result = json.loads(json.dumps(metadata))
    statistics = source[source_tag]
    names = ("min", "max", "mean", "std", "q01", "q99")
    for modality in ("action", "state"):
        keys = [key.split(".", 1)[1] for key in modality_keys.get(modality, [])]
        if not keys:
            continue
        if modality not in statistics:
            raise ValueError(f"Fixed TTT normalization is missing {modality} statistics")
        shapes = [result["modalities"][modality][key]["shape"] for key in keys]
        widths = [math.prod(shape) for shape in shapes]
        total = sum(widths)
        values = statistics[modality]
        for name in names:
            array = values.get(name)
            if not isinstance(array, list) or len(array) != total:
                raise ValueError(f"Fixed {modality}.{name} must contain {total} values in configured modality order")
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in array):
                raise ValueError(f"Fixed {modality}.{name} contains a non-finite or nonnumeric value")
        if any(value < 0 for value in values["std"]):
            raise ValueError(f"Fixed {modality}.std must be nonnegative")
        if any(lo > hi for lo, hi in zip(values["min"], values["max"])) or any(
            lo > hi for lo, hi in zip(values["q01"], values["q99"])
        ):
            raise ValueError(f"Fixed {modality} statistics contain reversed bounds")
        if modality == "action":
            # The saved mask marks gripper dimensions False (see
            # generate_action_mask_for_used_keys in gr00t_lerobot/datasets.py);
            # it encodes key-name layout, not the dtype-derived "continuous"
            # flag, so validate it against the same gripper-name rule.
            expected_mask = [
                "gripper" not in key.lower()
                for key, width in zip(keys, widths) for _ in range(width)
            ]
            mask = values.get("mask")
            if not isinstance(mask, list) or any(type(v) is not bool for v in mask) or mask != expected_mask:
                raise ValueError("Fixed action mask does not match the configured action dimensions")
        start = 0
        for key, width in zip(keys, widths):
            result["statistics"][modality][key] = {name: values[name][start:start + width] for name in names}
            start += width
    return result
