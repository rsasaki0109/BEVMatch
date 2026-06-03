"""Plugin capability manifests (§7.3).

A manifest declares *what a plugin can do and where it can be trusted* — not its
implementation. BEVMatch uses manifests so a pipeline knows a plugin's input
modality, required representation, output artifact, pose assumptions, invariances,
runtime/scale profile, uncertainty support, license, dataset compatibility, and
failure modes. New plugins (e.g. a paper method) must ship a manifest (§20.6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

MANIFEST_VERSION = "1.0"


@dataclass(frozen=True)
class PluginManifest:
    name: str
    category: str  # one of registry.CATEGORIES
    output_artifact: str
    input_modality: tuple[str, ...] = ()
    required_representation: tuple[str, ...] = ()
    pose_assumption: str = "none"  # none / rough_gnss / odom_prior / map_prior
    invariance: tuple[str, ...] = ()  # rotation / viewpoint / season / illumination / sensor_height
    runtime_profile: str = "offline"  # offline / realtime / gpu_required / cpu_only
    scale_profile: str = "single_scene"  # single_scene / route_scale / city_scale
    uncertainty_support: str = "none"  # none / score_only / covariance / calibrated
    license: str = "Apache-2.0"
    dataset_compatibility: tuple[str, ...] = ("synthetic",)
    failure_modes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "manifest_version": MANIFEST_VERSION,
            "name": self.name,
            "category": self.category,
            "output_artifact": self.output_artifact,
            "input_modality": list(self.input_modality),
            "required_representation": list(self.required_representation),
            "pose_assumption": self.pose_assumption,
            "invariance": list(self.invariance),
            "runtime_profile": self.runtime_profile,
            "scale_profile": self.scale_profile,
            "uncertainty_support": self.uncertainty_support,
            "license": self.license,
            "dataset_compatibility": list(self.dataset_compatibility),
            "failure_modes": list(self.failure_modes),
        }


MANIFESTS: dict[str, PluginManifest] = {}


def register_manifest(manifest: PluginManifest) -> PluginManifest:
    MANIFESTS[manifest.name] = manifest
    return manifest


def get_manifest(name: str) -> PluginManifest:
    if name not in MANIFESTS:
        raise KeyError(f"no manifest for plugin {name!r}; known: {sorted(MANIFESTS)}")
    return MANIFESTS[name]


def list_manifests() -> list[PluginManifest]:
    return [MANIFESTS[n] for n in sorted(MANIFESTS)]


# --- built-in plugin manifests ---
for _m in [
    PluginManifest(
        name="scan-context", category="descriptor", output_artifact="descriptor",
        input_modality=("lidar", "radar"), required_representation=("scan_context",),
        invariance=("rotation",), runtime_profile="cpu_only", scale_profile="city_scale",
        uncertainty_support="score_only",
        failure_modes=("repetitive_structure", "sparse_scenes"),
    ),
    PluginManifest(
        name="bev-grid", category="descriptor", output_artifact="descriptor",
        input_modality=("lidar", "radar"), required_representation=("bev_occupancy",),
        runtime_profile="cpu_only", uncertainty_support="score_only",
        failure_modes=("viewpoint_rotation",),
    ),
    PluginManifest(
        name="camera-embedding", category="descriptor", output_artifact="descriptor",
        input_modality=("camera",), required_representation=("image_embedding",),
        invariance=("viewpoint",), runtime_profile="cpu_only", uncertainty_support="score_only",
        failure_modes=("illumination", "season"),
    ),
    PluginManifest(
        name="se2-bev-xcorr", category="alignment", output_artifact="alignment_hypothesis",
        input_modality=("lidar", "radar"), required_representation=("bev_occupancy",),
        pose_assumption="none", invariance=("rotation",), runtime_profile="cpu_only",
        uncertainty_support="covariance",
        failure_modes=("overlap_insufficient", "repetitive_structure"),
    ),
    PluginManifest(
        name="se3-icp", category="alignment", output_artifact="alignment_hypothesis",
        input_modality=("lidar",), required_representation=("point_cloud",),
        pose_assumption="odom_prior", runtime_profile="cpu_only", uncertainty_support="covariance",
        failure_modes=("planar_degeneracy", "initial_pose_too_far"),
    ),
    PluginManifest(
        name="bev-occupancy-diff", category="change_detector", output_artifact="change_hypothesis",
        input_modality=("lidar", "radar"), required_representation=("bev_occupancy",),
        pose_assumption="map_prior", runtime_profile="cpu_only", uncertainty_support="score_only",
        failure_modes=("alignment_error", "dynamic_objects", "occlusion"),
    ),
    PluginManifest(
        name="pointcloud-map-validator", category="map_validator",
        output_artifact="map_validation_issue", input_modality=("lidar",),
        required_representation=("point_cloud",), pose_assumption="map_prior",
        runtime_profile="cpu_only", uncertainty_support="score_only",
        failure_modes=("alignment_failure", "sparse_observation"),
    ),
    PluginManifest(
        name="brute-force", category="index_backend", output_artifact="candidate_list",
        runtime_profile="cpu_only", scale_profile="route_scale",
    ),
    PluginManifest(
        name="faiss", category="index_backend", output_artifact="candidate_list",
        runtime_profile="cpu_only", scale_profile="city_scale",
    ),
]:
    register_manifest(_m)
