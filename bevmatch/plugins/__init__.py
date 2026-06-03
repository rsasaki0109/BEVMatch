"""Plugin manifests (§7.3, §20.6 contribution: plugin manifest required)."""

from bevmatch.plugins.manifest import (
    MANIFEST_VERSION,
    MANIFESTS,
    PluginManifest,
    get_manifest,
    list_manifests,
    register_manifest,
)

__all__ = [
    "PluginManifest",
    "MANIFEST_VERSION",
    "MANIFESTS",
    "get_manifest",
    "list_manifests",
    "register_manifest",
]
