"""Versioned Orbax artifacts for deterministic Flax forecasting inference."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import tempfile
from typing import Any, Callable, Sequence

import flax
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import orbax.checkpoint as ocp

from load_forecasting.models.registry_flax import FLAX_MODEL_BUILDERS


ARTIFACT_SCHEMA_VERSION = 1
_PARAMS_DIR = "params"
_PREPROCESS_FILE = "preprocess.npz"
_GOLDEN_FILE = "golden_io.npz"
_MANIFEST_FILE = "manifest.json"


def _shape(value: Sequence[int], name: str) -> tuple[int, int]:
    shape = tuple(int(item) for item in value)
    if len(shape) != 2 or any(item < 1 for item in shape):
        raise ValueError(f"{name} must contain two positive dimensions, got {shape}")
    return shape


def _names(value: Sequence[str], expected_size: int, name: str) -> tuple[str, ...]:
    names = tuple(value)
    if len(names) != expected_size or any(
        not isinstance(item, str) or not item for item in names
    ):
        raise ValueError(f"{name} must contain {expected_size} non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{name} must not contain duplicates")
    return names


def _vector(value, expected_size: int, name: str, *, positive: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(array <= 0):
        raise ValueError(f"{name} must contain only positive values")
    return array


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(_json_value(payload), handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        np.savez_compressed(handle, **arrays)
        temporary = Path(handle.name)
    temporary.replace(path)


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "flax": flax.__version__,
        "orbax_checkpoint": importlib.metadata.version("orbax-checkpoint"),
    }


def _resolved_builder_args(builder_key: str, module, builder_args: dict[str, Any]) -> dict[str, Any]:
    """Persist defaults and derived values so reconstruction does not depend on future defaults."""
    resolved = dict(builder_args)
    signature = inspect.signature(FLAX_MODEL_BUILDERS[builder_key])
    for name, parameter in signature.parameters.items():
        if name in {"input_shape", "output_shape"}:
            continue
        if name in resolved and resolved[name] is not None:
            continue
        if hasattr(module, name):
            resolved[name] = getattr(module, name)
        elif parameter.default is not inspect.Parameter.empty:
            resolved[name] = parameter.default
    return resolved


def _validate_input_batch(inputs, input_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(inputs, dtype=np.float32)
    expected = (input_shape[0], input_shape[1])
    if array.ndim != 3 or array.shape[1:] != expected:
        raise ValueError(
            f"inputs must have shape (batch, {expected[0]}, {expected[1]}), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("inputs must contain only finite values")
    return array


def _parameter_signature(params):
    leaves, structure = jax.tree_util.tree_flatten(params)
    leaves = tuple((tuple(value.shape), np.dtype(value.dtype).str) for value in leaves)
    return structure, leaves


def save_flax_deployment_artifact(
    module,
    params,
    artifact_dir,
    *,
    builder_key: str,
    builder_args: dict[str, Any] | None,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    feature_names: Sequence[str],
    target_names: Sequence[str],
    feature_mean,
    feature_scale,
    target_mean,
    target_scale,
    sampling_rate: str | float | None = None,
    golden_input_standardized=None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save parameters, preprocessing state, reconstruction metadata, and golden I/O."""
    if builder_key not in FLAX_MODEL_BUILDERS:
        raise ValueError(f"Unknown Flax builder key: {builder_key!r}")
    if golden_input_standardized is None:
        raise ValueError("golden_input_standardized is required for deployment artifacts")

    input_shape = _shape(input_shape, "input_shape")
    output_shape = _shape(output_shape, "output_shape")
    feature_names = _names(feature_names, input_shape[1], "feature_names")
    target_names = _names(target_names, output_shape[1], "target_names")
    feature_mean = _vector(feature_mean, input_shape[1], "feature_mean")
    feature_scale = _vector(feature_scale, input_shape[1], "feature_scale", positive=True)
    target_mean = _vector(target_mean, output_shape[1], "target_mean")
    target_scale = _vector(target_scale, output_shape[1], "target_scale", positive=True)
    builder_args = _resolved_builder_args(builder_key, module, dict(builder_args or {}))

    rebuilt = FLAX_MODEL_BUILDERS[builder_key](
        input_shape=input_shape,
        output_shape=output_shape,
        **builder_args,
    )
    if rebuilt != module:
        raise ValueError(f"builder {builder_key!r} and its arguments do not recreate the model")

    sample = jnp.zeros((1, *input_shape), dtype=jnp.float32)
    params_rng, dropout_rng = jax.random.split(jax.random.PRNGKey(0))
    template_params = rebuilt.init(
        {"params": params_rng, "dropout": dropout_rng},
        sample,
        deterministic=True,
    )["params"]
    if _parameter_signature(params) != _parameter_signature(template_params):
        raise ValueError("model parameters do not match the resolved builder specification")

    artifact_dir = Path(artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    params_dir = (artifact_dir / _PARAMS_DIR).resolve()
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(params_dir, params, force=True)
        checkpointer.wait_until_finished()

    _write_npz(
        artifact_dir / _PREPROCESS_FILE,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
    )

    golden_z = _validate_input_batch(golden_input_standardized, input_shape)
    prediction_z = np.asarray(
        module.apply({"params": params}, jnp.asarray(golden_z), deterministic=True),
        dtype=np.float32,
    )
    expected_output = (golden_z.shape[0], *output_shape)
    if prediction_z.shape != expected_output:
        raise ValueError(
            f"model prediction must have shape {expected_output}, got {prediction_z.shape}"
        )
    golden_inputs = (
        golden_z * feature_scale.reshape((1, 1, -1))
        + feature_mean.reshape((1, 1, -1))
    )
    golden_predictions = (
        prediction_z * target_scale.reshape((1, 1, -1))
        + target_mean.reshape((1, 1, -1))
    )
    _write_npz(
        artifact_dir / _GOLDEN_FILE,
        inputs=np.asarray(golden_inputs, dtype=np.float32),
        predictions=np.asarray(golden_predictions, dtype=np.float32),
    )

    files = {
        "params": _PARAMS_DIR,
        "preprocessing": _PREPROCESS_FILE,
        "golden": _GOLDEN_FILE,
    }

    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "backend": "flax",
        "dtype": "float32",
        "model": {
            "builder_key": builder_key,
            "builder_args": builder_args,
            "input_shape": list(input_shape),
            "output_shape": list(output_shape),
        },
        "data": {
            "feature_names": list(feature_names),
            "target_names": list(target_names),
            "sampling_rate": sampling_rate,
            "input_contract": "engineered_unstandardized_windows",
        },
        "files": files,
        "runtime": _runtime_versions(),
        "metadata": metadata or {},
    }
    _write_json(artifact_dir / _MANIFEST_FILE, manifest)
    return manifest


@dataclass(frozen=True)
class FlaxForecastArtifact:
    """Loaded forecasting artifact with standardized and unstandardized inference."""

    artifact_dir: Path
    module: Any
    params: Any
    manifest: dict[str, Any]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    _apply: Callable = field(repr=False)

    @property
    def input_shape(self) -> tuple[int, int]:
        return tuple(self.manifest["model"]["input_shape"])

    @property
    def output_shape(self) -> tuple[int, int]:
        return tuple(self.manifest["model"]["output_shape"])

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.manifest["data"]["feature_names"])

    @property
    def target_names(self) -> tuple[str, ...]:
        return tuple(self.manifest["data"]["target_names"])

    def _check_feature_order(self, feature_names: Sequence[str]) -> None:
        if tuple(feature_names) != self.feature_names:
            raise ValueError(
                f"feature order mismatch: expected {self.feature_names}, got {tuple(feature_names)}"
            )

    def predict_standardized(
        self,
        inputs,
        *,
        feature_names: Sequence[str],
    ) -> np.ndarray:
        """Predict standardized targets from standardized input features."""
        self._check_feature_order(feature_names)
        inputs = _validate_input_batch(inputs, self.input_shape)
        prediction = np.asarray(self._apply(jnp.asarray(inputs)), dtype=np.float32)
        expected = (inputs.shape[0], *self.output_shape)
        if prediction.shape != expected:
            raise ValueError(f"model prediction must have shape {expected}, got {prediction.shape}")
        return prediction

    def predict(self, inputs, *, feature_names: Sequence[str]) -> np.ndarray:
        """Predict unstandardized targets from engineered, unstandardized input features."""
        self._check_feature_order(feature_names)
        inputs = _validate_input_batch(inputs, self.input_shape)
        standardized = (
            inputs - self.feature_mean.reshape((1, 1, -1))
        ) / self.feature_scale.reshape((1, 1, -1))
        prediction = self.predict_standardized(
            standardized,
            feature_names=self.feature_names,
        )
        return (
            prediction * self.target_scale.reshape((1, 1, -1))
            + self.target_mean.reshape((1, 1, -1))
        )

    def verify_golden(self, *, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
        """Run the stored physical-unit example and fail if its prediction drifts."""
        if self.manifest["files"].get("golden") != _GOLDEN_FILE:
            raise ValueError("artifact does not contain golden prediction data")
        with np.load(self.artifact_dir / _GOLDEN_FILE, allow_pickle=False) as golden:
            if not {"inputs", "predictions"}.issubset(golden.files):
                raise ValueError("artifact golden prediction data is incomplete")
            inputs = np.asarray(golden["inputs"], dtype=np.float32)
            expected = np.asarray(golden["predictions"], dtype=np.float32)
        inputs = _validate_input_batch(inputs, self.input_shape)
        expected_shape = (inputs.shape[0], *self.output_shape)
        if expected.shape != expected_shape or not np.all(np.isfinite(expected)):
            raise ValueError(
                f"golden predictions must be finite with shape {expected_shape}, got {expected.shape}"
            )
        actual = self.predict(inputs, feature_names=self.feature_names)
        if not np.allclose(actual, expected, rtol=rtol, atol=atol):
            max_error = float(np.max(np.abs(actual - expected)))
            raise ValueError(f"golden prediction mismatch; maximum absolute error is {max_error:.6g}")
        return True


def _read_manifest(artifact_dir: Path) -> dict[str, Any]:
    manifest_path = artifact_dir / _MANIFEST_FILE
    if not manifest_path.is_file():
        if (artifact_dir / "params.msgpack").is_file():
            raise ValueError(
                "legacy MsgPack artifact cannot be reconstructed because it lacks schema v1 metadata"
            )
        raise FileNotFoundError(f"Flax artifact manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Flax artifact manifest must contain a JSON object")
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported Flax artifact schema: {manifest.get('schema_version')!r}"
        )
    if manifest.get("backend") != "flax" or manifest.get("dtype") != "float32":
        raise ValueError("artifact must use the Flax backend and float32 dtype")
    expected_files = {
        "params": _PARAMS_DIR,
        "preprocessing": _PREPROCESS_FILE,
        "golden": _GOLDEN_FILE,
    }
    files = manifest.get("files")
    if not isinstance(files, dict) or any(
        files.get(key) != value for key, value in expected_files.items()
    ):
        raise ValueError("artifact contains invalid checkpoint file metadata")
    return manifest


def load_flax_artifact(artifact_dir, *, verify_golden: bool = True) -> FlaxForecastArtifact:
    """Rebuild an allowlisted Flax model and restore its parameters from Orbax."""
    artifact_dir = Path(artifact_dir).resolve()
    manifest = _read_manifest(artifact_dir)
    model = manifest.get("model")
    data = manifest.get("data")
    if not isinstance(model, dict) or not isinstance(data, dict):
        raise ValueError("artifact manifest is missing model or data metadata")

    builder_key = model.get("builder_key")
    if builder_key not in FLAX_MODEL_BUILDERS:
        raise ValueError(f"unknown or unavailable Flax builder key: {builder_key!r}")
    builder_args = model.get("builder_args", {})
    if not isinstance(builder_args, dict):
        raise ValueError("model.builder_args must be an object")

    input_shape = _shape(model.get("input_shape", ()), "input_shape")
    output_shape = _shape(model.get("output_shape", ()), "output_shape")
    if data.get("input_contract") != "engineered_unstandardized_windows":
        raise ValueError("artifact contains an unsupported input contract")
    feature_names = _names(data.get("feature_names", ()), input_shape[1], "feature_names")
    _names(data.get("target_names", ()), output_shape[1], "target_names")

    preprocess_path = artifact_dir / _PREPROCESS_FILE
    if not preprocess_path.is_file():
        raise FileNotFoundError(f"Flax artifact preprocessing data not found: {preprocess_path}")
    with np.load(preprocess_path, allow_pickle=False) as preprocessing:
        required = {"feature_mean", "feature_scale", "target_mean", "target_scale"}
        if not required.issubset(preprocessing.files):
            raise ValueError("artifact preprocessing data is incomplete")
        feature_mean = _vector(preprocessing["feature_mean"], input_shape[1], "feature_mean")
        feature_scale = _vector(
            preprocessing["feature_scale"], input_shape[1], "feature_scale", positive=True
        )
        target_mean = _vector(preprocessing["target_mean"], output_shape[1], "target_mean")
        target_scale = _vector(
            preprocessing["target_scale"], output_shape[1], "target_scale", positive=True
        )

    module = FLAX_MODEL_BUILDERS[builder_key](
        input_shape=input_shape,
        output_shape=output_shape,
        **builder_args,
    )
    sample = jnp.zeros((1, *input_shape), dtype=jnp.float32)
    params_rng, dropout_rng = jax.random.split(jax.random.PRNGKey(0))
    variables = module.init(
        {"params": params_rng, "dropout": dropout_rng},
        sample,
        deterministic=True,
    )
    # Pin the restore target to the current host's device so checkpoints saved
    # on a different topology (e.g. GPU-trained, CPU-restored) load cleanly.
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    abstract_params = jax.tree_util.tree_map(
        lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=sharding),
        variables["params"],
    )

    params_dir = (artifact_dir / _PARAMS_DIR).resolve()
    if not params_dir.is_dir():
        raise FileNotFoundError(f"Flax Orbax parameters not found: {params_dir}")
    with ocp.StandardCheckpointer() as checkpointer:
        params = checkpointer.restore(params_dir, target=abstract_params, strict=True)

    apply_fn = jax.jit(
        lambda inputs: module.apply({"params": params}, inputs, deterministic=True)
    )
    artifact = FlaxForecastArtifact(
        artifact_dir=artifact_dir,
        module=module,
        params=params,
        manifest=manifest,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        _apply=apply_fn,
    )
    if verify_golden:
        artifact.verify_golden()
    return artifact


@dataclass
class FlaxInferenceModel:
    """One restored model and compiled callable, reusable across archived splits."""

    module: Any
    params: Any
    metadata: dict[str, Any]
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    feature_names: tuple[str, ...]
    _predict: Callable = field(repr=False)

    def predict_standardized(self, inputs, *, batch_size=512) -> np.ndarray:
        """Batch standardized windows using the same restored model and JIT callable."""
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        inputs = _validate_input_batch(inputs, self.input_shape)
        predictions = []
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start:start + batch_size]
            prediction = np.asarray(self._predict(batch), dtype=np.float32)
            expected = (len(batch), *self.output_shape)
            if prediction.shape != expected or not np.all(np.isfinite(prediction)):
                raise ValueError(f"Expected finite predictions with shape {expected}, got {prediction.shape}")
            predictions.append(prediction)
        # Empty validation splits should retain the model's output dimensions.
        return np.concatenate(predictions) if predictions else np.empty((0, *self.output_shape), dtype=np.float32)


def legacy_flax_builder(artifact_dir, spec, input_shape, output_shape, feature_names, target_names):
    """Resolve a legacy builder from saved metadata or an explicit model spec."""
    manifest_path = Path(artifact_dir) / _MANIFEST_FILE
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = manifest.get("model", {})
        if "builder_key" in model:
            # A saved reconstruction definition takes precedence over current defaults.
            data = manifest.get("data", {})
            for label, saved, expected in (
                ("input shape", model.get("input_shape"), input_shape),
                ("output shape", model.get("output_shape"), output_shape),
                ("feature order", data.get("feature_names"), feature_names),
                ("target order", data.get("target_names"), target_names),
            ):
                if saved and tuple(saved) != tuple(expected):
                    raise ValueError(f"{artifact_dir}: manifest {label} mismatch")
            return str(model["builder_key"]), dict(model.get("builder_args", {}))

    if spec is None:
        raise ValueError(f"{artifact_dir}: legacy MsgPack loading requires a model spec")
    builder = spec["builder"]
    args = dict(spec.get("builder_args", {}))
    for key in ("mlp_units", "dilations"):
        if key in args:
            args[key] = tuple(args[key])
    if builder == "itransformer":
        args.setdefault("target_indices", tuple(feature_names.index(name) for name in target_names))
    if builder == "dual_stream_film":
        from load_forecasting.models.dual_stream_flax import resolve_feature_group_indices

        args.update(resolve_feature_group_indices(feature_names, target_names, spec.get("feature_groups")))
    return builder, args


def load_flax_inference(
    artifact_dir, *, input_shape, output_shape, feature_names, target_names,
    spec=None, feature_mean=None, feature_scale=None, target_mean=None, target_scale=None,
    verify_golden=True,
) -> FlaxInferenceModel:
    """Load either Flax format and validate its contract against archived windows."""
    artifact_dir = Path(artifact_dir)
    input_shape = _shape(input_shape, "input_shape")
    output_shape = _shape(output_shape, "output_shape")
    feature_names = _names(feature_names, input_shape[1], "feature_names")
    target_names = _names(target_names, output_shape[1], "target_names")
    params_path = artifact_dir / "params.msgpack"
    manifest_path = artifact_dir / _MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    # Never downgrade a versioned artifact to stray MsgPack files after a failed check.
    if "schema_version" in manifest or (artifact_dir / _PARAMS_DIR).is_dir() or not params_path.is_file():
        artifact = load_flax_artifact(artifact_dir, verify_golden=verify_golden)
        for label, saved, expected in (
            ("input shape", artifact.input_shape, input_shape),
            ("output shape", artifact.output_shape, output_shape),
            ("feature order", artifact.feature_names, feature_names),
            ("target order", artifact.target_names, target_names),
        ):
            if tuple(saved) != tuple(expected):
                raise ValueError(f"{artifact_dir}: artifact {label} does not match archived windows")
        for label, saved, expected in (
            ("feature mean", artifact.feature_mean, feature_mean),
            ("feature scale", artifact.feature_scale, feature_scale),
            ("target mean", artifact.target_mean, target_mean),
            ("target scale", artifact.target_scale, target_scale),
        ):
            if expected is None:
                raise ValueError(f"{artifact_dir}: archive {label} is required")
            expected = _vector(expected, len(saved), label, positive=label.endswith("scale"))
            if not np.allclose(saved, expected, rtol=1e-6, atol=1e-7):
                raise ValueError(f"{artifact_dir}: {label} does not match archive metadata")
        return FlaxInferenceModel(
            artifact.module, artifact.params, artifact.manifest.get("model", {}),
            input_shape, output_shape, feature_names,
            lambda batch: artifact.predict_standardized(batch, feature_names=feature_names),
        )

    from flax import serialization

    builder, args = legacy_flax_builder(artifact_dir, spec, input_shape, output_shape, feature_names, target_names)
    if builder not in FLAX_MODEL_BUILDERS:
        raise ValueError(f"Unknown Flax builder key {builder!r} in {artifact_dir}")
    module = FLAX_MODEL_BUILDERS[builder](input_shape=input_shape, output_shape=output_shape, **args)
    params_rng, dropout_rng = jax.random.split(jax.random.PRNGKey(0))
    template = module.init(
        {"params": params_rng, "dropout": dropout_rng},
        jnp.zeros((1, *input_shape), dtype=jnp.float32), deterministic=True,
    )["params"]
    try:
        params = serialization.from_bytes(template, params_path.read_bytes())
        # MsgPack restoration alone can accept incompatible leaf shapes.
        if _parameter_signature(params) != _parameter_signature(template):
            raise ValueError("parameter shape or dtype mismatch")
    except Exception as exc:
        raise ValueError(f"Could not restore {params_path}; ensure --model-specs matches the archive") from exc
    metadata_path = artifact_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    apply_fn = jax.jit(lambda batch: module.apply({"params": params}, batch, deterministic=True))
    return FlaxInferenceModel(module, params, metadata, input_shape, output_shape, feature_names,
                              lambda batch: apply_fn(jnp.asarray(batch)))


def load_flax_run_model(artifact_dir, meta, specs_file=None, *, spec=None, verify_golden=True):
    """Adapt run metadata and legacy spec files to the shared inference loader."""
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / _MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    legacy = (artifact_dir / "params.msgpack").is_file() and "schema_version" not in manifest
    if spec is None and legacy and not (artifact_dir / _PARAMS_DIR).is_dir() and specs_file is not None:
        from load_forecasting.config import load_model_specs

        metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        deep, hybrid = load_model_specs(specs_file)
        spec = {**deep, **hybrid}[metadata["model_key"]]
    features, targets = tuple(meta["featureNames"]), tuple(meta["targetNames"])
    return load_flax_inference(
        artifact_dir, input_shape=(int(meta["Lookback window L"]), len(features)),
        output_shape=(int(meta["Horizon forecasting window H"]), len(targets)),
        feature_names=features, target_names=targets, spec=spec,
        feature_mean=meta["MU"].loc[list(features)], feature_scale=meta["SIG"].loc[list(features)],
        target_mean=meta["MU"].loc[list(targets)], target_scale=meta["SIG"].loc[list(targets)],
        verify_golden=verify_golden,
    )
