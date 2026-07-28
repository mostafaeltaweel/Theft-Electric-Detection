"""
Model Loader & Keras Compatibility Engine for ETD-XAI.
Loads the trained CNN-LSTM Keras model safely with Keras 3 deserialization compatibility.
"""
import io
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from src.config import MODEL_PATH

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_TF = None

def get_tf():
    """Lazy import of TensorFlow with Keras 3 compatibility shim."""
    global _TF
    if _TF is None:
        import tensorflow as _t
        _t.get_logger().setLevel("ERROR")
        try:
            import keras
            _orig = keras.layers.Dense.from_config.__func__

            @classmethod
            def _compat(cls, config):
                config = dict(config)
                config.pop("quantization_config", None)
                dt = config.get("dtype")
                if isinstance(dt, dict):
                    config["dtype"] = dt.get("config", {}).get("name", "float32")
                return _orig(cls, config)
                
            keras.layers.Dense.from_config = _compat
        except Exception:
            pass
        _TF = _t
    return _TF

class ModelEngine:
    """Process-wide singleton maintaining loaded Keras model state."""
    model = None
    name = "CNN-LSTM"
    path = ""
    load_time = ""
    input_shape = ()
    output_shape = ()
    total_params = 0
    seq_len = 120
    is_dual = True
    stat_size = 59

ENGINE = ModelEngine()

def load_active_model(path: Optional[Path] = None) -> bool:
    """Loads Keras model into memory."""
    target_path = path or MODEL_PATH
    if not target_path.exists():
        return False
        
    try:
        _t = get_tf()
        model = _t.keras.models.load_model(str(target_path))
        
        ENGINE.model = model
        ENGINE.path = str(target_path)
        ENGINE.name = target_path.name
        ENGINE.load_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        ENGINE.input_shape = tuple(model.input_shape) if isinstance(model.input_shape, (list, tuple)) else (model.input_shape,)
        ENGINE.output_shape = tuple(model.output_shape)
        ENGINE.total_params = int(model.count_params())
        
        # Dual input check (Sequence input + Stat features input)
        ENGINE.is_dual = (len(model.inputs) == 2)
        return True
    except Exception as e:
        print(f"[Model Engine Error] Failed loading Keras model from {target_path}: {e}")
        ENGINE.model = None
        return False

def is_model_loaded() -> bool:
    """Checks whether the active CNN-LSTM model is loaded."""
    if ENGINE.model is None:
        return load_active_model()
    return True

def get_model_metadata() -> Dict[str, Any]:
    """Returns model metadata for dashboard & diagnostics."""
    tf_ver = get_tf().__version__ if is_model_loaded() else "Not loaded"
    import keras as _k
    keras_ver = getattr(_k, "__version__", "unknown")
    
    return {
        "loaded": ENGINE.model is not None,
        "name": ENGINE.name,
        "path": ENGINE.path,
        "architecture": "CNN-LSTM Dual-Input Engine" if ENGINE.is_dual else "CNN-LSTM Engine",
        "input_shape": str(ENGINE.input_shape) if ENGINE.model else "(None, 120, 1)",
        "output_shape": str(ENGINE.output_shape) if ENGINE.model else "(None, 1)",
        "total_params": ENGINE.total_params,
        "total_params_fmt": f"{ENGINE.total_params:,}" if ENGINE.model else "0",
        "tf_version": tf_ver,
        "keras_version": keras_ver,
        "load_time": ENGINE.load_time
    }
