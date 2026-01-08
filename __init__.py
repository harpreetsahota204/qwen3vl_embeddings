"""Qwen3-VL-Embedding FiftyOne Zoo Model.

This module provides the Qwen3-VL-Embedding model for multimodal
embedding generation in FiftyOne.

Capabilities:
    - Video/image embedding via compute_embeddings()
    - Text embedding for natural language queries
    - Text-video/image similarity search via sort_by_similarity()
    - Zero-shot classification via apply_model() with classes

Exports:
    Qwen3VLEmbeddingModel: Main model class for embeddings
    Qwen3VLEmbeddingModelConfig: Configuration class
    download_model: Download model from HuggingFace
    load_model: Load model for inference
    resolve_input: UI input configuration
"""

import logging

from huggingface_hub import snapshot_download
from fiftyone.operators import types

from .zoo import Qwen3VLEmbeddingModel, Qwen3VLEmbeddingModelConfig

logger = logging.getLogger(__name__)


def download_model(model_name, model_path):
    """Downloads the Qwen3-VL-Embedding model from HuggingFace.

    Args:
        model_name: the name of the model to download, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which to download the
            model, as declared by the ``base_filename`` field of the manifest
    """
    snapshot_download(repo_id=model_name, local_dir=model_path)


def load_model(model_name=None, model_path=None, **kwargs):
    """Load a Qwen3-VL-Embedding model for use with FiftyOne.
    
    Args:
        model_name: Model name (unused, for compatibility)
        model_path: HuggingFace model ID or path to model files
            Default: "Qwen/Qwen3-VL-Embedding-8B"
        **kwargs: Additional config parameters:
            - media_type: "video" or "image" (default: "video")
            - fps: Frame sampling rate for videos (default: 1.0)
            - max_frames: Maximum frames to sample (default: 64)
            - num_frames: Target number of frames (default: 64)
            - min_pixels: Minimum pixels for processing
            - max_pixels: Maximum pixels for processing
            - total_pixels: Total pixel budget for video frames
            - text_prompt: Prefix for zero-shot classification
            - classes: List of class labels for zero-shot classification
        
    Returns:
        Qwen3VLEmbeddingModel: Initialized model ready for inference
    
    Example - Compute embeddings:
        model = load_model(model_path="Qwen/Qwen3-VL-Embedding-8B")
        dataset.compute_embeddings(model, embeddings_field="embeddings")
    
    Example - Similarity search:
        model = load_model()
        query = model.embed_prompt("A person cooking")
        view = dataset.sort_by_similarity(query, k=10, brain_key="embeddings")
    
    Example - Zero-shot classification:
        model = load_model(
            classes=["sports", "cooking", "travel"],
            text_prompt="A video showing"
        )
        dataset.apply_model(model, label_field="predictions")
    """
    if model_path is None:
        model_path = "Qwen/Qwen3-VL-Embedding-8B"
    
    config_dict = {"model_path": model_path}
    config_dict.update(kwargs)
    
    config = Qwen3VLEmbeddingModelConfig(config_dict)
    return Qwen3VLEmbeddingModel(config)


def resolve_input(model_name, ctx):
    """Defines properties to collect the model's custom parameters.

    Args:
        model_name: the name of the model
        ctx: an ExecutionContext

    Returns:
        a fiftyone.operators.types.Property
    """
    inputs = types.Object()
    
    # Media type selection
    inputs.enum(
        "media_type",
        values=["video", "image"],
        default="video",
        label="Media Type",
        description="Type of media in the dataset (video or image)",
    )
    
    # Video processing parameters
    inputs.float(
        "fps",
        default=1.0,
        label="FPS",
        description="Frame sampling rate for videos",
    )
    
    inputs.int(
        "max_frames",
        default=64,
        label="Max Frames",
        description="Maximum number of frames to sample from video",
    )
    
    inputs.int(
        "num_frames",
        default=64,
        label="Num Frames",
        description="Target number of frames to extract",
    )
    
    inputs.int(
        "total_pixels",
        default=10 * 768 * 32 * 32,
        label="Total Pixels",
        description="Total pixel budget for video frames",
    )
    
    # Classification parameters
    inputs.str(
        "text_prompt",
        default="",
        required=False,
        label="Text Prompt",
        description="Prefix for class label prompts in zero-shot classification",
    )
    
    inputs.list(
        "classes",
        types.String(),
        default=None,
        required=False,
        label="Classes",
        description="Class labels for zero-shot classification",
    )
    
    return types.Property(inputs)
