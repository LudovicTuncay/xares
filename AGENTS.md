# AGENTS.md - X-ARES Codebase Guide

## Project Overview

X-ARES (eXtensive Audio Representation and Evaluation Suite) is a benchmark for evaluating
audio encoder models. It downloads datasets, encodes audio with user-provided encoders,
and evaluates embeddings using MLP fine-tuning and k-NN classification.

## Build & Run Commands

### Environment Setup
```bash
uv sync --all-extras
```

### Running the Full Benchmark
```bash
# Run all tasks with an encoder (parallel)
uv run -m xares.run --max-jobs 8 example/dasheng/dasheng_encoder.py src/tasks/*.py

# Run single task
uv run -m xares.run example/dasheng/dasheng_encoder.py src/tasks/esc50_task.py
```

### Running Individual Tasks from Python
```python
from example.dasheng.dasheng_encoder import DashengEncoder
from tasks.esc50_task import esc50_config
from xares.task import XaresTask

task = XaresTask(config=esc50_config(encoder=DashengEncoder()))
task.run()
```

### Stage-Specific Execution
```bash
# Stage 0: Download datasets only
uv run -m xares.run encoder.py src/tasks/*.py --from-stage 0 --to-stage 0

# Stage 1: Encode audio (requires encoder)
uv run -m xares.run encoder.py src/tasks/*.py --from-stage 1 --to-stage 1

# Stage 2: MLP/KNN evaluation only (uses cached embeddings)
uv run -m xares.run encoder.py src/tasks/*.py --from-stage 2
```

## Code Style Guidelines

### Python Version & Imports
- Requires Python >= 3.12
- Import order: future annotations, stdlib, third-party (torch, numpy, loguru), local xares

### Type Hints
- Use type hints for function parameters and return types
- Use `Literal` for constrained string values
- Use union syntax: `int | None` (Python 3.10+ style)
- For complex types use `Tuple`, `Dict`, `List` from typing

### Naming Conventions
- Classes: PascalCase (`XaresTask`, `TaskConfig`, `DashengEncoder`)
- Functions/methods: snake_case (`download_audio_tar`, `make_encoded_tar`)
- Variables: snake_case (`encoded_tar_dir`, `mlp_score`)
- Constants: UPPER_SNAKE_CASE (`METRICS_TYPE`, `ALL_METRICS`)
- Private methods: prefix with underscore (`_make_splits`)
- Encoder classes: MUST end with "Encoder" (`DashengEncoder`, `AudioJEPAEncoder`)
- Task config functions: MUST end with "_config" (`esc50_config`, `desed_config`)

### Dataclasses
- Use `@dataclass` for configuration and settings classes
- Use `field(default_factory=...)` for mutable defaults
- Implement `__post_init__` for validation and derived values

### Error Handling
- Use `loguru.logger` for logging (not stdlib logging)
- Log levels: `logger.info()`, `logger.warning()`, `logger.error()`, `logger.debug()`
- Use `torch.inference_mode()` for inference code

### Encoder Implementation Requirements
Encoders must:
1. Extend `torch.nn.Module`
2. Class name ends with "Encoder"
3. Have attributes: `sampling_rate`, `output_dim`, `hop_size_in_ms`
4. Accept input shape `[B, T]` (batch, time samples)
5. Return output shape `[B, T', D]` (batch, frames, embedding_dim)
6. Support variable-length audio up to 10 minutes

```python
class MyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sampling_rate = 16000
        self.output_dim = 768
        self.hop_size_in_ms = 40
        self.model = load_model()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        with torch.inference_mode():
            output = self.model(audio)
        return output  # Shape: [B, T', D]
```

### Task Configuration Pattern
```python
from xares.task import TaskConfig

def my_task_config(encoder) -> TaskConfig:
    return TaskConfig(
        encoder=encoder,
        name="my_task",
        formal_name="My Task Display Name",
        output_dim=10,  # Number of classes
        label_processor=lambda x: x["label"],
        zenodo_id="12345678",
        epochs=10,
        batch_size_train=32,
        metric="accuracy",
    )
```

### WebDataset Usage
- Audio tars use format: `{split}*.tar` or `wds-audio-fold-{n}-*.tar`
- Encoded tars use format: `wds-encoded-{split}*.tar`
- Use `create_rawaudio_webdataset()` for audio loading
- Use `create_embedding_webdataset()` for embedding loading

### Device Handling & File Operations
- Use `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Use `pathlib.Path` for path manipulation
- Use `mkdir_if_not_exists()` utility for directory creation

## Project Structure
```
xares/
├── src/
│   ├── xares/           # Core library
│   │   ├── task.py      # XaresTask, TaskConfig
│   │   ├── trainer.py   # Trainer, KNNTrainer
│   │   ├── metrics.py   # Evaluation metrics
│   │   ├── audiowebdataset.py
│   │   ├── encoders/    # Custom encoder implementations
│   │   └── models/      # MLP, ASR, Retrieval models
│   └── tasks/           # Task configuration files (*_task.py)
├── example/             # Example encoder implementations
├── scripts/             # Evaluation shell scripts
├── tools/               # Data preparation utilities
└── pyproject.toml
```

## Key Dependencies
- torch >= 2.2.1, < 2.6
- webdataset >= 0.2.100
- pytorch-ignite >= 0.5.1
- transformers >= 4.47.1
- loguru (logging)
- scikit-learn >= 1.6.0

## Environment Variables
- `AUDIO_JEPA_CHECKPOINT`: Path to AudioJEPA checkpoint (for AudioJEPA encoder)
- `AUDIO_JEPA_CONFIG`: Path to AudioJEPA config (for AudioJEPA encoder)
- `CUDA_VISIBLE_DEVICES`: GPU device selection
