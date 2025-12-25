# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nano-vLLM is a lightweight vLLM implementation built from scratch in ~1,200 lines of Python code. It provides fast offline inference comparable to vLLM with a readable codebase focused on educational clarity.

**Key Features:**
- Fast offline inference with optimization suite (prefix caching, tensor parallelism, torch compilation, CUDA graph)
- Clean, minimal implementation
- vLLM-compatible API with minor differences

## Architecture

### Core Components

**LLM Engine (`nanovllm/engine/llm_engine.py`)**
- Entry point for inference requests
- Manages multi-process tensor parallelism via `torch.multiprocessing`
- Coordinates between scheduler and model runner
- Main loop: `add_request()` → `step()` → `postprocess()` until finished

**Scheduler (`nanovllm/engine/scheduler.py`)**
- Manages request queues: `waiting` and `running` deques
- Schedules sequences for prefill or decode phases
- Handles preemption when KV cache blocks are exhausted
- Enforces `max_num_seqs` and `max_num_batched_tokens` constraints

**Block Manager (`nanovllm/engine/block_manager.py`)**
- Manages KV cache memory as fixed-size blocks (default: 256 tokens)
- Implements prefix caching using xxhash for block deduplication
- Tracks block reference counts for shared prefixes
- Allocates/deallocates blocks as sequences progress

**Model Runner (`nanovllm/engine/model_runner.py`)**
- Runs the actual model forward pass
- Handles tensor parallel communication via NCCL
- Manages CUDA graph capture for decode phase (batch sizes: 1, 2, 4, 8, 16, 32, ...)
- Prepares inputs differently for prefill vs decode:
  - Prefill: variable-length sequences with flash attention varlen
  - Decode: single token per sequence with KV cache lookup
- Uses shared memory for inter-process communication in tensor parallel mode

**Sequence (`nanovllm/engine/sequence.py`)**
- Represents a single inference request
- Tracks token IDs, block table, status (WAITING/RUNNING/FINISHED)
- Manages prompt vs completion tokens
- Handles block-level token organization

### Model Implementation

**Qwen3 Model (`nanovllm/models/qwen3.py`)**
- Currently only Qwen3 architecture is supported
- Standard transformer decoder with:
  - Attention with RoPE and RMS norm on Q/K
  - SwiGLU MLP (gate_up_proj merged for efficiency)
  - Tensor parallel support via custom linear layers

**Custom Layers (`nanovllm/layers/`)**
- `attention.py`: Flash attention integration with KV cache management via Triton kernel
- `linear.py`: Tensor parallel linear layers (QKVParallelLinear, ColumnParallelLinear, RowParallelLinear)
- `sampler.py`: Temperature-based sampling (greedy sampling not permitted)
- `rotary_embedding.py`: RoPE implementation
- `embed_head.py`: Vocab parallel embedding and LM head

### Utils

**`utils/context.py`:**
Thread-local context manager storing attention metadata (cu_seqlens, block_tables, slot_mapping) that is set before model forward and accessed in attention layers.

**`utils/loader.py`:**
Handles loading model weights from safetensors files with support for packed modules (merged QKV, gate_up projections) and custom weight loaders.

### Root Level

**`__init__.py`:**
Exports the main LLM and SamplingParams classes for public API access.

**`config.py`:**
Defines the Config dataclass containing all system configuration parameters including model path, memory settings, tensor parallelism size, and KV cache block configuration.

**`llm.py`:**
Provides the LLM class as an alias to LLMEngine for user-facing API.

**`sampling_params.py`:**
Defines SamplingParams dataclass for controlling generation behavior (temperature, max_tokens, ignore_eos); enforces non-greedy sampling constraint.

### Key Design Patterns

**Tensor Parallelism**
- Rank 0 process handles scheduling and sampling
- Worker processes (rank > 0) run in a loop waiting for commands via shared memory
- All ranks participate in model forward pass with NCCL collectives
- Communication via `write_shm()` / `read_shm()` with event synchronization

**Context Management (`nanovllm/utils/context.py`)**
- Thread-local context stores attention metadata (cu_seqlens, block_tables, etc.)
- Set before model forward, accessed in attention layers, reset after
- Different context for prefill vs decode phases

**CUDA Graph Optimization**
- Captured for decode phase only (prefill is dynamic)
- Multiple graphs for different batch sizes to handle variable workloads
- Graph variables reused across invocations to avoid memory allocation

**Prefix Caching**
- Block-level deduplication using xxhash
- Tracks `num_cached_tokens` per sequence to skip redundant computation
- Reference counting allows multiple sequences to share cached blocks
