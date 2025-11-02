# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nano-vLLM is a lightweight, educational implementation of vLLM (fast offline LLM inference) built from scratch in ~1,350 lines of Python. It achieves comparable performance to vLLM while maintaining a readable codebase focused on clarity over production features.

**Key optimizations implemented:**
- Prefix caching (automatic KV cache reuse via hash-based block management)
- Tensor parallelism (distributed inference across multiple GPUs)
- CUDA graphs (reduced kernel launch overhead)
- Flash Attention (efficient attention computation)
- Paged attention (memory-efficient KV cache management)

## Architecture Overview

### Core Request Flow

1. **LLM/LLMEngine** (`llm.py`, `engine/llm_engine.py`): Top-level API that mirrors vLLM's interface
   - Initializes tokenizer, scheduler, and model runner(s)
   - Manages multiprocessing for tensor parallelism
   - Provides `generate()` method for batch inference

2. **Scheduler** (`engine/scheduler.py`): Manages request lifecycle and batching
   - Maintains `waiting` and `running` queues of Sequences
   - Decides which sequences to process in each step (prefill vs decode)
   - Handles preemption when KV cache is full
   - Works with BlockManager to allocate/deallocate cache blocks

3. **BlockManager** (`engine/block_manager.py`): Handles KV cache memory management
   - Implements automatic prefix caching using xxhash for block-level deduplication
   - Manages reference counting for shared cache blocks
   - Uses paged memory with configurable block size (default: 256 tokens)

4. **ModelRunner** (`engine/model_runner.py`): Executes model inference
   - Handles both prefill (initial prompt processing) and decode (autoregressive generation)
   - Manages CUDA graph capture for decode batches (sizes: [1, 2, 4, 8] + [16, 32, 48, ...] in steps of 16)
   - Coordinates tensor parallelism via PyTorch distributed (NCCL)
   - Prepares inputs: block_tables, slot_mapping, cu_seqlens for flash attention

5. **Sequence** (`engine/sequence.py`): Represents a single generation request
   - Tracks token IDs, block allocation, and sampling parameters
   - States: WAITING → RUNNING → FINISHED

### Key Execution Paths

**Prefill**: Process multiple sequences with variable input lengths
- Flash attention with `cu_seqlens_q/k` for ragged batching
- Automatic prefix cache lookup/allocation
- Position IDs start from `num_cached_tokens` for cache hits

**Decode**: Generate one token per sequence (batch size ≤ max_num_seqs)
- Uses CUDA graphs for batch sizes ≤ 512 (unless `enforce_eager=True`)
- Paged attention with block_tables and context_lens
- Single position per sequence

### Model Architecture

Currently supports **Qwen3** (`models/qwen3.py`):
- Standard decoder-only transformer with GQA (Grouped Query Attention)
- RoPE embeddings with configurable theta and scaling
- SwiGLU activation (SiluAndMul)
- RMSNorm pre/post normalization with fused residual connections

**Tensor Parallelism**: Weights are sharded across GPUs via:
- `QKVParallelLinear`, `MergedColumnParallelLinear` (column-parallel)
- `RowParallelLinear` (row-parallel with all-reduce)
- `VocabParallelEmbedding`, `ParallelLMHead`

### Custom Layers (`layers/`)

- **attention.py**: Flash attention 2 wrapper with prefill/decode modes, includes custom Triton kernel for efficient KV cache storage
- **linear.py**: Tensor-parallel linear layers using PyTorch distributed
- **rotary_embedding.py**: RoPE implementation
- **sampler.py**: Temperature-based sampling (no greedy sampling, temperature > 1e-10)
- **activation.py**, **layernorm.py**, **embed_head.py**: Standard components with TP support

### Utilities

- **utils/context.py**: Thread-local context for passing attention metadata (is_prefill, cu_seqlens, block_tables, etc.)
- **utils/loader.py**: Loads HuggingFace model weights with support for packed weight mappings (e.g., qkv_proj, gate_up_proj)

### Configuration (`config.py`)

Key parameters:
- `max_num_batched_tokens` (default: 16384): Max tokens per prefill batch
- `max_num_seqs` (default: 512): Max concurrent sequences in decode
- `max_model_len` (default: 4096): Max sequence length
- `kvcache_block_size` (default: 256): Must be multiple of 256
- `tensor_parallel_size` (default: 1): Number of GPUs (1-8)
- `enforce_eager` (default: False): Disable CUDA graphs

## Implementation Notes

### Multiprocessing for Tensor Parallelism
- Main process (rank 0) runs LLMEngine and communicates with workers via SharedMemory
- Worker processes (rank 1+) run `ModelRunner.loop()` waiting for commands
- Synchronization via `multiprocessing.Event` and pickle-serialized commands

### CUDA Graph Capture
- Graphs are captured for specific batch sizes during initialization: [1, 2, 4, 8] + multiples of 16 up to max_bs
- Uses memory pool (`graph.pool()`) to share memory across graphs
- Input/output tensors are reused via `graph_vars` dictionary
- Graph replay avoids Python overhead for small decode batches (≤ 512)

### Prefix Caching Implementation
- Block-level hashing: hash includes previous block's hash (chain of hashes)
- Only full blocks (256 tokens) are cached; partial blocks use `hash = -1`
- `num_cached_tokens` tracks how many prompt tokens were cache hits
- Scheduler skips cached tokens during prefill scheduling

### Position Indexing
- Positions are 0-based (fixed in recent commit f5b4840)
- During prefill with cache hits: positions start from `num_cached_tokens`
- During decode: position is `len(seq) - 1`

## Common Patterns

### Adding Support for New Models
1. Create `models/{model_name}.py` following Qwen3 structure
2. Implement attention, MLP, decoder layer, and CausalLM classes
3. Define `packed_modules_mapping` for weight loading
4. Ensure all layers support tensor parallelism (use parallel linear layers)
5. Model should accept (input_ids, positions) and return hidden states

### Modifying Sampling
- Edit `layers/sampler.py` to add new sampling methods
- Update `SamplingParams` dataclass to include new parameters
- Sampler is called only on rank 0, results are serialized back to scheduler

### Debugging Attention
- Check `utils/context.py` for current metadata being passed
- Verify block_tables and slot_mapping in `model_runner.py:prepare_prefill/decode`
- Flash attention requires contiguous tensors and correct cu_seqlens

## Known Constraints

- Only supports temperature sampling (no greedy decoding, beam search, or advanced methods)
- Single model architecture (Qwen3) currently implemented
- Tensor parallelism limited to 1-8 GPUs
- CUDA graphs disabled for batch sizes > 512
- No pipeline parallelism or sequence parallelism
