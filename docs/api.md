# API Reference

## RAG scoring

::: pymrsf.rag.score_chunk
::: pymrsf.rag.score_chunks
::: pymrsf.rag.score_chunks_batch
::: pymrsf.rag.explain_chunk
::: pymrsf.rag.filter_chunks
::: pymrsf.rag.smart_filter
::: pymrsf.rag.WeightConfig
::: pymrsf.rag.score_chunk_async
::: pymrsf.rag.score_chunks_async
::: pymrsf.rag.filter_chunks_async

## Chunking

::: pymrsf.chunker.smart_chunk

## Knowledge probing

::: pymrsf.probe.probe
::: pymrsf.probe.probe_compare

## Embeddings

::: pymrsf.embeddings.embed
::: pymrsf.embeddings.get_embedding_dim

## Core backend

::: pymrsf.core.tokenize
::: pymrsf.core.detokenize
::: pymrsf.core.quantized_argmax
::: pymrsf.core.get_surprises
::: pymrsf.core.compute_delta
::: pymrsf.core.next_token_greedy
::: pymrsf.core.ModelSession
::: pymrsf.core.get_backend
::: pymrsf.core.get_raw_lm
::: pymrsf.core.provider_capabilities
::: pymrsf.core.set_provider

## Cache

::: pymrsf.cache.configure_cache
::: pymrsf.cache.get_cache_stats
::: pymrsf.cache.get_embedding_cache_stats
::: pymrsf.cache.reset_cache_stats
::: pymrsf.cache.clear_cache
::: pymrsf.cache.clear_embedding_cache

## Configuration

::: pymrsf.Config
::: pymrsf.configure
::: pymrsf.get_config
::: pymrsf.configure_logging

## Experimental storage

::: pymrsf.experimental.storage.mrsf_write
::: pymrsf.experimental.storage.mrsf_read
::: pymrsf.experimental.storage.mrsf_read_novel
::: pymrsf.experimental.storage.mrsf_delete
::: pymrsf.experimental.storage.rebuild_index
::: pymrsf.experimental.storage.save_index
::: pymrsf.experimental.storage.load_index
::: pymrsf.experimental.storage.reset_index_metadata
::: pymrsf.experimental.storage.close_connections

## Experimental inspection

::: pymrsf.experimental.inspect.mrsf_inspect
::: pymrsf.experimental.inspect.mrsf_rebuild_explained

## Experimental benchmarks

::: pymrsf.experimental.benchmark.mrsf_benchmark_canterbury
::: pymrsf.experimental.benchmark.mrsf_latency_benchmark
