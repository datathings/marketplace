# GreyCat Library References

Complete type definitions for all available GreyCat libraries. These files contain the full API surface, method signatures, and documentation for each library.

## Standard Library (std) - Required

Core types, collections, I/O, and runtime utilities:

- **[std/core.gcl](std/core.gcl)** — Core types: any, null, type, field, function, bool, char, int, float, nodeTime, nodeList, nodeIndex, nodeGeo, String, geo, time, duration, Date, TimeZone, Array, Map, Set, Stack, Queue, Tuple, Tensor, Table, Buffer, Gaussian, SamplingMode, SortOrder
- **[std/io.gcl](std/io.gcl)** — I/O types: Reader, Writer, BinReader, GcbReader/Writer, TextReader/Writer, XmlReader, JsonReader/Writer, Json, CsvReader/Writer, CsvFormat, CsvStatistics, Csv, File, FileWalker, Url, Http, HttpRequest, HttpResponse, HttpMethod, Email, Smtp, SmtpMode, SmtpAuth
- **[std/runtime.gcl](std/runtime.gcl)** — Runtime types: Runtime, Task, PeriodicTask, Job, StoreStat, License, UserCredential, UserRole, UserGroupPolicyType, SystemInfo, Spi
- **[std/util.gcl](std/util.gcl)** — Utility types: Assert, Histogram, GeoBox, GeoCircle, GeoPoly, BoxWhisker, Quantile, Random, TimeWindow, SlidingWindow, Crypto, HashMode, BitSet, StringBuilder

## AI Library (ai)

LLM integration via llama.cpp:

- **[ai/llm_model.gcl](ai/llm_model.gcl)** — Model type: Model::load, Model::info, embed, generate, chat, tokenize, detokenize, free, SplitMode, RopeScalingType, PoolingType, AttentionType
- **[ai/llm_context.gcl](ai/llm_context.gcl)** — Context management: Context, ContextParams, ContextBatch, KvCacheView, KvCellInfo
- **[ai/llm_sampler.gcl](ai/llm_sampler.gcl)** — Sampling strategies: Sampler, SamplerParams, SamplerChain, LogitBias, GrammarType
- **[ai/llm_lora.gcl](ai/llm_lora.gcl)** — LoRA adapters: LoraAdapter, LoraParams
- **[ai/llm_types.gcl](ai/llm_types.gcl)** — Common types: ModelParams, ChatMessage, ChatCompletionChunk, VocabType, LogLevel

## Algebra Library (algebra)

Machine learning, neural networks, and numerical computing:

- **[algebra/ml.gcl](algebra/ml.gcl)** — Machine learning utilities
- **[algebra/compute.gcl](algebra/compute.gcl)** — Computational operations
- **[algebra/nn.gcl](algebra/nn.gcl)** — Neural network types and operations
- **[algebra/nn_layers_names.gcl](algebra/nn_layers_names.gcl)** — Neural network layer naming conventions
- **[algebra/patterns.gcl](algebra/patterns.gcl)** — Pattern recognition algorithms
- **[algebra/transforms.gcl](algebra/transforms.gcl)** — Data transformation utilities
- **[algebra/kmeans.gcl](algebra/kmeans.gcl)** — K-means clustering
- **[algebra/climate.gcl](algebra/climate.gcl)** — Climate data modeling

## Integration Libraries

External system integrations:

- **[kafka/kafka.gcl](kafka/kafka.gcl)** — Apache Kafka producer/consumer integration
- **[sql/postgres.gcl](sql/postgres.gcl)** — PostgreSQL database integration
- **[s3/s3.gcl](s3/s3.gcl)** — Amazon S3 object storage integration
- **[opcua/opcua.gcl](opcua/opcua.gcl)** — OPC UA industrial protocol integration
- **[useragent/useragent.gcl](useragent/useragent.gcl)** — User agent parsing utilities

## Domain-Specific Libraries

- **[finance/finance.gcl](finance/finance.gcl)** — Financial modeling and calculations
- **[powerflow/powerflow.gcl](powerflow/powerflow.gcl)** — Electrical power flow analysis

## Using Libraries in Your Project

Add libraries to your `project.gcl`:

```gcl
@library("std", "7.5.125-dev");      // Standard library (required)
@library("ai", "7.5.51-dev");        // AI/LLM support
@library("algebra", "7.5.51-dev");   // ML and numerical computing
@library("kafka", "7.5.51-dev");     // Kafka integration
@library("sql", "7.5.51-dev");       // PostgreSQL support (postgres library)
@library("s3", "7.5.51-dev");        // S3 storage
@library("finance", "7.5.51-dev");   // Financial utilities
@library("powerflow", "7.5.51-dev"); // Power flow analysis
@library("opcua", "7.5.51-dev");     // OPC UA integration
@library("useragent", "7.5.51-dev"); // User agent parsing
@library("explorer", "7.5.3-dev");   // Graph UI (dev only)
```

**Library Installation:**
```bash
greycat install    # downloads all declared @library dependencies
```
