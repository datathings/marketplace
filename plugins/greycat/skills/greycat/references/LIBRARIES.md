# GreyCat Library References

Complete type definitions for all available GreyCat libraries. These files contain the full API surface, method signatures, and documentation for each library.

## Standard Library (std) - Required

Core types, collections, I/O, and runtime utilities:

- **[std/core.gcl](std/core.gcl)** — Core types: any, null, type, field, function, bool, char, int, float, nodeTime, nodeList, nodeIndex, nodeGeo, String, geo, time, duration, Date, TimeZone, Array, Map, Set, Stack, Queue, Tuple, Tensor, Table, Buffer, Gaussian, SamplingMode, SortOrder
- **[std/io.gcl](std/io.gcl)** — I/O types: Reader, Writer, BinReader, GcbReader/Writer, TextReader/Writer, XmlReader, JsonReader/Writer, Json, CsvReader/Writer, CsvFormat, CsvStatistics, Csv, File, FileWalker, Url, Http, HttpRequest, HttpResponse, HttpMethod, Email, Smtp, SmtpMode, SmtpAuth
- **[std/runtime.gcl](std/runtime.gcl)** — Runtime types: Runtime, Task, PeriodicTask, Job, StoreStat, License, UserCredential, UserRole, UserGroupPolicyType, SystemInfo, Spi
- **[std/util.gcl](std/util.gcl)** — Utility types: Assert, Histogram, GeoBox, GeoCircle, GeoPoly, BoxWhisker, Quantile, Random, TimeWindow, SlidingWindow, Crypto, HashMode, BitSet, StringBuilder

## AI Library (ai)

**[📚 AI Library Guide](ai/README.md)** — Comprehensive LLM integration via llama.cpp for text generation, chat, embeddings, and fine-tuning with LoRA adapters.

### Core API

- **[llm_model.md](ai/llm_model.md)** | [llm_model.gcl](ai/llm_model.gcl) — Model loading, text generation, chat, embeddings, tokenization (Model, LLM, ModelParams, ContextParams)
- **[llm_types.md](ai/llm_types.md)** | [llm_types.gcl](ai/llm_types.gcl) — Supporting types and parameters (GenerationParams, GenerationResult, ChatMessage, SamplerParams, ModelInfo, PerfData)

### Advanced Control

- **[llm_context.md](ai/llm_context.md)** | [llm_context.gcl](ai/llm_context.gcl) — Low-level context and KV cache management (Context, Batch, SeqId, StateData)
- **[llm_sampler.md](ai/llm_sampler.md)** | [llm_sampler.gcl](ai/llm_sampler.gcl)** — Custom sampling strategies (SamplerChain, Sampler, TokenCandidates)
- **[llm_lora.md](ai/llm_lora.md)** | [llm_lora.gcl](ai/llm_lora.gcl) — LoRA fine-tuning adapters (LoraAdapter, LoraParams)

## Algebra Library (algebra)

**[📚 ALGEBRA Library Guide](algebra/README.md)** — Comprehensive machine learning, neural networks, and numerical computing with 163+ types for building production ML/NN applications.

### Machine Learning & Neural Networks

- **[ml.md](algebra/ml.md)** | [ml.gcl](algebra/ml.gcl) — Machine learning utilities (GaussianND, PCA, Polynomial regression, TimeSeriesDecomposition)
- **[compute.md](algebra/compute.md)** | [compute.gcl](algebra/compute.gcl) — ComputeEngine, optimizers (Adam, SGD, RMSprop), layers (Dense, LSTM, Activation), initializers
- **[nn.md](algebra/nn.md)** | [nn.gcl](algebra/nn.gcl) — Neural networks (RegressionNetwork, ClassificationNetwork, AutoEncoderNetwork, training/inference)

### Pattern Recognition & Analysis

- **[patterns.md](algebra/patterns.md)** | [patterns.gcl](algebra/patterns.gcl) — Pattern detection (Euclidean, DTW, FFT, SAX) for time series matching
- **[transforms.md](algebra/transforms.md)** | [transforms.gcl](algebra/transforms.gcl) — FFT operations, frequency analysis, time series forecasting
- **[kmeans.md](algebra/kmeans.md)** | [kmeans.gcl](algebra/kmeans.gcl) — K-means clustering with meta-learning and automatic cluster selection

### Specialized Algorithms

- **[climate.md](algebra/climate.md)** | [climate.gcl](algebra/climate.gcl) — UTCI climate index calculation and thermal comfort assessment
- **[nn_layers_names.gcl](algebra/nn_layers_names.gcl)** — Neural network layer naming conventions

## Integration Libraries

External system integrations for databases, messaging, storage, and industrial protocols:

- **[kafka.md](kafka/kafka.md)** | [kafka.gcl](kafka/kafka.gcl) — Apache Kafka event streaming with type-safe producers/consumers (KafkaReader, KafkaWriter, KafkaConf with 172+ parameters)
- **[postgres.md](sql/postgres.md)** | [postgres.gcl](sql/postgres.gcl) — PostgreSQL database with transactions, SQL execution, CSV import/export via COPY command (Postgres)
- **[s3.md](s3/s3.md)** | [s3.gcl](s3/s3.gcl) — S3-compatible object storage for AWS S3 and MinIO (S3, S3Object, S3Bucket, upload/download/list)
- **[opcua.md](opcua/opcua.md)** | [opcua.gcl](opcua/opcua.gcl) — OPC UA industrial automation protocol with read/write/subscribe, historical data, security modes (OpcuaClient)
- **[useragent.md](useragent/useragent.md)** | [useragent.gcl](useragent/useragent.gcl) — User agent parsing for browser, OS, and device detection (UserAgent::parse)

## Domain-Specific Libraries

Specialized libraries for specific industries and applications:

- **[finance.md](finance/finance.md)** | [finance.gcl](finance/finance.gcl) — IBAN parsing and validation for payment processing (Iban::parse, country/bank/account extraction)
- **[powerflow.md](powerflow/powerflow.md)** | [powerflow.gcl](powerflow/powerflow.gcl) — Electrical power network analysis and load flow computation (PowerNetwork, bus/line results, grid optimization)

## Using Libraries in Your Project

Add libraries to your `project.gcl`:

```gcl
@library("std", "7.5.138-dev");      // Standard library (required)
@library("ai", "7.5.70-dev");        // AI/LLM support
@library("algebra", "7.5.70-dev");   // ML and numerical computing
@library("kafka", "7.5.70-dev");     // Kafka integration
@library("sql", "7.5.70-dev");       // PostgreSQL support (postgres library)
@library("s3", "7.5.70-dev");        // S3 storage
@library("finance", "7.5.70-dev");   // Financial utilities
@library("powerflow", "7.5.70-dev"); // Power flow analysis
@library("opcua", "7.5.70-dev");     // OPC UA integration
@library("useragent", "7.5.70-dev"); // User agent parsing
@library("explorer", "7.5.3-dev");   // Graph UI (dev only)
```

**Library Installation:**
```bash
greycat install    # downloads all declared @library dependencies
```
