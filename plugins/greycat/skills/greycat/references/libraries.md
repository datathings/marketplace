# GreyCat Libraries

Available libraries and their `@library()` declarations. Use the LSP (hover, completion, go-to-def) for type signatures and API details.

Latest versions: https://get.greycat.io

## project.gcl declarations

```gcl
@library("std", "7.7.158-dev");        // Required — core types, collections, I/O, runtime, util
@library("explorer", "7.7.0-dev");     // Graph UI served at /explorer

// AI/ML
@library("ai", "7.7.164-dev");        // LLM inference (llama.cpp): Model, ChatMessage, embeddings, LoRA
@library("algebra", "7.7.164-dev");   // ML, NN, patterns, transforms, clustering, climate (UTCI)

// Integrations
@library("kafka", "7.7.164-dev");     // Kafka: KafkaReader, KafkaWriter, KafkaConf
@library("sql", "7.7.164-dev");       // PostgreSQL: Postgres, transactions, COPY
@library("s3", "7.7.105-dev");        // S3/MinIO: S3, S3Object, S3Bucket (also built into std)
@library("opcua", "7.7.164-dev");     // OPC UA: OpcuaClient, read/write/subscribe
@library("ftp", "7.7.164-dev");       // FTP/FTPS: Ftp, FtpEntry
@library("ssh", "7.7.164-dev");       // SSH/SFTP: Sftp, SshAuth, SftpFile

// Domain
@library("finance", "7.7.164-dev");   // IBAN parsing: Iban::parse
@library("powerflow", "7.7.164-dev"); // Power flow analysis: PowerNetwork
@library("powergrid", "7.7.164-dev"); // Power grid (Newton-Raphson): PowerNetwork, bus/line results
@library("fcs", "7.7.164-dev");       // Flow cytometry: FcsReader, FcsMeta, FcsChannel
@library("useragent", "7.7.164-dev"); // User agent parsing: UserAgent::parse
```

## Installation

```bash
greycat install    # downloads all declared @library dependencies
greycat install --force # if lib,webroot,bin manually modified
```
