# GreyCat Libraries

Available libraries and their `@library()` declarations. Use the LSP (hover, completion, go-to-def) for type signatures and API details.

**Fetching latest versions**: `curl https://get.greycat.io/files/<lib>/stable/latest` — returns `<MAJOR>.<MINOR>/<MAJOR>.<MINOR>.<PATCH>-<BRANCH>` (use the part after `/`). Exception: `std` uses `core` as the URL path.

## project.gcl declarations

```gcl
@library("std", "<version>");        // Required — core types, collections, I/O, runtime, util
@library("explorer", "<version>");   // Graph UI and Administration tool served at /explorer

// AI/ML
@library("ai", "<version>");        // LLM inference (llama.cpp): Model, ChatMessage, embeddings, LoRA
@library("algebra", "<version>");   // ML, NN, patterns, transforms, clustering, climate (UTCI)

// Integrations
@library("kafka", "<version>");     // Kafka: KafkaReader, KafkaWriter, KafkaConf
@library("sql", "<version>");       // PostgreSQL: Postgres, transactions, COPY
@library("opcua", "<version>");     // OPC UA: OpcuaClient, read/write/subscribe
@library("ftp", "<version>");       // FTP/FTPS: Ftp, FtpEntry
@library("ssh", "<version>");       // SSH/SFTP: Sftp, SshAuth, SftpFile

// Domain
@library("finance", "<version>");   // IBAN parsing: Iban::parse
@library("powerflow", "<version>"); // Power flow analysis: PowerNetwork
@library("powergrid", "<version>"); // Power grid (Newton-Raphson): PowerNetwork, bus/line results
@library("fcs", "<version>");       // Flow cytometry: FcsReader, FcsMeta, FcsChannel
@library("useragent", "<version>"); // User agent parsing: UserAgent::parse
```

## Installation

```bash
greycat install    # downloads all declared @library dependencies
greycat install --force # if lib,webroot,bin manually modified
```
