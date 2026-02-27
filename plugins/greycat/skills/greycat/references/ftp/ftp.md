# FTP File Transfer

FTP and FTPS file transfer integration for GreyCat.

## Overview

The FTP library provides seamless integration with FTP and FTPS servers, enabling GreyCat applications to upload, download, list, and delete files on remote servers. It supports both plain FTP and secure FTPS (FTP over TLS) connections.

Key features include:
- **FTP and FTPS support** via `ftp://` and `ftps://` URL schemes
- **File operations** including upload, download, list, and delete
- **Credential authentication** using username/password pairs
- **Anonymous access** when credentials are omitted
- **Configurable timeout** with a sensible default of 30 seconds

This library is ideal for integrating with legacy file transfer systems, exchanging data with external partners, automating file synchronization, or any scenario requiring standard FTP protocol access.

## Installation

Add the FTP library to your GreyCat project:

```gcl
@library("ftp", "7.7.151-dev")
```

## Quick Start

### Connect and Download a File

```gcl
var ftp = Ftp {
  url: "ftp://ftp.example.com",
  credentials: "user:password"
};

// Download a remote file
ftp.get("/remote/path/report.csv", "/local/path/report.csv");

// Upload a local file
ftp.put("/local/path/data.csv", "/remote/path/data.csv");
```

### List Files in a Directory

```gcl
var entries = ftp.list("/remote/data/");

for (entry in entries) {
  if (entry.is_dir) {
    print("[DIR]  ${entry.name}");
  } else {
    print("[FILE] ${entry.name} - ${entry.size} bytes - ${entry.modify}");
  }
}
```

## Types

### Ftp

Main connection type representing an FTP client.

**Fields:**
- `url: String` - FTP server address using `ftp://` or `ftps://` scheme (e.g., `"ftp://ftp.example.com"`, `"ftps://secure.example.com"`)
- `credentials: String?` - Authentication in `"username:password"` format (null = anonymous access)
- `timeout: duration?` - Connection and operation timeout (defaults to 30 seconds)

**Methods:**
- `get(remote, local)` - Download a remote file to a local path
- `list(dirpath): Array<FtpEntry>` - List entries in a remote directory
- `delete(filepath)` - Delete a remote file
- `put(local, remote)` - Upload a local file to a remote path

**Example:**

```gcl
// Authenticated FTP connection
var ftp = Ftp {
  url: "ftp://ftp.example.com",
  credentials: "myuser:mypassword"
};

// Secure FTPS connection
var ftps = Ftp {
  url: "ftps://secure.example.com",
  credentials: "myuser:mypassword",
  timeout: 60s
};

// Anonymous FTP access
var anonFtp = Ftp {
  url: "ftp://public.example.com"
};
```

### FtpEntry

Represents an entry (file or directory) returned by a directory listing.

**Fields:**
- `name: String` - Name of the file or directory
- `is_dir: bool` - Whether the entry is a directory (`true`) or a file (`false`)
- `size: int` - Size of the entry in bytes (0 for directories on most servers)
- `modify: String` - Last modification timestamp in RFC 3659 format (`YYYYMMDDhhmmss`)

**Example:**

```gcl
var entries = ftp.list("/uploads/");

for (entry in entries) {
  print("Name: ${entry.name}");
  print("  Type: ${entry.is_dir ? "directory" : "file"}");
  print("  Size: ${entry.size} bytes");
  print("  Modified: ${entry.modify}");
}
```

## Methods

### get()

Downloads a remote file from the FTP server to a local path.

**Signature:** `fn get(remote: String, local: String)`

**Parameters:**
- `remote: String` - Absolute path to the file on the FTP server
- `local: String` - Local destination file path

**Behavior:**
- Downloads the entire file
- Overwrites existing local file
- Throws error if remote file does not exist

**Example:**

```gcl
// Download a file
ftp.get("/data/export.csv", "/tmp/export.csv");

// Download from nested directory
ftp.get("/reports/2024/01/monthly.pdf", "/local/reports/monthly.pdf");

// Download with error handling
try {
  ftp.get("/remote/important.txt", "/tmp/important.txt");
  print("Download successful");
} catch (e) {
  print("Download failed: ${e}");
}
```

### list()

Lists entries (files and directories) in a remote directory.

**Signature:** `fn list(dirpath: String): Array<FtpEntry>`

**Parameters:**
- `dirpath: String` - Absolute path to the directory on the FTP server

**Returns:** Array of `FtpEntry` instances

**Behavior:**
- Returns all entries in the specified directory
- Each entry includes name, type, size, and modification time
- Throws error if directory does not exist or is inaccessible

**Example:**

```gcl
// List root directory
var root = ftp.list("/");

// List a specific directory
var files = ftp.list("/data/incoming/");

for (entry in files) {
  print("${entry.name} (${entry.size} bytes)");
}

// Filter for files only (skip directories)
var entries = ftp.list("/uploads/");
for (entry in entries) {
  if (!entry.is_dir) {
    print("File: ${entry.name}");
  }
}
```

### delete()

Removes a file from the FTP server.

**Signature:** `fn delete(filepath: String)`

**Parameters:**
- `filepath: String` - Absolute path to the file to delete on the FTP server

**Behavior:**
- Permanently deletes the remote file
- Throws error if file does not exist or permissions are insufficient

**Example:**

```gcl
// Delete a single file
ftp.delete("/tmp/old-report.csv");

// Delete files after processing
var entries = ftp.list("/incoming/");
for (entry in entries) {
  if (!entry.is_dir) {
    ftp.get("/incoming/${entry.name}", "/local/processed/${entry.name}");
    ftp.delete("/incoming/${entry.name}");
    print("Processed and removed: ${entry.name}");
  }
}
```

### put()

Uploads a local file to the FTP server.

**Signature:** `fn put(local: String, remote: String)`

**Parameters:**
- `local: String` - Local source file path
- `remote: String` - Absolute destination path on the FTP server

**Behavior:**
- Uploads the entire file
- Overwrites existing remote file with the same path
- Throws error if local file does not exist

**Example:**

```gcl
// Upload a file
ftp.put("/local/data/report.csv", "/remote/reports/report.csv");

// Upload multiple files
var files = ["data1.csv", "data2.csv", "data3.csv"];
for (file in files) {
  ftp.put("/local/exports/${file}", "/remote/imports/${file}");
  print("Uploaded: ${file}");
}

// Upload with error handling
try {
  ftp.put("/local/large-file.bin", "/remote/uploads/large-file.bin");
  print("Upload successful");
} catch (e) {
  print("Upload failed: ${e}");
}
```

## Common Use Cases

### File Download Pipeline

```gcl
var ftp = Ftp {
  url: "ftp://ftp.example.com",
  credentials: "user:password"
};

// List available files in the incoming directory
var entries = ftp.list("/incoming/");

for (entry in entries) {
  if (!entry.is_dir) {
    // Download each file
    var localPath = "/tmp/downloads/${entry.name}";
    ftp.get("/incoming/${entry.name}", localPath);
    print("Downloaded: ${entry.name} (${entry.size} bytes)");
  }
}
```

### Directory Listing and Filtering

```gcl
var ftp = Ftp {
  url: "ftp://ftp.example.com",
  credentials: "user:password"
};

var entries = ftp.list("/data/");

// Separate files from directories
for (entry in entries) {
  if (entry.is_dir) {
    print("[DIR]  ${entry.name}");
  } else {
    print("[FILE] ${entry.name} - ${entry.size} bytes - modified ${entry.modify}");
  }
}
```

### Upload and Distribution

```gcl
var ftp = Ftp {
  url: "ftp://ftp.example.com",
  credentials: "user:password"
};

// Generate and upload reports
var reports = ["summary.pdf", "details.csv", "charts.png"];
for (report in reports) {
  ftp.put("/local/reports/${report}", "/remote/published/${report}");
  print("Published: ${report}");
}
```

### Secure FTPS Transfer

```gcl
// Use ftps:// for encrypted file transfer
var ftps = Ftp {
  url: "ftps://secure.partner.com",
  credentials: "transfer-user:secure-password",
  timeout: 120s
};

// Upload sensitive data over encrypted connection
ftps.put("/local/sensitive-data.csv", "/partner/inbox/data.csv");

// Download partner response
ftps.get("/partner/outbox/response.csv", "/local/responses/response.csv");

print("Secure transfer complete");
```

### Process and Clean Up

```gcl
var ftp = Ftp {
  url: "ftp://ftp.example.com",
  credentials: "user:password"
};

// Download, process, and remove files from remote server
var entries = ftp.list("/queue/");

for (entry in entries) {
  if (!entry.is_dir) {
    var remotePath = "/queue/${entry.name}";
    var localPath = "/tmp/processing/${entry.name}";

    // Download
    ftp.get(remotePath, localPath);

    // Process locally (application-specific logic)
    processFile(localPath);

    // Archive on remote server before deleting
    ftp.put(localPath, "/archive/${entry.name}");

    // Remove from queue
    ftp.delete(remotePath);

    print("Processed: ${entry.name}");
  }
}
```

## Best Practices

### Credential Management

- **Never hardcode credentials**: Use environment variables or secure configuration
- **Use FTPS for sensitive data**: Always prefer `ftps://` over `ftp://` for production
- **Rotate passwords regularly**: Update FTP credentials periodically for security

```gcl
// Good: Load from environment
var ftp = Ftp {
  url: env("FTP_URL"),
  credentials: "${env("FTP_USER")}:${env("FTP_PASSWORD")}"
};

// Bad: Hardcoded credentials
var ftp = Ftp {
  url: "ftp://ftp.example.com",
  credentials: "admin:secret123" // NEVER DO THIS!
};
```

### Timeout Configuration

- **Set appropriate timeouts**: Increase for large files or slow connections
- **Default is 30 seconds**: Sufficient for most small file operations
- **Long transfers need longer timeouts**: Set explicitly for large uploads/downloads

```gcl
// Default timeout (30s) for small files
var ftp = Ftp {
  url: "ftp://ftp.example.com",
  credentials: "user:password"
};

// Extended timeout for large file transfers
var ftpLargeFiles = Ftp {
  url: "ftp://ftp.example.com",
  credentials: "user:password",
  timeout: 300s
};
```

### Error Handling

- **Wrap operations in try-catch**: Network issues, permissions, and missing files are common
- **Verify downloads**: Check that the local file exists after download
- **Handle missing files gracefully**: `get()` fails if the remote file does not exist

```gcl
try {
  ftp.get("/remote/file.txt", "/local/file.txt");
  print("Download successful");
} catch (e) {
  print("Failed to download: ${e}");
  // Handle missing file or permission error
}
```

### Gotchas

- **FTP vs FTPS**: Use `ftp://` for plain FTP and `ftps://` for FTP over TLS; they are not interchangeable
- **Anonymous access**: Omit `credentials` (or set to `null`) for anonymous FTP; do not pass an empty string
- **Path format**: Remote paths should be absolute (starting with `/`)
- **Parameter order differs between get and put**: `get(remote, local)` downloads while `put(local, remote)` uploads; note the reversed parameter order
- **File overwrites**: Both `get()` and `put()` silently overwrite existing files at the destination
- **No recursive operations**: `list()` returns entries for a single directory; traverse subdirectories manually
- **No mkdir**: The library does not provide a method to create remote directories; ensure target directories exist on the server
- **Modify timestamp format**: `FtpEntry.modify` uses RFC 3659 format (`YYYYMMDDhhmmss`), not ISO 8601
- **Deletion is permanent**: `delete()` cannot be undone; there is no server-side recycle bin
- **Timeout applies per operation**: A single `get()` or `put()` must complete within the configured timeout
