# SSH/SFTP Secure File Transfer

SSH-based secure file transfer integration for GreyCat.

## Overview

The SSH library provides secure file transfer capabilities over the SSH protocol using SFTP (SSH File Transfer Protocol). It enables GreyCat applications to upload, download, list, and delete files on remote servers through encrypted connections.

Key features include:
- **Password and key-based authentication** supporting both `SshPasswordAuth` and `SshKeyAuth`
- **SFTP file operations** including upload, download, list, and delete
- **Automatic remote directory creation** when uploading files with `put()`
- **File metadata access** including size, permissions, timestamps, and ownership
- **File type detection** distinguishing regular files, directories, symlinks, and more

This library is ideal for automated file transfers, remote server management, secure backup operations, or building ETL pipelines that pull data from external SFTP servers.

## Installation

Add the SSH library to your GreyCat project:

```gcl
@library("ssh", "7.7.120-dev")
```

## Quick Start

### Connect and Download a File

```gcl
var sftp = Sftp {
  addr: "example.com",
  auth: SshPasswordAuth {
    username: "deploy",
    password: "s3cureP@ss"
  }
};

// Download a remote file
sftp.get("/remote/data/report.csv", "/local/data/report.csv");

// Upload a local file (remote directories created automatically)
sftp.put("/local/data/results.csv", "/remote/exports/2024/results.csv");
```

### List Files in a Remote Directory

```gcl
var files = sftp.list("/remote/data/");

for (file in files) {
  print("${file.path} - ${file.stat.size} bytes - ${file.type}");
}
```

## Types

### SshAuth

Abstract base type for SSH authentication. Cannot be instantiated directly; use `SshPasswordAuth` or `SshKeyAuth` instead.

**Fields:**
- `username: String` - SSH username for authentication (private)

### SshPasswordAuth

Password-based SSH authentication. Extends `SshAuth`.

**Fields:**
- `username: String` - SSH username (inherited from `SshAuth`, private)
- `password: String` - SSH password (private)

**Example:**

```gcl
var auth = SshPasswordAuth {
  username: "deploy",
  password: env("SSH_PASSWORD")
};

var sftp = Sftp {
  addr: "server.example.com",
  auth: auth
};
```

### SshKeyAuth

Public/private key-based SSH authentication. Extends `SshAuth`.

**Fields:**
- `username: String` - SSH username (inherited from `SshAuth`, private)
- `pubkey: String` - Path to the public key file (private)
- `privkey: String` - Path to the private key file (private)
- `passphrase: String?` - Optional passphrase for the private key (private)

**Example:**

```gcl
// Key-based auth without passphrase
var auth = SshKeyAuth {
  username: "deploy",
  pubkey: "/home/app/.ssh/id_rsa.pub",
  privkey: "/home/app/.ssh/id_rsa"
};

// Key-based auth with passphrase
var authProtected = SshKeyAuth {
  username: "deploy",
  pubkey: "/home/app/.ssh/id_ed25519.pub",
  privkey: "/home/app/.ssh/id_ed25519",
  passphrase: env("SSH_KEY_PASSPHRASE")
};

var sftp = Sftp {
  addr: "secure-server.example.com",
  auth: authProtected
};
```

### FileStat

Metadata about a file on the remote server.

**Fields:**
- `size: int?` - File size in bytes
- `uid: int?` - Owner user ID
- `gid: int?` - Owner group ID
- `perm: int?` - File permissions (Unix-style, e.g., `0o644`)
- `atime: time?` - Last access time
- `mtime: time?` - Last modification time

**Example:**

```gcl
var files = sftp.list("/remote/data/");

for (file in files) {
  var stat = file.stat;
  print("File: ${file.path}");
  print("  Size: ${stat.size} bytes");
  print("  Owner UID: ${stat.uid}");
  print("  Permissions: ${stat.perm}");
  print("  Modified: ${stat.mtime}");
}
```

### FileType

Enum representing the type of a filesystem entry.

**Values:**
- `NamedPipe` - Named pipe (FIFO)
- `CharDevice` - Character device
- `Directory` - Directory
- `BlockDevice` - Block device
- `RegularFile` - Regular file
- `Symlink` - Symbolic link
- `Socket` - Unix socket
- `Other` - Other/unknown file type

**Example:**

```gcl
var entries = sftp.list("/remote/data/");

for (entry in entries) {
  if (entry.type == FileType::Directory) {
    print("DIR:  ${entry.path}");
  } else if (entry.type == FileType::RegularFile) {
    print("FILE: ${entry.path} (${entry.stat.size} bytes)");
  } else if (entry.type == FileType::Symlink) {
    print("LINK: ${entry.path}");
  }
}
```

### Sftp

Main connection type representing an SFTP client over SSH.

**Fields:**
- `addr: String` - Server address (e.g., `"example.com"` or `"192.168.1.10:22"`, private)
- `auth: SshAuth` - Authentication method: `SshPasswordAuth` or `SshKeyAuth` (private)

**Methods:**
- `get(remote, local)` - Download a remote file to a local path
- `list(dirpath): Array<SftpFile>` - List files in a remote directory
- `delete(filepath)` - Delete a remote file
- `put(local, remote)` - Upload a local file to a remote path (auto-creates remote directories)

**Example:**

```gcl
// Password-based connection
var sftp = Sftp {
  addr: "sftp.example.com",
  auth: SshPasswordAuth {
    username: "user",
    password: env("SFTP_PASSWORD")
  }
};

// Key-based connection with custom port
var sftpKey = Sftp {
  addr: "secure-server.example.com:2222",
  auth: SshKeyAuth {
    username: "deploy",
    pubkey: "/home/app/.ssh/id_rsa.pub",
    privkey: "/home/app/.ssh/id_rsa"
  }
};
```

### SftpFile

Represents a file or directory entry returned by `list()`.

**Fields:**
- `path: String` - Full path of the entry
- `stat: FileStat` - File metadata (size, permissions, timestamps, etc.)
- `type: FileType` - Type of the filesystem entry

**Example:**

```gcl
var files = sftp.list("/remote/uploads/");

for (file in files) {
  print("Path: ${file.path}");
  print("  Type: ${file.type}");
  print("  Size: ${file.stat.size} bytes");
  print("  Modified: ${file.stat.mtime}");
}
```

## Methods

### get()

Downloads a file from the remote server to a local path.

**Signature:** `fn get(remote: String, local: String)`

**Parameters:**
- `remote: String` - Remote file path to download
- `local: String` - Local destination file path

**Behavior:**
- Downloads the entire file over SFTP
- Overwrites existing local file
- Throws error if remote file does not exist

**Example:**

```gcl
// Download a file
sftp.get("/remote/data/report.csv", "/local/data/report.csv");

// Download from nested directory
sftp.get("/home/deploy/exports/2024/01/data.json", "/tmp/data.json");

// Download with error handling
try {
  sftp.get("/remote/important-file.txt", "/tmp/important.txt");
  print("Download successful");
} catch (e) {
  print("Download failed: ${e}");
}
```

### list()

Lists files and directories in a remote directory.

**Signature:** `fn list(dirpath: String): Array<SftpFile>`

**Parameters:**
- `dirpath: String` - Remote directory path to list

**Returns:** Array of `SftpFile` instances containing path, metadata, and type information

**Behavior:**
- Returns all entries in the specified directory
- Each entry includes full path, file stat metadata, and file type
- Throws error if directory does not exist or is not accessible

**Example:**

```gcl
// List all entries in a directory
var entries = sftp.list("/remote/data/");

for (entry in entries) {
  print("${entry.path} - ${entry.type}");
}

// Filter for regular files only
var entries = sftp.list("/remote/uploads/");

for (entry in entries) {
  if (entry.type == FileType::RegularFile) {
    print("${entry.path} (${entry.stat.size} bytes)");
  }
}

// Find large files
var entries = sftp.list("/remote/logs/");

for (entry in entries) {
  if (entry.type == FileType::RegularFile && entry.stat.size > 1_000_000) {
    print("Large file: ${entry.path} (${entry.stat.size} bytes)");
  }
}
```

### delete()

Deletes a file on the remote server.

**Signature:** `fn delete(filepath: String)`

**Parameters:**
- `filepath: String` - Remote file path to delete

**Behavior:**
- Permanently deletes the specified remote file
- Throws error if file does not exist or is not accessible

**Example:**

```gcl
// Delete a single file
sftp.delete("/remote/tmp/old-report.csv");

// Delete multiple files after processing
var files = sftp.list("/remote/incoming/");

for (file in files) {
  if (file.type == FileType::RegularFile) {
    // Process file first
    sftp.get(file.path, "/local/processed/${file.path}");
    // Then delete remote copy
    sftp.delete(file.path);
    print("Processed and removed: ${file.path}");
  }
}
```

### put()

Uploads a local file to the remote server. Automatically creates any missing directories in the remote path.

**Signature:** `fn put(local: String, remote: String)`

**Parameters:**
- `local: String` - Local source file path
- `remote: String` - Remote destination file path

**Behavior:**
- Uploads the entire file over SFTP
- Overwrites existing remote file with the same path
- **Automatically creates intermediate remote directories** if they do not exist
- Throws error if local file does not exist

**Example:**

```gcl
// Upload a file
sftp.put("/local/data/export.csv", "/remote/imports/export.csv");

// Upload to a deeply nested path (directories created automatically)
sftp.put("/local/report.pdf", "/remote/reports/2024/01/15/daily-report.pdf");

// Batch upload
var files = ["data1.csv", "data2.csv", "data3.csv"];
for (file in files) {
  sftp.put("/local/exports/${file}", "/remote/imports/${file}");
  print("Uploaded: ${file}");
}

// Upload with error handling
try {
  sftp.put("/local/big-file.bin", "/remote/uploads/big-file.bin");
  print("Upload successful");
} catch (e) {
  print("Upload failed: ${e}");
}
```

## Common Use Cases

### Secure File Transfer

```gcl
var sftp = Sftp {
  addr: "sftp.partner.com",
  auth: SshPasswordAuth {
    username: env("SFTP_USER"),
    password: env("SFTP_PASSWORD")
  }
};

// Download incoming files from partner
var incoming = sftp.list("/incoming/");

for (file in incoming) {
  if (file.type == FileType::RegularFile) {
    sftp.get(file.path, "/local/partner-data/${file.path}");
    print("Downloaded: ${file.path}");
  }
}

// Upload outgoing files to partner
sftp.put("/local/exports/daily-feed.csv", "/outgoing/daily-feed.csv");
print("Sent daily feed to partner");
```

### Directory Listing and Filtering

```gcl
var sftp = Sftp {
  addr: "fileserver.internal:22",
  auth: SshPasswordAuth {
    username: "reader",
    password: env("SFTP_PASSWORD")
  }
};

// List and categorize directory contents
var entries = sftp.list("/shared/data/");

var fileCount = 0;
var dirCount = 0;
var totalSize = 0;

for (entry in entries) {
  if (entry.type == FileType::RegularFile) {
    fileCount = fileCount + 1;
    totalSize = totalSize + entry.stat.size;
  } else if (entry.type == FileType::Directory) {
    dirCount = dirCount + 1;
  }
}

print("Files: ${fileCount}, Directories: ${dirCount}, Total size: ${totalSize} bytes");
```

### Key-Based Authentication

```gcl
// Production deployment using SSH keys (more secure than passwords)
var sftp = Sftp {
  addr: "production-server.example.com",
  auth: SshKeyAuth {
    username: "deploy",
    pubkey: "/etc/app/ssh/deploy_key.pub",
    privkey: "/etc/app/ssh/deploy_key",
    passphrase: env("DEPLOY_KEY_PASSPHRASE")
  }
};

// Deploy configuration files
sftp.put("/local/config/app.conf", "/etc/myapp/app.conf");
sftp.put("/local/config/db.conf", "/etc/myapp/db.conf");
print("Configuration deployed");

// Pull logs back for analysis
var logFiles = sftp.list("/var/log/myapp/");

for (logFile in logFiles) {
  if (logFile.type == FileType::RegularFile) {
    sftp.get(logFile.path, "/local/logs/${logFile.path}");
  }
}
print("Logs retrieved for analysis");
```

### ETL Pipeline with Remote SFTP Source

```gcl
var sftp = Sftp {
  addr: "data-provider.example.com",
  auth: SshKeyAuth {
    username: "etl-service",
    pubkey: "/app/keys/etl.pub",
    privkey: "/app/keys/etl"
  }
};

// Pull new data files
var remoteFiles = sftp.list("/exports/daily/");

for (file in remoteFiles) {
  if (file.type == FileType::RegularFile) {
    var localPath = "/tmp/etl/${file.path}";
    sftp.get(file.path, localPath);

    // Process the file locally
    processData(localPath);

    // Clean up remote file after successful processing
    sftp.delete(file.path);
    print("ETL complete for: ${file.path}");
  }
}
```

## Best Practices

### Credential Management

- **Never hardcode credentials**: Use environment variables or secure configuration
- **Prefer key-based authentication**: SSH keys are more secure than passwords
- **Protect private keys**: Ensure key files have restrictive permissions
- **Use passphrases on keys**: Add an extra layer of protection for private keys

```gcl
// Good: Load credentials from environment
var sftp = Sftp {
  addr: env("SFTP_HOST"),
  auth: SshKeyAuth {
    username: env("SFTP_USER"),
    pubkey: env("SFTP_PUBKEY_PATH"),
    privkey: env("SFTP_PRIVKEY_PATH"),
    passphrase: env("SFTP_KEY_PASSPHRASE")
  }
};

// Bad: Hardcoded credentials
var sftp = Sftp {
  addr: "server.example.com",
  auth: SshPasswordAuth {
    username: "admin",       // NEVER DO THIS!
    password: "password123"  // NEVER DO THIS!
  }
};
```

### Address Format

- **Default port**: If no port is specified, SSH defaults to port `22`
- **Custom port**: Append the port to the address (e.g., `"server.example.com:2222"`)
- **IP addresses**: Both hostnames and IP addresses are supported

```gcl
// Default port 22
var sftp1 = Sftp { addr: "example.com", auth: auth };

// Custom port
var sftp2 = Sftp { addr: "example.com:2222", auth: auth };

// IP address
var sftp3 = Sftp { addr: "192.168.1.10", auth: auth };

// IP address with custom port
var sftp4 = Sftp { addr: "192.168.1.10:2222", auth: auth };
```

### Error Handling

- **Wrap operations in try-catch**: Network issues, authentication failures, missing files
- **Verify downloads**: Check that local files were written successfully
- **Handle missing files gracefully**: `get()` and `delete()` fail if the file does not exist

```gcl
try {
  sftp.get("/remote/data.csv", "/local/data.csv");
  print("Download successful");
} catch (e) {
  print("Failed to download: ${e}");
  // Handle network issue, auth failure, or missing file
}

try {
  sftp.put("/local/upload.csv", "/remote/upload.csv");
  print("Upload successful");
} catch (e) {
  print("Failed to upload: ${e}");
}
```

### Gotchas

- **Auto-creation of remote directories on `put()`**: The `put()` method automatically creates any missing intermediate directories in the remote path. This is convenient but means typos in remote paths will silently create unintended directory structures.
- **No `put()` directory creation for `get()`**: Unlike `put()`, the `get()` method does not automatically create local directories. Ensure the local destination directory exists before downloading.
- **Address format**: The `addr` field accepts either a hostname (`"example.com"`) or a host with port (`"example.com:22"`). Port 22 is used by default if omitted.
- **Private fields**: All fields on `SshAuth`, `SshPasswordAuth`, `SshKeyAuth`, and `Sftp` are private. They are set at construction time and cannot be read or modified afterward.
- **File vs. directory operations**: The `delete()` method operates on files. To remove directories, ensure they are empty first.
- **FileStat nullable fields**: All fields on `FileStat` are nullable (`int?`, `time?`). Always check for `null` before using these values, as not all servers populate every field.
- **Key file paths**: `SshKeyAuth` expects file system paths to the key files, not the key contents. Ensure the key files are accessible from the GreyCat runtime environment.
- **Overwriting on upload**: `put()` silently overwrites an existing remote file with the same path, with no confirmation or warning.
