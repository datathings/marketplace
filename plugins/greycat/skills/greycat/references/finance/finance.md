# Finance Library

Financial data types and validation for GreyCat.

## Overview

The Finance library provides specialized types and utilities for working with financial data in GreyCat applications. Currently focused on International Bank Account Number (IBAN) parsing and validation, it enables applications to handle banking information with proper validation and structure.

Key features include:
- **IBAN parsing** with automatic validation
- **Country code extraction** from IBAN strings
- **Bank code identification** for routing
- **Account number parsing** from IBAN format
- **Format validation** ensuring IBAN compliance

This library is ideal for payment processing systems, banking applications, financial data validation, accounting software, and any system that handles international bank transfers.

## Installation

Add the Finance library to your GreyCat project:

```gcl
@library("finance", "7.7.2-dev")
```

## Quick Start

### Parse and Validate IBAN

```gcl
// Parse a valid IBAN
var iban = Iban::parse("GB82WEST12345698765432");

if (iban != null) {
  print("Country: ${iban.country}");
  print("Bank: ${iban.bank}");
  print("Account: ${iban.account}");
  print("Full IBAN: ${iban.iban}");
} else {
  print("Invalid IBAN");
}
```

### Validate User Input

```gcl
fn processPayment(ibanString: String, amount: float) {
  var iban = Iban::parse(ibanString);

  if (iban == null) {
    throw "Invalid IBAN: ${ibanString}";
  }

  print("Processing payment of ${amount} to ${iban.country} account");
  // ... payment logic
}
```

## Types

### Iban

Represents a validated International Bank Account Number.

**Fields:**
- `iban: String` - Full IBAN string in standard format
- `country: String` - ISO 3166-1 alpha-2 country code (e.g., "GB", "DE", "FR")
- `bank: String` - Bank identifier code
- `account: String` - Account number

**Static Methods:**
- `parse(iban: String): Iban?` - Parse and validate IBAN string

**Example:**

```gcl
var iban = Iban::parse("DE89370400440532013000");

print(iban.iban);      // "DE89370400440532013000"
print(iban.country);   // "DE"
print(iban.bank);      // "37040044"
print(iban.account);   // "0532013000"
```

## Methods

### parse()

Parses and validates an IBAN string, returning a structured Iban object.

**Signature:** `static fn parse(iban: String): Iban?`

**Parameters:**
- `iban: String` - IBAN string to parse (with or without spaces)

**Returns:**
- `Iban` object if valid
- `null` if invalid or malformed

**Validation Performed:**
- Length validation (varies by country)
- Character validation (alphanumeric only)
- Check digit validation using modulo-97 algorithm
- Country code validation

**Example:**

```gcl
// Valid IBANs from different countries
var gbIban = Iban::parse("GB82WEST12345698765432");
var deIban = Iban::parse("DE89370400440532013000");
var frIban = Iban::parse("FR1420041010050500013M02606");
var esIban = Iban::parse("ES9121000418450200051332");

// IBAN with spaces (automatically normalized)
var spacedIban = Iban::parse("GB82 WEST 1234 5698 7654 32");
print(spacedIban.iban); // "GB82WEST12345698765432" (spaces removed)

// Invalid IBANs return null
var invalid1 = Iban::parse("INVALID");           // null
var invalid2 = Iban::parse("GB82WEST123456");    // null (wrong length)
var invalid3 = Iban::parse("GB00WEST12345698765432"); // null (bad check digit)
```

## Common Use Cases

### Payment Processing

```gcl
type PaymentRequest {
  sender_iban: String;
  recipient_iban: String;
  amount: float;
  currency: String;
}

fn validateAndProcessPayment(request: PaymentRequest) {
  // Validate sender IBAN
  var senderIban = Iban::parse(request.sender_iban);
  if (senderIban == null) {
    throw "Invalid sender IBAN: ${request.sender_iban}";
  }

  // Validate recipient IBAN
  var recipientIban = Iban::parse(request.recipient_iban);
  if (recipientIban == null) {
    throw "Invalid recipient IBAN: ${request.recipient_iban}";
  }

  // Check for same-country transfer
  var domestic = senderIban.country == recipientIban.country;
  var fee = domestic ? 1.0 : 5.0;

  print("Transfer type: ${domestic ? 'Domestic' : 'International'}");
  print("Fee: ${fee} ${request.currency}");
  print("From: ${senderIban.country} - ${senderIban.account}");
  print("To: ${recipientIban.country} - ${recipientIban.account}");

  // Process payment
  executeTransfer(senderIban, recipientIban, request.amount, fee);
}
```

### Batch IBAN Validation

```gcl
var ibanList = [
  "GB82WEST12345698765432",
  "DE89370400440532013000",
  "INVALID_IBAN",
  "FR1420041010050500013M02606",
  "GB00WEST12345698765432" // Invalid check digit
];

var validIbans = Array<Iban>{};
var invalidIbans = Array<String>{};

for (ibanString in ibanList) {
  var iban = Iban::parse(ibanString);

  if (iban != null) {
    validIbans.add(iban);
  } else {
    invalidIbans.add(ibanString);
  }
}

print("Valid IBANs: ${validIbans.size()}");
print("Invalid IBANs: ${invalidIbans.size()}");

for (invalidIban in invalidIbans) {
  print("  - ${invalidIban}");
}
```

### Country-Based Routing

```gcl
fn routePayment(ibanString: String, amount: float) {
  var iban = Iban::parse(ibanString);

  if (iban == null) {
    throw "Invalid IBAN";
  }

  // Route to appropriate payment processor based on country
  if (iban.country == "GB") {
    processFasterPayment(iban, amount);
  } else if (iban.country == "DE" || iban.country == "FR" || iban.country == "ES") {
    processSEPAPayment(iban, amount);
  } else if (iban.country == "US") {
    throw "US does not use IBAN, use ACH routing";
  } else {
    processSWIFTPayment(iban, amount);
  }
}

fn processSEPAPayment(iban: Iban, amount: float) {
  print("Processing SEPA payment to ${iban.country}");
  print("Bank code: ${iban.bank}");
  print("Account: ${iban.account}");
  print("Amount: ${amount} EUR");
}
```

### Account Masking for Security

```gcl
fn maskIban(iban: Iban): String {
  // Show only last 4 digits of account
  var accountLength = iban.account.length();
  var masked = "";

  if (accountLength > 4) {
    for (i in 0..(accountLength - 4)) {
      masked = masked + "*";
    }
    masked = masked + iban.account.substring(accountLength - 4, accountLength);
  } else {
    masked = iban.account;
  }

  return "${iban.country}** **** **** ${masked}";
}

var iban = Iban::parse("GB82WEST12345698765432");
print(maskIban(iban)); // "GB** **** **** 5432"
```

### Database Storage

```gcl
type BankAccount {
  id: int;
  iban_string: String;
  country: String;
  bank_code: String;
  account_number: String;
  validated: bool;
}

fn saveBankAccount(ibanString: String): BankAccount? {
  var iban = Iban::parse(ibanString);

  if (iban == null) {
    print("Cannot save invalid IBAN: ${ibanString}");
    return null;
  }

  var account = BankAccount {
    id: generateId(),
    iban_string: iban.iban,
    country: iban.country,
    bank_code: iban.bank,
    account_number: iban.account,
    validated: true
  };

  // Save to database
  db.execute(
    "INSERT INTO bank_accounts (id, iban, country, bank_code, account_number, validated) VALUES (${account.id}, '${account.iban_string}', '${account.country}', '${account.bank_code}', '${account.account_number}', ${account.validated})"
  );

  return account;
}
```

### Payment Report Generation

```gcl
type Payment {
  id: String;
  iban: String;
  amount: float;
  date: time;
}

fn generatePaymentReport(payments: Array<Payment>) {
  var byCountry = Map<String, float>{};
  var invalidCount = 0;

  for (payment in payments) {
    var iban = Iban::parse(payment.iban);

    if (iban == null) {
      invalidCount = invalidCount + 1;
      continue;
    }

    var current = byCountry.get(iban.country);
    if (current == null) {
      byCountry.put(iban.country, payment.amount);
    } else {
      byCountry.put(iban.country, current + payment.amount);
    }
  }

  print("=== Payment Report ===");
  for (entry in byCountry.entries()) {
    print("${entry.key}: ${entry.value}");
  }
  print("Invalid IBANs: ${invalidCount}");
}
```

## Best Practices

### Validation Before Processing

Always validate IBANs before using them in financial operations:

```gcl
// Good: Validate first
fn processPayment(ibanString: String) {
  var iban = Iban::parse(ibanString);
  if (iban == null) {
    throw "Invalid IBAN";
  }
  // Safe to proceed
  executeTransfer(iban);
}

// Bad: Assuming validity
fn processPayment(ibanString: String) {
  // No validation - risky!
  executeTransfer(ibanString);
}
```

### User Input Handling

Accept IBANs with or without spaces, but normalize before storage:

```gcl
// Accept user input with spaces
var userInput = "GB82 WEST 1234 5698 7654 32";
var iban = Iban::parse(userInput);

if (iban != null) {
  // Store normalized version without spaces
  saveToDatabase(iban.iban); // "GB82WEST12345698765432"
}
```

### Error Messages

Provide clear feedback for invalid IBANs:

```gcl
fn validateIbanWithFeedback(ibanString: String): Iban? {
  var iban = Iban::parse(ibanString);

  if (iban == null) {
    // Could add more specific validation to identify issue
    if (ibanString.length() < 15) {
      print("IBAN too short");
    } else if (ibanString.length() > 34) {
      print("IBAN too long");
    } else {
      print("Invalid IBAN format or check digit");
    }
  }

  return iban;
}
```

### Null Checks

Always check for null after parsing:

```gcl
// Good: Null check
var iban = Iban::parse(input);
if (iban != null) {
  print(iban.country);
}

// Bad: May crash if invalid
var iban = Iban::parse(input);
print(iban.country); // Error if iban is null!
```

### Country-Specific Logic

Different countries have different banking systems:

```gcl
fn getPaymentMethod(iban: Iban): String {
  // SEPA countries
  var sepaCountries = ["AT", "BE", "DE", "ES", "FR", "IT", "NL", "PT"];

  if (sepaCountries.contains(iban.country)) {
    return "SEPA";
  } else if (iban.country == "GB") {
    return "Faster Payments";
  } else if (iban.country == "CH") {
    return "Swiss Interbank Clearing";
  } else {
    return "SWIFT";
  }
}
```

### Performance

For bulk validation, batch operations when possible:

```gcl
// Efficient: Batch validation
fn validateBatch(ibans: Array<String>): Map<String, Iban?> {
  var results = Map<String, Iban?>{};

  for (ibanString in ibans) {
    results.put(ibanString, Iban::parse(ibanString));
  }

  return results;
}
```

### Gotchas

- **Null return**: `parse()` returns `null` for invalid IBANs, always check
- **Spaces are ignored**: Input can have spaces, output `iban` field will not
- **Case sensitivity**: IBANs should be uppercase, but parser may be tolerant
- **Country codes**: 2-letter ISO codes, not 3-letter or country names
- **Length varies**: Different countries have different IBAN lengths (15-34 characters)
- **Not all countries use IBAN**: USA, Canada, Australia, and others use different systems

### IBAN Format Reference

Common IBAN formats by country:

| Country | Code | Length | Example |
|---------|------|--------|---------|
| United Kingdom | GB | 22 | GB82WEST12345698765432 |
| Germany | DE | 22 | DE89370400440532013000 |
| France | FR | 27 | FR1420041010050500013M02606 |
| Spain | ES | 24 | ES9121000418450200051332 |
| Italy | IT | 27 | IT60X0542811101000000123456 |
| Netherlands | NL | 18 | NL91ABNA0417164300 |
| Belgium | BE | 16 | BE68539007547034 |

For a complete list, see the [IBAN Registry](https://www.swift.com/standards/data-standards/iban).

### Security Considerations

- **Don't log full IBANs**: Mask account numbers in logs
- **Encrypt in storage**: IBANs are sensitive financial data
- **Validate on server**: Don't rely solely on client-side validation
- **Audit trail**: Log who accessed or modified IBAN data

```gcl
// Secure logging
fn logPayment(iban: Iban, amount: float) {
  var masked = "${iban.country}**************${iban.account.substring(iban.account.length() - 4)}";
  print("Payment to ${masked}: ${amount}");
}
```

### Future Extensions

This library currently focuses on IBAN validation. Future versions may include:

- **BIC/SWIFT code** validation and parsing
- **Credit card** validation (Luhn algorithm)
- **Currency** types and conversion
- **Money** type with precision handling
- **Tax ID** validation (VAT, EIN, etc.)

## Error Handling

The library uses null returns instead of exceptions:

```gcl
// Pattern 1: Null check with error
var iban = Iban::parse(input);
if (iban == null) {
  throw "Invalid IBAN: ${input}";
}
// Continue with valid iban

// Pattern 2: Early return
var iban = Iban::parse(input);
if (iban == null) {
  return;
}
// Continue with valid iban

// Pattern 3: Default value
var iban = Iban::parse(input);
var country = iban != null ? iban.country : "UNKNOWN";
```

## IBAN Structure

Understanding IBAN components:

```
GB82 WEST 1234 5698 7654 32
│ │  │    │              │
│ │  │    │              └─ Account Number
│ │  │    └──────────────── Bank Identifier
│ │  └─────────────────────┤
│ └──────────────────────── Check Digits (modulo-97)
└────────────────────────── Country Code (ISO 3166-1)
```

The check digits ensure IBAN integrity using the modulo-97 algorithm, which detects typos and transposition errors with high accuracy.
