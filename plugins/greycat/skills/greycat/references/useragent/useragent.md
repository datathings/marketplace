# User Agent Parser

HTTP User Agent string parsing for browser and device detection in GreyCat.

## Overview

The User Agent library provides parsing and analysis of HTTP User-Agent strings, enabling applications to identify browsers, operating systems, and devices from web traffic. Built on industry-standard user agent parsing, it extracts structured information from the complex User-Agent header sent by web browsers and applications.

Key features include:
- **Browser identification** including family, version (major, minor, patch)
- **Operating system detection** with version information
- **Device detection** including family, brand, and model
- **Comprehensive version parsing** with granular version components
- **Support for modern and legacy user agents** across desktop, mobile, and IoT devices

This library is ideal for web analytics, device-specific content delivery, compatibility detection, security analysis, and understanding user demographics in web applications.

## Installation

Add the User Agent library to your GreyCat project:

```gcl
@library("useragent", "7.7.2-dev")
```

## Quick Start

### Parse a User Agent String

```gcl
var uaString = "Mozilla/5.0 (Linux; Android 4.4.2; QMV7A Build/KOT49H) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.135 Safari/537.36";

var ua = UserAgent::parse(uaString);

print("Browser: ${ua.browserFamily} ${ua.browserMajor}.${ua.browserMinor}");
print("OS: ${ua.osFamily} ${ua.osMajor}.${ua.osMinor}");
print("Device: ${ua.deviceBrand} ${ua.deviceModel}");
```

### Detect Mobile Devices

```gcl
fn isMobile(uaString: String): bool {
  var ua = UserAgent::parse(uaString);

  var mobileFamilies = ["Android", "iOS", "Windows Phone", "BlackBerry"];

  return mobileFamilies.contains(ua.osFamily);
}
```

## Types

### UserAgent

Parsed user agent information with browser, OS, and device details.

**Fields:**

**Browser Information:**
- `browserFamily: String?` - Browser name (e.g., "Chrome", "Firefox", "Safari")
- `browserMajor: String?` - Browser major version (e.g., "36")
- `browserMinor: String?` - Browser minor version (e.g., "0")
- `browserPatch: String?` - Browser patch version (e.g., "1985")

**Operating System Information:**
- `osFamily: String?` - OS name (e.g., "Android", "iOS", "Windows", "Mac OS X")
- `osMajor: String?` - OS major version (e.g., "4")
- `osMinor: String?` - OS minor version (e.g., "4")
- `osPatch: String?` - OS patch version (e.g., "2")
- `osPatchMinor: String?` - OS minor patch version (e.g., "4")

**Device Information:**
- `deviceFamily: String?` - Device family (e.g., "iPhone", "Samsung Galaxy")
- `deviceBrand: String?` - Device manufacturer (e.g., "Apple", "Samsung")
- `deviceModel: String?` - Device model (e.g., "11", "S21")

**Static Methods:**
- `parse(userAgent: String): UserAgent` - Parse a user agent string

**Note:** All fields are optional (`?`) as not all information may be present in every user agent string.

**Example:**

```gcl
var ua = UserAgent::parse("Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1");

print(ua.browserFamily);  // "Mobile Safari"
print(ua.browserMajor);   // "14"
print(ua.osFamily);       // "iOS"
print(ua.osMajor);        // "14"
print(ua.osMinor);        // "6"
print(ua.deviceFamily);   // "iPhone"
print(ua.deviceBrand);    // "Apple"
```

## Methods

### parse()

Parses a User-Agent string into structured components.

**Signature:** `static fn parse(userAgent: String): UserAgent`

**Parameters:**
- `userAgent: String` - The User-Agent string from HTTP headers

**Returns:** `UserAgent` object with parsed information

**Example:**

```gcl
// Desktop browser
var desktop = UserAgent::parse("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36");

print("Browser: ${desktop.browserFamily} ${desktop.browserMajor}");
// "Browser: Chrome 120"

print("OS: ${desktop.osFamily} ${desktop.osMajor}.${desktop.osMinor}");
// "OS: Windows 10.0"

// Mobile browser
var mobile = UserAgent::parse("Mozilla/5.0 (Linux; Android 13; SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36");

print("Device: ${mobile.deviceBrand} ${mobile.deviceModel}");
// "Device: Samsung SM-S918B"

// Bot/Crawler
var bot = UserAgent::parse("Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)");

print("Browser: ${bot.browserFamily}");
// "Browser: Googlebot"
```

## Common Use Cases

### Browser-Specific Features

```gcl
fn supportsWebP(uaString: String): bool {
  var ua = UserAgent::parse(uaString);

  // WebP supported in Chrome 23+, Firefox 65+, Edge 18+
  if (ua.browserFamily == "Chrome") {
    var major = ua.browserMajor as int;
    return major >= 23;
  } else if (ua.browserFamily == "Firefox") {
    var major = ua.browserMajor as int;
    return major >= 65;
  } else if (ua.browserFamily == "Edge") {
    var major = ua.browserMajor as int;
    return major >= 18;
  }

  return false;
}

fn serveImage(uaString: String, imageName: String): String {
  if (supportsWebP(uaString)) {
    return "/images/${imageName}.webp";
  } else {
    return "/images/${imageName}.jpg";
  }
}
```

### Analytics Dashboard

```gcl
type VisitorStats {
  total: int;
  byBrowser: Map<String, int>;
  byOS: Map<String, int>;
  byDevice: Map<String, int>;
}

fn analyzeVisitors(userAgents: Array<String>): VisitorStats {
  var stats = VisitorStats {
    total: userAgents.size(),
    byBrowser: Map<String, int>{},
    byOS: Map<String, int>{},
    byDevice: Map<String, int>{}
  };

  for (uaString in userAgents) {
    var ua = UserAgent::parse(uaString);

    // Count browsers
    if (ua.browserFamily != null) {
      var count = stats.byBrowser.get(ua.browserFamily);
      stats.byBrowser.put(ua.browserFamily, count == null ? 1 : count + 1);
    }

    // Count operating systems
    if (ua.osFamily != null) {
      var count = stats.byOS.get(ua.osFamily);
      stats.byOS.put(ua.osFamily, count == null ? 1 : count + 1);
    }

    // Count device brands
    if (ua.deviceBrand != null) {
      var count = stats.byDevice.get(ua.deviceBrand);
      stats.byDevice.put(ua.deviceBrand, count == null ? 1 : count + 1);
    }
  }

  return stats;
}

// Generate report
var stats = analyzeVisitors(visitorUserAgents);
print("=== Visitor Statistics ===");
print("Total visitors: ${stats.total}");
print("\nTop Browsers:");
for (entry in stats.byBrowser.entries()) {
  var percentage = (entry.value as float / stats.total) * 100;
  print("  ${entry.key}: ${entry.value} (${percentage}%)");
}
```

### Mobile vs Desktop Detection

```gcl
type DeviceType {
  mobile: bool;
  tablet: bool;
  desktop: bool;
}

fn detectDeviceType(uaString: String): DeviceType {
  var ua = UserAgent::parse(uaString);

  var mobileOS = ["Android", "iOS", "Windows Phone", "BlackBerry"];
  var tabletDevices = ["iPad", "Android"]; // Simplified

  var isMobile = false;
  var isTablet = false;

  if (ua.osFamily != null && mobileOS.contains(ua.osFamily)) {
    isMobile = true;

    // Check if it's actually a tablet
    if (ua.deviceFamily != null) {
      if (ua.deviceFamily.contains("iPad") || ua.deviceFamily.contains("Tablet")) {
        isTablet = true;
        isMobile = false;
      }
    }
  }

  return DeviceType {
    mobile: isMobile,
    tablet: isTablet,
    desktop: !isMobile && !isTablet
  };
}

// Serve appropriate content
fn getTemplate(uaString: String): String {
  var deviceType = detectDeviceType(uaString);

  if (deviceType.mobile) {
    return "mobile-template.html";
  } else if (deviceType.tablet) {
    return "tablet-template.html";
  } else {
    return "desktop-template.html";
  }
}
```

### Bot Detection

```gcl
fn isBot(uaString: String): bool {
  var ua = UserAgent::parse(uaString);

  if (ua.browserFamily == null) {
    return false;
  }

  var botNames = [
    "Googlebot",
    "Bingbot",
    "Slurp",        // Yahoo
    "DuckDuckBot",
    "Baiduspider",
    "YandexBot",
    "facebookexternalhit"
  ];

  for (botName in botNames) {
    if (ua.browserFamily.contains(botName)) {
      return true;
    }
  }

  return false;
}

fn handleRequest(uaString: String) {
  if (isBot(uaString)) {
    // Serve static, SEO-optimized content
    return serveBotContent();
  } else {
    // Serve interactive JavaScript application
    return serveAppContent();
  }
}
```

### Browser Version Compatibility

```gcl
type BrowserSupport {
  supported: bool;
  reason: String?;
}

fn checkBrowserSupport(uaString: String): BrowserSupport {
  var ua = UserAgent::parse(uaString);

  if (ua.browserFamily == null || ua.browserMajor == null) {
    return BrowserSupport {
      supported: false,
      reason: "Unable to detect browser"
    };
  }

  var major = ua.browserMajor as int;

  // Define minimum versions
  if (ua.browserFamily == "Chrome" && major < 90) {
    return BrowserSupport {
      supported: false,
      reason: "Chrome 90+ required"
    };
  } else if (ua.browserFamily == "Firefox" && major < 88) {
    return BrowserSupport {
      supported: false,
      reason: "Firefox 88+ required"
    };
  } else if (ua.browserFamily == "Safari" && major < 14) {
    return BrowserSupport {
      supported: false,
      reason: "Safari 14+ required"
    };
  } else if (ua.browserFamily == "IE") {
    return BrowserSupport {
      supported: false,
      reason: "Internet Explorer is not supported"
    };
  }

  return BrowserSupport {
    supported: true,
    reason: null
  };
}

// Usage in web application
fn renderPage(uaString: String) {
  var support = checkBrowserSupport(uaString);

  if (!support.supported) {
    return renderUpgradePage(support.reason);
  }

  return renderApplication();
}
```

### Geographic and Localization

```gcl
fn getLocaleFromUA(uaString: String): String {
  var ua = UserAgent::parse(uaString);

  // Some user agents include locale information
  // This is a simplified example
  if (ua.osFamily == "iOS") {
    return "en-US"; // Could parse from UA string
  } else if (ua.osFamily == "Android") {
    return "en-US"; // Could parse from UA string
  }

  return "en-US"; // Default
}
```

### Security Analysis

```gcl
type SecurityRisk {
  level: String;
  issues: Array<String>;
}

fn assessSecurityRisk(uaString: String): SecurityRisk {
  var ua = UserAgent::parse(uaString);
  var issues = Array<String>{};
  var level = "LOW";

  // Check for very old browsers
  if (ua.browserFamily == "IE") {
    issues.add("Internet Explorer detected (unsupported)");
    level = "HIGH";
  }

  if (ua.browserFamily == "Chrome" && ua.browserMajor != null) {
    var major = ua.browserMajor as int;
    if (major < 100) {
      issues.add("Outdated Chrome version");
      level = "MEDIUM";
    }
  }

  // Check for uncommon user agents (potential scrapers)
  if (ua.browserFamily == null && ua.osFamily == null) {
    issues.add("Incomplete user agent (potential bot)");
    level = "MEDIUM";
  }

  return SecurityRisk {
    level: level,
    issues: issues
  };
}
```

## Best Practices

### Null Checks

Always check for null values since not all fields are populated:

```gcl
// Good: Null-safe
var ua = UserAgent::parse(uaString);

if (ua.browserFamily != null) {
  print("Browser: ${ua.browserFamily}");
} else {
  print("Browser: Unknown");
}

// Bad: May cause error
var ua = UserAgent::parse(uaString);
print("Browser: ${ua.browserFamily}"); // Could be null!
```

### Version Comparison

When comparing versions, handle null and convert to int:

```gcl
fn isModernChrome(ua: UserAgent): bool {
  if (ua.browserFamily != "Chrome" || ua.browserMajor == null) {
    return false;
  }

  var major = ua.browserMajor as int;
  return major >= 100;
}
```

### Caching Results

Parse once and reuse:

```gcl
// Good: Parse once
var ua = UserAgent::parse(uaString);
var isMobile = checkMobile(ua);
var browser = getBrowserName(ua);
var os = getOSName(ua);

// Bad: Parse multiple times
var isMobile = checkMobile(UserAgent::parse(uaString));
var browser = getBrowserName(UserAgent::parse(uaString));
var os = getOSName(UserAgent::parse(uaString));
```

### Fallback Values

Provide defaults for unknown values:

```gcl
fn getBrowserDisplay(ua: UserAgent): String {
  var browser = ua.browserFamily != null ? ua.browserFamily : "Unknown Browser";
  var version = ua.browserMajor != null ? ua.browserMajor : "?";

  return "${browser} ${version}";
}
```

### Feature Detection Over UA Sniffing

When possible, use feature detection instead of user agent sniffing:

```gcl
// Better: Feature detection (client-side)
// if (window.WebGLRenderingContext) { ... }

// Acceptable: UA parsing for analytics or server-side decisions
var ua = UserAgent::parse(uaString);
logBrowserStats(ua);
```

### Gotchas

- **All fields are optional**: Any field can be `null`
- **Version strings**: Versions are strings, not numbers (cast when comparing)
- **Browser aliases**: Same browser may have different names (e.g., "Chrome Mobile")
- **Device detection limitations**: Not all devices are accurately detected
- **User agent spoofing**: Clients can send fake user agents
- **Empty strings vs null**: Some fields may be empty strings rather than null
- **Case sensitivity**: Browser/OS names are case-sensitive
- **Bot detection**: Simple bot detection is unreliable; use additional signals

### Common User Agent Patterns

Desktop browsers:
```
Chrome (Windows): Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36
Firefox (Mac): Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0
Safari (Mac): Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15
```

Mobile browsers:
```
Chrome (Android): Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.43 Mobile Safari/537.36
Safari (iOS): Mozilla/5.0 (iPhone; CPU iPhone OS 17_1_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1
```

Bots:
```
Googlebot: Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)
```

### Security Considerations

- **Don't trust user agents**: They can be spoofed
- **Combine with other signals**: IP address, behavior analysis
- **Log suspicious patterns**: Unusual or malformed user agents
- **Rate limiting**: Use UA parsing to help identify bots

```gcl
fn isLikelySuspicious(uaString: String): bool {
  var ua = UserAgent::parse(uaString);

  // Very short or missing critical fields
  if (uaString.length() < 20) {
    return true;
  }

  // No browser family detected
  if (ua.browserFamily == null && ua.osFamily == null) {
    return true;
  }

  return false;
}
```

### Performance

Parsing is relatively fast, but consider caching for high-traffic applications:

```gcl
var uaCache = Map<String, UserAgent>{};

fn getCachedUserAgent(uaString: String): UserAgent {
  var cached = uaCache.get(uaString);

  if (cached == null) {
    cached = UserAgent::parse(uaString);
    uaCache.put(uaString, cached);
  }

  return cached;
}
```

### Browser Family Values

Common values you might encounter:

**Desktop Browsers:**
- `"Chrome"`, `"Firefox"`, `"Safari"`, `"Edge"`, `"Opera"`, `"IE"`

**Mobile Browsers:**
- `"Mobile Safari"`, `"Chrome Mobile"`, `"Firefox Mobile"`, `"Samsung Internet"`

**Bots/Crawlers:**
- `"Googlebot"`, `"Bingbot"`, `"Baiduspider"`, `"YandexBot"`

### OS Family Values

Common operating systems:
- `"Windows"`, `"Mac OS X"`, `"Linux"`, `"Ubuntu"`
- `"Android"`, `"iOS"`, `"Chrome OS"`
- `"Windows Phone"`, `"BlackBerry"`

### Device Brand Values

Common device manufacturers:
- `"Apple"`, `"Samsung"`, `"Huawei"`, `"Xiaomi"`, `"Google"`, `"LG"`, `"Sony"`

## Example: Complete Request Handler

```gcl
type RequestInfo {
  ua: UserAgent;
  isMobile: bool;
  isBot: bool;
  browserSupported: bool;
  deviceType: String;
}

fn analyzeRequest(uaString: String): RequestInfo {
  var ua = UserAgent::parse(uaString);

  // Determine device type
  var deviceType = "desktop";
  if (ua.osFamily != null) {
    if (["Android", "iOS"].contains(ua.osFamily)) {
      deviceType = ua.deviceFamily != null && ua.deviceFamily.contains("iPad") ? "tablet" : "mobile";
    }
  }

  // Check if bot
  var isBot = ua.browserFamily != null &&
    (ua.browserFamily.contains("bot") || ua.browserFamily.contains("Bot"));

  // Check browser support
  var supported = true;
  if (ua.browserFamily == "IE") {
    supported = false;
  } else if (ua.browserMajor != null) {
    var major = ua.browserMajor as int;
    if (ua.browserFamily == "Chrome" && major < 90) {
      supported = false;
    }
  }

  return RequestInfo {
    ua: ua,
    isMobile: deviceType == "mobile",
    isBot: isBot,
    browserSupported: supported,
    deviceType: deviceType
  };
}

// Use in request handling
fn handleWebRequest(uaString: String) {
  var info = analyzeRequest(uaString);

  print("Device: ${info.deviceType}");
  print("Browser: ${info.ua.browserFamily} ${info.ua.browserMajor}");
  print("OS: ${info.ua.osFamily}");

  if (info.isBot) {
    return serveBotContent();
  } else if (!info.browserSupported) {
    return serveUpgradePage();
  } else if (info.isMobile) {
    return serveMobilePage();
  } else {
    return serveDesktopPage();
  }
}
```
