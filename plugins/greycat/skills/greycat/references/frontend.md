# Frontend Integration

@greycat/web SDK for TypeScript frontend integration. Framework-agnostic — works with any framework (vanilla TS, React, Vue, Svelte, etc.).

## Overview

**@greycat/web SDK provides**: Auto-generated TypeScript types, auth/session management, typed API communication, error handling.

**Flow**: `Backend (GCL) -> greycat codegen ts -> project.d.ts -> Frontend (TS) -> Global gc namespace`

## Installation & Setup

```bash
# Install SDK (check https://get.greycat.io for latest)
npm install https://get.greycat.io/files/sdk/web/dev/7.7/7.7.1-dev.tgz
```

**vite.config.ts:**
```typescript
import { defineConfig } from 'vite';
import greycat from '@greycat/web/vite-plugin';

export default defineConfig({
  plugins: [greycat({ greycat: process.env.VITE_GREYCAT_URL || 'http://127.0.0.1:8080' })],
});
```

**tsconfig.json:**
```json
{
  "compilerOptions": { "target": "ES2020", "lib": ["ES2020", "DOM", "DOM.Iterable"], "module": "ESNext", "moduleResolution": "bundler", "strict": true },
  "include": ["src", "../project.d.ts"]
}
```

**src/vite-env.d.ts:**
```typescript
/// <reference types="vite/client" />
```

## SDK Initialization

**CRITICAL**: SDK must initialize BEFORE your app renders.

**index.ts:**
```typescript
import '@greycat/web';

await gc.sdk.init({
  numFmt: new Intl.NumberFormat('en-GB', { notation: 'compact', maximumSignificantDigits: 5 }), // optional
});

await import('./main');
```

**main.ts** then bootstraps your chosen framework (or vanilla TS).

## Backend to Frontend Workflow

### 1. Backend API Definition

**backend/src/api/country_api.gcl:**
```gcl
@expose
fn getStats(name: String): StatsView { /* ... */ }

@expose
fn listCountries(offset: int, maxCount: int): Array<CountryView> { /* ... */ }
```

### 2. Generate TypeScript Types

```bash
greycat codegen ts  # Generates project.d.ts
```

**package.json automation:**
```json
{
  "scripts": {
    "types": "cd .. && greycat codegen ts",
    "dev": "npm run types && vite",
    "build": "npm run types && tsc && vite build"
  }
}
```

**Generated types:**
```typescript
declare namespace gc {
  namespace country_api {
    const getStats: gc.sdk.ExposedFn<[string], gc.api_types.StatsView>;
    const listCountries: gc.sdk.ExposedFn<[number, number], Array<gc.api_types.CountryView>>;
  }
  export import getStats = gc.country_api.getStats;
  export import listCountries = gc.country_api.listCountries;
}
```

**NAMESPACE STRUCTURE**:
- `gc.<module>.<type>` - Types/Enums
- `gc.<module>.<function>` - API function namespaces
- `gc.<function|type>` - Top-level convenience re-exports (not recommended)

### 3. HTTP API Call

#### 3.1 GCB Format

**CRITICAL**: All calls use **POST** with GCB binary payload.

**Format**: `POST /<module>::<function>` (module function) or `POST /<module>::<type>::<function>` (type method)

```bash
curl -X POST http://localhost:8080/country_api::getStats -H "Content-Type: application/octet-stream" --data-binary @payload.gcb
```

> By default the SDK (de)serializes GCB.

#### 3.2 JSON Format

**CRITICAL**: All calls use **POST** with parameters as **JSON array**.

**Format**: `POST /<module>::<function>` (module function) or `POST /<module>::<type>::<function>` (type method)

```bash
# Single parameter
curl -X POST http://localhost:8080/country_api::getStats -H "Content-Type: application/json" -d '["France"]'

# Multiple parameters
curl -X POST http://localhost:8080/country_api::listCountries -H "Content-Type: application/json" -d '[0, 20]'

# No parameters
curl -X POST http://localhost:8080/someApi::noParams -H "Content-Type: application/json" -d '[]'
```

### 4. TypeScript/JavaScript Usage

```typescript
// Direct usage
const stats = await gc.country_api.getStats("France");
const countries = await gc.country_api.listCountries(0, 20);

// TypeScript catches errors at compile time
// gc.country_api.getStats(123);      // Error: Expected string
// gc.country_api.getStats("France"); // Correct
```

**Naming Convention**: Backend snake_case (`get_city_by_id`) -> Frontend camelCase (`getCityById`).

## Return Types

### Simple Array
```gcl
@expose
fn listCountries(offset: int, maxCount: int): Array<CountryView> { /* ... */ }
```

### PaginatedResult
```gcl
@expose
fn getDocumentsByYear(year: int, yearType: YearType, filters: DocumentFilters?, offset: int, maxResults: int): PaginatedResult<DocumentView> {
    return PaginatedResult<DocumentView> {
        items: Array<DocumentView> {},
        offset: offset,
        limit: maxResults,
        total: totalCount,
        hasMore: (offset + maxResults) < totalCount,
        page: (offset / maxResults) + 1,
        totalPages: (totalCount + maxResults - 1) / maxResults,
    };
}
```

**Frontend:**
```typescript
const result = await gc.getDocumentsByYear(2020, gc.vocabulary.YearType.case, null, 0, 20);
console.log(result.items, result.total, result.hasMore, result.page, result.totalPages);
```

## Handling GreyCat Enums

**Enum Serialization**: Use `.key!` to serialize enums.

```typescript
// CORRECT
const data = { role: user.role.key! };  // "Admin" as string

// INCORRECT
const data = { role: user.role };  // Sends full object

// Display
const roleText = user.role.key;

// Map keys
const countsByRole = new Map<string, number>();
for (const user of users) { countsByRole.set(user.role.key!, (countsByRole.get(user.role.key!) ?? 0) + 1); }
```

## Authentication

**Backend:**
```gcl
type Person { email: String; firstName: String; lastName: String; userId: int; }
var persons_by_id: nodeIndex<int, node<Person>>;

abstract type PersonService {
    static fn create(email: String, firstName: String, password: String): node<Person> {
        var userId = UserGroup::Default.add(UserRole::Admin, email, password);
        var person = node<Person>{ Person { email, firstName, lastName: "", userId } };
        persons_by_id.set(userId, person);
        return person;
    }
}

@volatile type PersonView { email: String; firstName: String; lastName: String; }
@expose fn getCurrentPerson(): PersonView? {
    var user = User::current(); if (user == null) { return null; }
    var person = persons_by_id.get(user.id); if (person == null) { return null; }
    return PersonView { email: person->email, firstName: person->firstName, lastName: person->lastName };
}
```

**Frontend Auth Service:**
```typescript
// src/services/auth.ts
export const authService = {
  login: async (username: string, password: string) => {
    await gc.sdk.login({ username, password, use_cookie: true });
    return await gc.getCurrentPerson();
  },
  logout: async () => { await gc.sdk.logout(); },
  getCurrentUser: async () => await gc.getCurrentPerson(),
  isAuthenticated: () => gc.sdk.token !== null,
};
```

## Service Layer Pattern

**Recommended pattern**: Create service layer wrapping gc calls for retry logic and error handling.

**src/services/apiUtils.ts:**
```typescript
export interface RetryOptions {
  maxRetries?: number;
  delayMs?: number;
  exponentialBackoff?: boolean;
}

export async function withRetry<T>(fn: () => Promise<T>, options: RetryOptions = {}): Promise<T> {
  const { maxRetries = 3, delayMs = 1000, exponentialBackoff = true } = options;
  let lastError: unknown;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      if (attempt === maxRetries) break;

      // Don't retry auth errors
      const statusCode = typeof error === 'object' && error !== null && 'status' in error
        ? (error as { status?: number }).status : undefined;
      if (statusCode === 401 || statusCode === 403) throw error;

      const delay = exponentialBackoff ? delayMs * Math.pow(2, attempt) : delayMs;
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  throw lastError;
}
```

**src/services/countryService.ts:**
```typescript
import { withRetry } from './apiUtils';

export const CountryService = {
  getStats: (name: string): Promise<gc.api_types.StatsView> =>
    withRetry(() => gc.country_api.getStats(name)),

  listCountries: (offset: number, maxCount: number): Promise<Array<gc.api_types.CountryView>> =>
    withRetry(() => gc.country_api.listCountries(offset, maxCount)),
};
```

## Complex Example with Enums

**Backend:**
```gcl
enum YearType { case, judgment, decision, published, filed }

@volatile type DocumentFilters { formexTypes: Array<FormexType>?; }

@expose
fn getDocumentsByYear(year: int, yearType: YearType, filters: DocumentFilters?, offset: int, maxResults: int): PaginatedResult<DocumentView> { /* ... */ }
```

**Frontend Service:**
```typescript
export const StatsService = {
  getDocumentsByYear: (
    year: number,
    yearType?: gc.vocabulary.YearType | null,
    offset = 0,
    maxResults = 20
  ): Promise<gc.pagination_service.PaginatedResult> =>
    withRetry(() =>
      gc.getDocumentsByYear(
        year,
        yearType ?? gc.vocabulary.YearType.case,
        null,
        offset,
        maxResults
      )
    ),
};
```

## Error Handling

```typescript
try {
  const city = await gc.country_api.getCity(cityId);
  // use city
} catch (error) {
  console.error('Failed to fetch city:', error);
  // handle error in your UI framework
}
```

## Best Practices

1. **Always run `greycat codegen ts`** after backend changes
2. **Initialize SDK before app renders**: `await gc.sdk.init()`
3. **Use top-level `gc` namespace**: Prefer fqn `gc.moduleName.fnName()` over shortcut `gc.fnName()`
5. **Service layer**: Wrap gc calls for retry logic, error handling
6. **Type safety**: Leverage generated types, avoid `any`
7. **Environment variables**: Use `VITE_GREYCAT_URL` for backend URL

## Common Pitfalls

| Wrong | Correct |
|-------|---------|
| `gc.getStats()` | `gc.project.getStats()` |
| App renders before SDK init | `gc.sdk.init().then(() => import('./main'))` |
| Forgot `greycat codegen ts` | Run after every backend changes |
| Direct gc calls everywhere | Use service layer with retry |
| `new Map<Enum, V>()` | `new Map<string, V>()` with `.key!` |
| Missing enum defaults | Use `?? gc.vocabulary.EnumName.default` |

## Integration Checklist

- [ ] Install @greycat/web SDK
- [ ] Configure Vite plugin + tsconfig.json
- [ ] Add vite-env.d.ts reference types
- [ ] Initialize SDK before app renders
- [ ] Run `greycat codegen ts` after backend changes
- [ ] Update package.json scripts
- [ ] Implement auth service
- [ ] Create service layer for API calls
- [ ] Serialize enums with `.key!`
- [ ] Test type safety: No `any` types
- [ ] Verify error handling
