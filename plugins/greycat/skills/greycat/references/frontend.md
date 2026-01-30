# Frontend Integration with React

React + GreyCat integration using `@greycat/web` SDK.

## Overview

**@greycat/web SDK provides**: Auto-generated TypeScript types, auth/session management, typed API communication, error handling.

**Flow**: `Backend (GCL) → greycat codegen ts → project.d.ts → Frontend (TS/React) → Global gc namespace`

## Installation & Setup

```bash
# Install SDK (check https://get.greycat.io for latest)
npm install https://get.greycat.io/files/sdk/web/dev/7.6/7.6.10-dev.tgz
```

**Dependencies**: `@greycat/web`, `react@^18.3`, `react-dom@^18.3`, `@tanstack/react-query@^5`, `react-router-dom@^6`

**vite.config.ts:**
```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import greycat from '@greycat/web/vite-plugin';

export default defineConfig(() => ({
  plugins: [react(), greycat({ greycat: process.env.VITE_GREYCAT_URL || 'http://127.0.0.1:8080' })],
  server: { port: 3000 },
}));
```

**tsconfig.json:**
```json
{
  "compilerOptions": { "target": "ES2020", "lib": ["ES2020", "DOM", "DOM.Iterable"], "module": "ESNext", "moduleResolution": "bundler", "jsx": "react-jsx", "strict": true },
  "include": ["src", "../project.d.ts"]
}
```

**src/vite-env.d.ts:**
```typescript
/// <reference types="vite/client" />
/// <reference types="@greycat/web/sdk" />
```

## SDK Initialization

**⚠️ CRITICAL**: SDK must initialize BEFORE React renders.

**index.tsx:**
```typescript
import '@greycat/web';

gc.sdk.init({
  numFmt: new Intl.NumberFormat('en-GB', { notation: 'compact', maximumSignificantDigits: 5 }),
}).then(() => import('./main'));
```

**main.tsx:**
```typescript
import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from './App';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 60_000, gcTime: 5 * 60 * 1000, refetchOnWindowFocus: true, retry: 1 }
  }
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}><App /></QueryClientProvider>
  </React.StrictMode>
);
```

## Backend to Frontend Workflow

### 1. Backend API Definition

**backend/src/api/country_api.gcl:**
```gcl
@expose @permission("public")
fn getStats(name: String): StatsView { /* ... */ }

@expose @permission("public")
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

**⚠️ NAMESPACE STRUCTURE**:
- `gc.project.*` - Database structure (Root, indices)
- `gc.<api_name>.*` - API function namespaces
- `gc.<function>()` - Top-level convenience exports (recommended)

### 3. HTTP API Call Format

**⚠️ CRITICAL**: All calls use **POST** with parameters as **JSON array**.

**Format**: `POST /api/<namespace>::<function_name>`

```bash
# Single parameter
curl -X POST http://localhost:8080/api/country_api::getStats -H "Content-Type: application/json" -d '["France"]'

# Multiple parameters
curl -X POST http://localhost:8080/api/country_api::listCountries -H "Content-Type: application/json" -d '[0, 20]'

# No parameters
curl -X POST http://localhost:8080/api/someApi::noParams -H "Content-Type: application/json" -d '[]'
```

### 4. TypeScript Usage

```typescript
// Direct usage - types are global (no import needed)
const stats: gc.api_types.StatsView = await gc.getStats("France");
const countries: Array<gc.api_types.CountryView> = await gc.listCountries(0, 20);

// TypeScript catches errors at compile time
// ❌ gc.getStats(123);           // Error: Expected string
// ✅ gc.getStats("France");       // Correct
```

**Naming Convention**: Backend snake_case (`get_city_by_id`) → Frontend camelCase (`getCityById`).

## Return Types

### Simple Array
```gcl
@expose @permission("public")
fn listCountries(offset: int, maxCount: int): Array<CountryView> { /* ... */ }
```

### PaginatedResult
```gcl
@expose @permission("public")
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
// ✅ CORRECT
const data = { role: user.role.key! };  // "Admin" as string

// ❌ INCORRECT
const data = { role: user.role };  // Sends full object

// Display
<div>Role: {user.role.key}</div>

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
@expose @permission("public") fn getCurrentPerson(): PersonView? {
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

// src/hooks/usePerson.ts
export function usePerson() {
  return useQuery({
    queryKey: ['currentPerson'],
    queryFn: authService.getCurrentUser,
    staleTime: 5 * 60 * 1000,
    enabled: authService.isAuthenticated(),
  });
}
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
    withRetry(() => gc.getStats(name)),

  listCountries: (offset: number, maxCount: number): Promise<Array<gc.api_types.CountryView>> =>
    withRetry(() => gc.listCountries(offset, maxCount)),
};
```

## React Component Usage

```typescript
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { CountryService } from '../services/countryService';
import { useState } from 'react';

export function CountryPage() {
  const [page, setPage] = useState(0);
  const pageSize = 20;

  // Query
  const { data: stats, isLoading, error } = useQuery({
    queryKey: ['countryStats', "France"],
    queryFn: () => CountryService.getStats("France"),
  });

  // List query with pagination
  const { data: countries } = useQuery({
    queryKey: ['countries', page, pageSize],
    queryFn: () => CountryService.listCountries(page * pageSize, pageSize),
  });

  // Mutation
  const queryClient = useQueryClient();
  const createCity = useMutation({
    mutationFn: (params: { name: string; countryId: number }) => gc.createCity(params.name, params.countryId),
    onSuccess: (newCity) => {
      queryClient.invalidateQueries({ queryKey: ['cities', newCity.countryId] });
      queryClient.setQueryData(['city', newCity.id], newCity);
    },
  });

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <div>
      <h1>France</h1>
      <p>Population: {stats?.population}</p>

      <ul>{countries?.map(c => <li key={c.id}>{c.name}</li>)}</ul>
      <button onClick={() => setPage(p => p - 1)} disabled={page === 0}>Previous</button>
      <button onClick={() => setPage(p => p + 1)} disabled={!countries || countries.length < pageSize}>Next</button>

      <button onClick={() => createCity.mutate({ name: "Paris", countryId: 1 })}>Add City</button>
    </div>
  );
}
```

## Complex Example with Enums

**Backend:**
```gcl
enum YearType { case, judgment, decision, published, filed }

@volatile type DocumentFilters { formexTypes: Array<FormexType>?; }

@expose @permission("public")
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

**Frontend Component:**
```typescript
export default function StatsPage() {
  const [yearType, setYearType] = useState<gc.vocabulary.YearType>(gc.vocabulary.YearType.case);
  const [selectedYear, setSelectedYear] = useState<number | null>(null);

  const { data: yearDocuments } = useQuery({
    queryKey: ['documentsByYear', selectedYear, yearType.key],
    queryFn: () => StatsService.getDocumentsByYear(selectedYear!, yearType, 0, 50),
    enabled: selectedYear !== null,
  });

  return (
    <div>
      <select value={yearType.key} onChange={(e) => {
        const key = e.target.value as gc.vocabulary.YearType.Field;
        setYearType(gc.vocabulary.YearType[key]);
        setSelectedYear(null);
      }}>
        <option value="case">Case Year</option>
        <option value="judgment">Judgment Date</option>
      </select>

      {yearDocuments && (
        <div>
          <p>Total: {yearDocuments.total} documents</p>
          <p>Page {yearDocuments.page} of {yearDocuments.totalPages}</p>
          <ul>{yearDocuments.items.map((doc: any) => <li key={doc.id}>{doc.celex}</li>)}</ul>
        </div>
      )}
    </div>
  );
}
```

## Error Handling

**Try-Catch:**
```typescript
try {
  const city = await gc.getCity(cityId);
  setCity(city);
} catch (error) {
  console.error('Failed to fetch city:', error);
  toast.error(error instanceof Error ? error.message : 'Unknown error');
}
```

**React Query:**
```typescript
const { data, error, isError } = useQuery({
  queryKey: ['city', cityId],
  queryFn: () => gc.getCity(cityId),
  retry: (failureCount, error) => {
    if (error.message?.includes('404')) return false;
    return failureCount < 3;
  },
});

if (isError) return <div>Error: {error.message}</div>;
```

## Time Handling

GreyCat `time` type (μs epoch) needs conversion:

```typescript
// Backend
@volatile type EventView { timestamp: time; }

// Frontend
const event = await gc.getEvents()[0];
const date = new Date(event.timestamp / 1000);  // Convert μs → ms

// Display component
function EventTime({ timestamp }: { timestamp: number }) {
  const date = new Date(timestamp / 1000);
  return <time dateTime={date.toISOString()}>{date.toLocaleString()}</time>;
}
```

## Best Practices

1. **Always run `greycat codegen ts`** after backend changes
2. **Initialize SDK before React**: `gc.sdk.init().then(() => import('./main'))`
3. **Use top-level `gc` namespace**: Prefer `gc.getStats()` over `gc.stats_api.getStats()`
4. **Serialize enums**: Use `.key!` when sending to backend/APIs
5. **Service layer**: Wrap gc calls for retry logic, error handling
6. **React Query**: Use for caching, auto-refetching, optimistic updates
7. **Type safety**: Leverage generated types, avoid `any`
8. **Environment variables**: Use `VITE_GREYCAT_URL` for backend URL

## Common Pitfalls

| ❌ Wrong | ✅ Correct |
|----------|-----------|
| `gc.project.getStats()` | `gc.getStats()` (APIs not in gc.project) |
| `{ role: user.role }` | `{ role: user.role.key! }` |
| React renders before SDK init | `gc.sdk.init().then(() => import('./main'))` |
| Forgot `greycat codegen ts` | Run after every backend change |
| Direct gc calls in components | Use service layer + React Query |
| `new Map<Enum, V>()` | `new Map<string, V>()` with `.key!` |
| Missing enum defaults | Use `?? gc.vocabulary.EnumName.default` |

## Integration Checklist

- [ ] Install @greycat/web SDK
- [ ] Configure Vite plugin + tsconfig.json
- [ ] Add vite-env.d.ts reference types
- [ ] Initialize SDK before React
- [ ] Run `greycat codegen ts` after backend changes
- [ ] Update package.json scripts
- [ ] Implement auth service + usePerson hook
- [ ] Create service layer for API calls
- [ ] Setup React Query with QueryClientProvider
- [ ] Serialize enums with `.key!`
- [ ] Test type safety: No `any` types
- [ ] Verify error handling
- [ ] Convert time (μs → ms) for display
