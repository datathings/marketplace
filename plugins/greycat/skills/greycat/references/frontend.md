# Frontend Integration with React

This section covers integrating React frontends with GreyCat backends using the official `@greycat/web` SDK.

## Table of Contents

- [Overview](#overview)
- [Installation & Setup](#installation--setup)
- [SDK Initialization](#sdk-initialization)
- [TypeScript Type Generation](#typescript-type-generation)
- [API Communication Patterns](#api-communication-patterns)
- [Handling GreyCat Enums](#handling-greycat-enums)
- [Authentication & Authorization](#authentication--authorization)
- [Service Layer Architecture](#service-layer-architecture)
- [React Query Integration](#react-query-integration)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)
- [Common Pitfalls](#common-pitfalls)
- [Integration Checklist](#integration-checklist)
- [Time Usage in Frontend](#time-usage-in-frontend)

## Overview

GreyCat provides an official frontend SDK (`@greycat/web`) that:
- Auto-generates TypeScript types from your GCL backend
- Provides built-in authentication and session management
- Handles API communication with full type safety
- Includes error handling utilities
- Works seamlessly with React and modern bundlers

**Architecture:**
```
Backend (GCL) → greycat codegen ts → project.d.ts → Frontend (TypeScript/React)
                                    ↓
                            Global `gc` namespace with full type safety
```

## Installation & Setup

### Install Dependencies

```bash
npm install https://get.greycat.io/files/sdk/web/dev/7.5/7.5.13-dev.tgz
pnpm add https://get.greycat.io/files/sdk/web/dev/7.5/7.5.13-dev.tgz
```

**Core Dependencies:**
```json
{
  "dependencies": {
    "@greycat/web": "https://get.greycat.io/files/sdk/web/dev/7.5/7.5.13-dev.tgz",
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "@tanstack/react-query": "^5.0.0",
    "react-router-dom": "^6.0.0"
  }
}
```

### Configure Vite (vite.config.ts)

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import greycat from '@greycat/web/vite-plugin';

export default defineConfig({
  plugins: [
    react(),
    greycat({
      greycat: 'http://127.0.0.1:8080'  // Your GreyCat backend URL
    })
  ],
  server: { port: 3000 },
});
```

### TypeScript Configuration (tsconfig.json)

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true
  },
  "include": ["src", "../project.d.ts"]
}
```

### Vite Environment Types (src/vite-env.d.ts)

```typescript
/// <reference types="vite/client" />
/// <reference types="@greycat/web/sdk" />
```

## SDK Initialization

### Entry Point Pattern (index.tsx)

**Critical: SDK must initialize BEFORE React renders**

```typescript
import '@greycat/web';

gc.sdk.init().then(() => {
  import('./main');
});
```

### Main App Entry (main.tsx)

```typescript
import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from './App';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000,
      gcTime: 5 * 60 * 1000,
      refetchOnWindowFocus: true,
      retry: 1,
    },
  },
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>
);
```

## TypeScript Type Generation

Generate types from your GCL backend:

```bash
greycat codegen ts
```

This generates `project.d.ts` with:
- All GCL types converted to TypeScript
- Enum definitions
- `@volatile` API response types
- Full type safety for the GreyCat API

**When to Regenerate:**
- After changing GCL type definitions
- After adding new @expose functions
- After modifying @volatile response types
- Before building for production

**Add to package.json:**
```json
{
  "scripts": {
    "types": "cd .. && greycat codegen ts",
    "dev": "npm run types && vite",
    "build": "npm run types && tsc && vite build"
  }
}
```

## API Communication Patterns

### ⚠️ CRITICAL: Correct API Namespace Structure

After running `greycat codegen ts`, @expose functions are exported at the **top-level `gc` namespace**, NOT under `gc.project`.

```typescript
// ✅ CORRECT - Functions are at top level
const city = await gc.getCity(cityId);
const results = await gc.searchCities(query, 20, options);
const stats = await gc.getStats();

// ❌ INCORRECT - Will fail
const city = await gc.project.getCity(cityId);  // ERROR!
```

### Direct gc Namespace Access

```typescript
// Auth and runtime APIs
await gc.runtime.User.current();
await gc.sdk.login({ username, password, use_cookie: true });
await gc.sdk.logout();

// Your @expose functions (top-level gc namespace)
const city = await gc.getCity(cityId);
const building = await gc.getBuilding(buildingId);
const results = await gc.searchCities(query, 20, options);
const buildings = await gc.listBuildings(offset, limit);

// Type-safe with full IntelliSense
const city: gc.geo_api.CityView | null = await gc.getCity(id);
```

### Naming Conventions

| Backend (GCL) | Frontend (TypeScript) | Notes |
|---------------|----------------------|-------|
| `@expose fn getCity()` | `gc.getCity()` | Top-level export |
| `@expose fn getBuilding()` | `gc.getBuilding()` | Top-level export |
| `type City { ... }` | `gc.geo.City` | Type namespace |
| `type Building { ... }` | `gc.geo.Building` | Type namespace |
| `@volatile type BuildingView` | `gc.geo_api.BuildingView` | Response type |
| `enum BuildingType` | `gc.geo.BuildingType` | Enum namespace |

## Handling GreyCat Enums

### Enum Serialization

GreyCat enums serialize to JSON as objects with `{offset: number, key: string}` structure.

**Backend GCL:**
```gcl
enum BuildingType {
  "Residential",
  "Commercial",
  "Industrial"
}

@volatile
type BuildingView {
  address: String;
  street: String;
  city: String;
  buildingType: BuildingType;  // Send as enum, not String
  constructionDate: time;
  residents: int;
}
```

**JSON Response:**
```json
{
  "buildingType": {
    "offset": 1,
    "key": "Commercial"
  }
}
```

### Displaying Enums in React

**CRITICAL**: Cannot render enum objects directly - use `.key` property.

```typescript
// ❌ Wrong - causes React error
<span>{building.buildingType}</span>

// ✅ Correct - use .key
<span>{building.buildingType.key}</span>  // "Commercial"
```

### Using Enums as Map Keys

**Backend Pattern (GCL):**
```gcl
@volatile
type CityStatsView {
    buildingsByType: Map<BuildingType, int>;
    totalResidents: int;
}

fn getCityStats(cityId: String): CityStatsView {
    var buildingsByType = Map<BuildingType, int> {};
    var totalResidents = 0;

    for (building_node in buildings_in_city) {
        var building = building_node.resolve();
        var typeKey = building.buildingType;
        buildingsByType.set(typeKey, (buildingsByType.get(typeKey) ?? 0) + 1);
        totalResidents += building.residents.size();
    }

    return CityStatsView {
        buildingsByType: buildingsByType,
        totalResidents: totalResidents
    };
}
```

**Frontend Pattern (TypeScript):**
```typescript
// IMPORTANT: GreyCat Maps are Map objects - use Array.from(map.entries())

{stats.buildingsByType && Array.from(stats.buildingsByType.entries()).map(([type, count]) => (
  <div key={type.key}>
    <div>{count}</div>
    <div>{type.key}</div>  {/* Extract string with .key */}
  </div>
))}

// ❌ WRONG - Object.entries() doesn't work on GreyCat Maps
Object.entries(stats.buildingsByType)  // Won't iterate
```

## Authentication & Authorization

### Person Type Definition

Define your application-level user type with GreyCat runtime integration:

```gcl
type Person {
    username: String;        // for authentication
    email: String;
    gcId: int;              // GreyCat runtime user ID
    name: String;           // display name
    birthDate: time;
    residence: node<Building>?;  // link to building
    role: PersonRole;       // Owner, Resident, Tenant
    createdAt: time;
    lastLogin: time?;
}

enum PersonRole {
    "Owner",
    "Resident",
    "Tenant"
}
```

### User Creation Backend Pattern

**⚠️ CRITICAL: SecurityEntity::set() is REQUIRED**

```gcl
fn createPerson(username: String, email: String, password: String, role: PersonRole): node<Person> {
    var username = username.trim().lowercase();
    var email = email.trim().lowercase();

    // Step 1: Hash password
    var passwordHash = Crypto::sha256hex(password);

    // Step 2: Create GreyCat runtime User
    var runtimeUser = User {
        id: 0,
        name: username,
        activated: true,
        external: false,
        role: type::enum_name(role),
    };

    // Step 3: ✅ CRITICAL - Register in security system
    var gcId = SecurityEntity::set(runtimeUser);

    if (gcId != null) {
        // Step 4: Set password
        User::setPassword(username, passwordHash);
    } else {
        throw "Failed to create user in security system";
    }

    // Step 5: Create Person with domain fields
    var person = node<Person>{
        Person {
            username: username,
            email: email,
            gcId: gcId,
            name: username,  // can be updated later
            birthDate: time::now(),  // placeholder
            residence: null,  // set when person moves into building
            role: role,
            createdAt: time::now(),
            lastLogin: null,
        }
    };

    // Step 6: Index for lookups
    persons_by_username.set(username, person);
    persons_by_gc_id.set(gcId, person);

    return person;
}
```

### PersonView Type

API response type for frontend consumption:

```gcl
@volatile
type PersonView {
    username: String;
    email: String;
    name: String;
    role: PersonRole;
    residence: BuildingView?;  // show building info
    createdAt: time;
    lastLogin: time?;
}
```

### Get Current Person API

```gcl
@expose
@permission("app.user")
fn getCurrentPerson(): PersonView? {
    var gcId = runtime::User::current();
    if (gcId == null || gcId == 0) {
        return null;
    }

    var person = persons_by_gc_id.get(gcId);
    if (person == null) {
        return null;
    }

    var residenceView: BuildingView? = null;
    if (person->residence != null) {
        var building = person->residence.resolve();
        residenceView = BuildingView {
            address: building->address,
            street: "...",  // lookup from street
            city: "...",    // lookup from city
            buildingType: building->buildingType,
            constructionDate: building->constructionDate,
            residents: building->residents.size()
        };
    }

    return PersonView {
        username: person->username,
        email: person->email,
        name: person->name,
        role: person->role,
        residence: residenceView,
        createdAt: person->createdAt,
        lastLogin: person->lastLogin
    };
}
```

### Frontend Auth Service

```typescript
export const AuthService = {
  getCurrentUserId: (): Promise<number | bigint | null> => {
    return gc.runtime.User.current();
  },

  isLoggedIn: async (): Promise<boolean> => {
    const userId = await AuthService.getCurrentUserId();
    return userId !== null && userId !== undefined && userId !== 0;
  },

  getCurrentPerson: async () => {
    const userId = await AuthService.getCurrentUserId();

    if (userId === null || userId === undefined || userId === 0) {
      return null;
    }

    return gc.getCurrentPerson();
  },

  login: (username: string, password: string): Promise<string> => {
    return gc.sdk.login({
      username,
      password,
      use_cookie: true
    });
  },

  logout: async (): Promise<void> => {
    try {
      await gc.sdk.logout();
    } catch (err) {
      if (String(err).includes('401')) return;
      throw err;
    }
  },
};
```

### Person Hook with React Query

```typescript
import { useQuery, useQueryClient } from '@tanstack/react-query';

export function usePerson() {
  return useQuery<Person | null>({
    queryKey: ['person'],
    queryFn: async () => {
      const userId = await AuthService.getCurrentUserId();

      if (userId === null || userId === undefined || userId === 0) {
        return null;
      }

      return await AuthService.getCurrentPerson();
    },
    staleTime: 60000,
    retry: false,
  });
}

export function useLogout() {
  const queryClient = useQueryClient();

  return async () => {
    await AuthService.logout();
    await queryClient.cancelQueries({ queryKey: ['person'] });
    queryClient.removeQueries({ queryKey: ['person'] });
  };
}
```

## Service Layer Architecture

Organize API calls into service modules:

```
src/
├── services/
│   ├── authService.ts
│   ├── geoService.ts
│   └── apiUtils.ts
├── hooks/
│   ├── usePerson.ts
│   └── useCities.ts
└── queries/
    └── queryKeys.ts
```

### Geographic Service Example

```typescript
import { withRetry } from './apiUtils';

export const GeoService = {
  getCity: (id: string) => withRetry(() => gc.getCity(id)),
  getBuilding: (id: string) => withRetry(() => gc.getBuilding(id)),
  listCities: (offset: number, limit: number) =>
    withRetry(() => gc.listCities(offset, limit)),
  listBuildings: (offset: number, limit: number) =>
    withRetry(() => gc.listBuildings(offset, limit)),
};
```

### Retry Utility

```typescript
export async function withRetry<T>(
  fn: () => Promise<T>,
  options = { maxRetries: 3, delayMs: 1000 }
): Promise<T> {
  let lastError: unknown;

  for (let attempt = 0; attempt <= options.maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      // Don't retry on auth errors
      if (String(error).includes('401') || String(error).includes('403')) {
        throw error;
      }

      if (attempt < options.maxRetries) {
        await new Promise(resolve =>
          setTimeout(resolve, options.delayMs * Math.pow(2, attempt))
        );
      }
    }
  }

  throw lastError;
}
```

## React Query Integration

### Query Hook Example

```typescript
import { useQuery } from '@tanstack/react-query';

export function useCity(id: string) {
  return useQuery({
    queryKey: ['city', id],
    queryFn: () => gc.getCity(id),
    enabled: !!id,
    staleTime: 5 * 60 * 1000,
    retry: 2,
  });
}

export function useBuilding(id: string) {
  return useQuery({
    queryKey: ['building', id],
    queryFn: () => gc.getBuilding(id),
    enabled: !!id,
    staleTime: 5 * 60 * 1000,
    retry: 2,
  });
}

export function useSearchCities(query: string) {
  return useQuery({
    queryKey: ['search', 'cities', query],
    queryFn: () => gc.searchCities(query, 20),
    enabled: query.length > 0,
    staleTime: 300,
    retry: 3,
  });
}

export function useSearchBuildings(query: string) {
  return useQuery({
    queryKey: ['search', 'buildings', query],
    queryFn: () => gc.searchBuildings(query, 20),
    enabled: query.length > 0,
    staleTime: 300,
    retry: 3,
  });
}
```

### Mutation Hook Example

```typescript
import { useMutation, useQueryClient } from '@tanstack/react-query';

export function useDeleteCity() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (cityId: string) => gc.deleteCity(cityId),

    onMutate: async (cityId) => {
      await queryClient.cancelQueries({ queryKey: ['cities'] });
      const previous = queryClient.getQueryData(['cities']);

      queryClient.setQueryData(['cities'], (old: any) =>
        old?.filter((city: any) => city.id !== cityId)
      );

      return { previous };
    },

    onError: (err, cityId, context) => {
      if (context?.previous) {
        queryClient.setQueryData(['cities'], context.previous);
      }
    },

    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['cities'] });
    },
  });
}
```

## Error Handling

### Error Utilities

```typescript
export function formatError(err: unknown, fallback: string): string {
  return gc.sdk.prettyError(err, fallback);
}

export function isAuthError(err: unknown): boolean {
  const msg = String(err).toLowerCase();
  return msg.includes('401') || msg.includes('403') ||
         msg.includes('unauthorized');
}
```

### Toast Notifications

```bash
npm install react-hot-toast
```

```typescript
import toast from 'react-hot-toast';

export const AppToast = {
  success: (message: string) => toast.success(message),
  error: (message: string) => toast.error(message),
};

// Usage
try {
  await GeoService.deleteCity(id);
  AppToast.success('City deleted');
} catch (err) {
  AppToast.error(formatError(err, 'Failed to delete'));
}
```

## Best Practices

### Type Safety

```typescript
// ✅ Good
const city: gc.geo_api.CityView | null = await gc.getCity(id);

// ❌ Bad
const city = await gc.getCity(id) as any;
```

### Error Handling

```typescript
// ✅ Good
try {
  const result = await GeoService.search(query);
  AppToast.success(`Found ${result.total} items`);
} catch (err) {
  AppToast.error(formatError(err, 'Search failed'));
}

// ❌ Bad - swallow errors
try {
  await GeoService.search(query);
} catch {}
```

### Loading States

```typescript
// ✅ Good
const { data, isLoading, error } = useCity(id);

if (isLoading) return <Spinner />;
if (error) return <ErrorMessage error={error} />;
if (!data) return <NotFound />;

return <CityView city={data} />;

// ❌ Bad - no feedback
const { data } = useCity(id);
return <CityView city={data} />;  // Crashes if undefined
```

## Common Pitfalls

### Forgetting SDK Initialization

```typescript
// ❌ Bad
import ReactDOM from 'react-dom/client';
ReactDOM.createRoot(root).render(<App />);

// ✅ Good
import '@greycat/web';
gc.sdk.init().then(() => import('./main'));
```

### Not Regenerating Types

```bash
# After changing GCL code:
greycat codegen ts
```

### Storing Server State in Context

```typescript
// ❌ Bad - loses React Query benefits
const [person, setPerson] = useState(null);
useEffect(() => {
  fetchPerson().then(setPerson);
}, []);

// ✅ Good
const { data: person } = usePerson();
```

### Not Handling Null/Undefined

```typescript
// ❌ Bad
const name = city.name;

// ✅ Good
const name = city?.name ?? 'Unknown';
```

### Using Wrong Namespace

```typescript
// ❌ Wrong
await gc.project.getCity(id);  // ERROR

// ✅ Correct
await gc.getCity(id);
```

## Integration Checklist

- [ ] Install `@greycat/web` package
- [ ] Configure Vite with GreyCat plugin
- [ ] Add `@greycat/web/sdk` to vite-env.d.ts
- [ ] Create index.tsx with SDK initialization
- [ ] Run `greycat codegen ts` to generate types
- [ ] Create service layer (authService, etc.)
- [ ] Set up React Query
- [ ] Create auth hooks (usePerson, useLogout)
- [ ] Implement login component
- [ ] Set up error handling
- [ ] Test authentication flow


## Frontend Development Best Practices

### Layout and Responsiveness

- **Mobile-first**: Start with small screens, add breakpoints for larger screens
- **Fluid layouts**: Use flexbox/grid, avoid hard-coded widths
- **Responsive units**: Prefer `rem`, `em`, `clamp()` for scalable text/spacing
- **Content width**: Limit max width for readability (`max-w-6xl`)

```tsx
<main className="mx-auto w-full max-w-6xl px-4 sm:px-6 lg:px-8">
  <section className="grid gap-6 md:grid-cols-[1fr_2fr]">
    <aside className="rounded-lg border p-4">Sidebar</aside>
    <div className="rounded-lg border p-4">Content</div>
  </section>
</main>
```

### Dark and Light Mode

- **Theme via CSS variables**: Keep colors in variables
- **Use `prefers-color-scheme`**: System default + user override
- **Tailwind dark mode**: Configure `darkMode: 'class'` in config

```css
:root {
  --bg: #ffffff;
  --fg: #101114;
  --accent: #1e6ce0;
}

:root[data-theme='dark'] {
  --bg: #0f1115;
  --fg: #f2f4f8;
  --accent: #7fb4ff;
}
```

### Typography and Spacing

- **Use a scale**: Define type and spacing scales in CSS variables
- **Line length**: Aim for 60-80 characters for paragraphs
- **Vertical rhythm**: Use consistent spacing steps (multiples of 4px)

```css
:root {
  --space-1: 0.25rem;  /* 4px */
  --space-2: 0.5rem;   /* 8px */
  --space-3: 0.75rem;  /* 12px */
  --space-4: 1rem;     /* 16px */
  --space-6: 1.5rem;   /* 24px */
  --space-8: 2rem;     /* 32px */
}
```

### Component Structure

- **Reuse patterns**: Build consistent components with Tailwind utilities
- **Prefer composition**: Simple components that compose well
- **Centralized icons**: Use `lucide-react`, export from single module

```tsx
// icons/Icon.tsx
import { Bell, ChevronRight, type LucideProps } from 'lucide-react';

export const Icons = {
  bell: (props: LucideProps) => <Bell {...props} className="h-4 w-4" />,
  chevronRight: (props: LucideProps) => <ChevronRight {...props} className="h-4 w-4" />,
};

// Usage
<Icons.bell className="h-5 w-5 text-slate-500" />
```

### Performance

- **Code split**: Use `React.lazy` for routes/heavy components
- **Query caching**: TanStack Query dedupes and caches API calls
- **Images**: Use modern formats (AVIF, WebP) with correct dimensions
- **Avoid layout shift**: Reserve space for images and dynamic content
- **Minimize re-renders**: Memoize where appropriate

**Gzip Compression:**

```typescript
// vite.config.ts
import compression from 'vite-plugin-compression';

export default defineConfig({
  plugins: [
    compression({ algorithm: 'gzip' }),
  ],
});
```

Ensure hosting server serves pre-compressed `.gz` assets with `Content-Encoding: gzip`.

### ESLint and Type Safety

- Keep `pnpm run lint` green
- Use `@typescript-eslint/parser` with React hooks rules
- Avoid `any` in shared utilities

### Testing and QA

- **Responsive testing**: Validate on multiple viewport sizes
- **Color modes**: Check both light and dark themes
- **Accessibility**: Use keyboard navigation, check contrast ratios (WCAG AA)
- **Cross-browser**: Test on Chrome, Firefox, Safari

### Suggested Defaults

- **Global container width**: `max-w-6xl` (1200px)
- **Base font size**: `16px`
- **Spacing scale**: Multiples of `4px` (0.25rem)
- **Colors**: Define in CSS variables for easy theming
- **Breakpoints**: `sm: 640px`, `md: 768px`, `lg: 1024px`, `xl: 1280px`


## Time Usage in Frontend

The frontend receives GreyCat time values as `gc.core.time`, not JavaScript `Date`. Use the GreyCat SDK helpers to convert, parse, and format times consistently.

### Key Concepts

- `gc.core.time` stores epoch in microseconds
- `gc.core.time` has helpers to convert to `Date` and format for display
- Do NOT assume `gc.core.time` is a number
- Avoid `new Date(timeValue)` and `timeValue / 1000` - use SDK helpers instead

### Common Patterns

**Format with GreyCat SDK (timezone-aware):**

```typescript
const label = gc.$.default.printTime(gcTime, gc.$.default.timezone, '%Y-%m-%d');
```

**Convert from JS Date:**

```typescript
const gcTime = gc.core.time.fromDate(new Date());
```

**Convert from milliseconds:**

```typescript
const gcTime = gc.core.time.fromMs(Date.now());
```

**Parse ISO string with timezone:**

```typescript
const gcTime = gc.$.default.parseTime('2025-03-15T10:30:00Z');
```

**Convert to JS Date:**

```typescript
const jsDate = gcTime.toDate();
```

**Get epoch milliseconds:**

```typescript
const ms = gcTime.epochMs;
```

### Where Time Fields Appear

Generated types reference `gc.core.time` for fields like:
- `CityStatsView.oldestBuilding`, `CityStatsView.newestBuilding`
- `BuildingView.constructionDate`, `BuildingView.dateRegistered`
- `BuildingSearchResult.constructionDate`

**Always use `.toDate()` or `.epochMs` when a JS `Date` is needed.**

### Format Specifiers

Common format specifiers for `gc.$.default.printTime()`:

| Specifier | Description | Example |
|-----------|-------------|---------|
| `%Y` | 4-digit year | 2025 |
| `%m` | 2-digit month | 03 |
| `%d` | 2-digit day | 15 |
| `%H` | 2-digit hour (24h) | 14 |
| `%M` | 2-digit minute | 30 |
| `%S` | 2-digit second | 45 |
| `%Y-%m-%d` | ISO date | 2025-03-15 |
| `%Y-%m-%d %H:%M:%S` | ISO datetime | 2025-03-15 14:30:45 |
