# Frontend Integration with React

React + GreyCat integration using `@greycat/web` SDK.

## Overview

**@greycat/web SDK provides**: Auto-generated TypeScript types, auth/session management, typed API communication, error handling.

**Flow**: `Backend (GCL) → greycat codegen ts → project.d.ts → Frontend (TS/React) → Global gc namespace`

## Installation & Setup

```bash
npm install https://get.greycat.io/files/sdk/web/dev/7.6/7.6.0-dev.tgz
```

**Dependencies**: `@greycat/web`, `react@^18.3`, `react-dom@^18.3`, `@tanstack/react-query@^5`, `react-router-dom@^6`

**vite.config.ts:**
```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import greycat from '@greycat/web/vite-plugin';

export default defineConfig({
  plugins: [react(), greycat({ greycat: 'http://127.0.0.1:8080' })],
  server: { port: 3000 },
});
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
gc.sdk.init().then(() => { import('./main'); });
```

**main.tsx:**
```typescript
import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from './App';

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 1000, gcTime: 5 * 60 * 1000, refetchOnWindowFocus: true, retry: 1 } }
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode><QueryClientProvider client={queryClient}><App /></QueryClientProvider></React.StrictMode>
);
```

## TypeScript Type Generation

```bash
greycat codegen ts  # Generates project.d.ts
```

**Regenerate when**: Changing GCL types, adding @expose functions, modifying @volatile types, before prod build.

**package.json scripts:**
```json
{ "scripts": { "types": "cd .. && greycat codegen ts", "dev": "npm run types && vite", "build": "npm run types && tsc && vite build" } }
```

## API Communication Patterns

**⚠️ CRITICAL**: @expose functions export to **top-level `gc` namespace**, NOT `gc.project`.

```typescript
// ✅ CORRECT
const city = await gc.getCity(cityId);
const results = await gc.searchCities(query, 20, options);

// ❌ INCORRECT
const city = await gc.project.getCity(cityId);  // ERROR!
```

**Direct gc namespace:**
```typescript
await gc.runtime.User.current();  // Auth/runtime APIs
await gc.sdk.login({ username, password, use_cookie: true });
await gc.sdk.logout();
const city = await gc.getCity(cityId);  // Your @expose functions (top-level)
```

**Naming**: Backend snake_case (get_city_by_id) → Frontend camelCase (getCityById).

## Handling GreyCat Enums

**Enum Serialization**: GreyCat enums are objects, not TypeScript enums. Use `.key!` to serialize.

```typescript
// Backend: enum UserRole { Admin, User, Guest }

// ✅ CORRECT - Serialize enum
const data = { role: user.role.key! };  // "Admin" as string
fetch('/api/update', { body: JSON.stringify(data) });

// ❌ INCORRECT - Sends full object
const data = { role: user.role };  // Sends { offset: 0, key: "Admin", ... }
```

**Deserialize**: `gc.core.UserRole.fromKey(json.role)`

**Display**:
```typescript
<div>Role: {user.role.key}</div>
```

**Map Keys**:
```typescript
const countsByRole = new Map<string, number>();  // Use .key! as string key
for (const user of users) { countsByRole.set(user.role.key!, (countsByRole.get(user.role.key!) ?? 0) + 1); }
```

## Authentication & Authorization

**Backend Person Type:**
```gcl
type Person { email: String; firstName: String; lastName: String; userId: int; roleId: int; }
var persons_by_id: nodeIndex<int, node<Person>>;
```

**Backend User Creation:**
```gcl
abstract type PersonService {
    static fn create(email: String, firstName: String, lastName: String, roleId: int, password: String): node<Person> {
        var userId = UserGroup::Default.add(UserRole::Admin, email, password);
        var person = node<Person>{ Person { email, firstName, lastName, userId, roleId } };
        persons_by_id.set(userId, person);
        return person;
    }
}
```

**Backend API:**
```gcl
@volatile type PersonView { email: String; firstName: String; lastName: String; roleId: int; }
@expose @permission("public") fn getCurrentPerson(): PersonView? {
    var user = User::current(); if (user == null) { return null; }
    var person = persons_by_id.get(user.id); if (person == null) { return null; }
    return PersonView { email: person->email, firstName: person->firstName, lastName: person->lastName, roleId: person->roleId };
}
```

**Frontend Auth Service (src/services/auth.ts):**
```typescript
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

**Person Hook (src/hooks/usePerson.ts):**
```typescript
import { useQuery } from '@tanstack/react-query';
import { authService } from '../services/auth';

export function usePerson() {
  return useQuery({
    queryKey: ['currentPerson'],
    queryFn: authService.getCurrentUser,
    staleTime: 5 * 60 * 1000,
    enabled: authService.isAuthenticated(),
  });
}
```

**Usage:**
```typescript
function App() {
  const { data: person, isLoading } = usePerson();
  if (isLoading) return <div>Loading...</div>;
  if (!person) return <LoginPage />;
  return <div>Welcome {person.firstName}!</div>;
}
```

## Service Layer Architecture

**Pattern**: Create service layer wrapping gc calls.

**src/services/geographic.ts:**
```typescript
export const geographicService = {
  getCountries: async () => await retryWithBackoff(() => gc.getCountries()),
  getCities: async (countryId: number) => await retryWithBackoff(() => gc.getCitiesByCountry(countryId)),
};

async function retryWithBackoff<T>(fn: () => Promise<T>, maxRetries = 3, delayMs = 1000): Promise<T> {
  let lastError: Error | undefined;
  for (let i = 0; i < maxRetries; i++) {
    try { return await fn(); }
    catch (error) { lastError = error as Error; if (i < maxRetries - 1) await new Promise(r => setTimeout(r, delayMs * Math.pow(2, i))); }
  }
  throw lastError;
}
```

## React Query Integration

**Query Hook:**
```typescript
import { useQuery } from '@tanstack/react-query';
export function useCities(countryId: number) {
  return useQuery({
    queryKey: ['cities', countryId],
    queryFn: () => gc.getCitiesByCountry(countryId),
    staleTime: 5 * 60 * 1000,
    enabled: !!countryId,
  });
}
```

**Mutation Hook:**
```typescript
import { useMutation, useQueryClient } from '@tanstack/react-query';
export function useCreateCity() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (params: { name: string; countryId: number }) => gc.createCity(params.name, params.countryId),
    onSuccess: (newCity) => {
      queryClient.invalidateQueries({ queryKey: ['cities', newCity.countryId] });
      queryClient.setQueryData(['city', newCity.id], newCity);
    },
  });
}
```

**Usage:**
```typescript
function CityList({ countryId }: { countryId: number }) {
  const { data: cities, isLoading, error } = useCities(countryId);
  const createCity = useCreateCity();

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <>
      <ul>{cities?.map(c => <li key={c.id}>{c.name}</li>)}</ul>
      <button onClick={() => createCity.mutate({ name: "Paris", countryId })}>Add City</button>
    </>
  );
}
```

## Error Handling

**Try-Catch Pattern:**
```typescript
try { const city = await gc.getCity(cityId); setCity(city); }
catch (error) { console.error('Failed to fetch city:', error); toast.error(error instanceof Error ? error.message : 'Unknown error'); }
```

**React Query Error Handling:**
```typescript
const { data, error, isError } = useQuery({
  queryKey: ['city', cityId],
  queryFn: () => gc.getCity(cityId),
  retry: (failureCount, error) => {
    if (error.message?.includes('404')) return false;  // Don't retry 404s
    return failureCount < 3;
  },
});

if (isError) return <div>Error: {error.message}</div>;
```

## Best Practices

1. **Always run `greycat codegen ts`** after backend changes
2. **Initialize SDK before React**: `gc.sdk.init().then(() => import('./main'))`
3. **Use top-level `gc` namespace**: NOT `gc.project`
4. **Serialize enums**: Use `.key!` when sending to backend/APIs
5. **Service layer**: Wrap gc calls for retry logic, error handling
6. **React Query**: Use for caching, auto-refetching, optimistic updates
7. **Type safety**: Leverage generated types, avoid `any`
8. **Error handling**: Try-catch or React Query error callbacks

## Common Pitfalls

| ❌ Wrong | ✅ Correct |
|----------|-----------|
| `gc.project.getCity()` | `gc.getCity()` |
| `{ role: user.role }` | `{ role: user.role.key! }` |
| React renders before SDK init | `gc.sdk.init().then(() => import('./main'))` |
| Forgot `greycat codegen ts` | Run after every backend change |
| Direct gc calls in components | Use service layer + React Query |
| `new Map<Enum, V>()` | `new Map<string, V>()` with `.key!` |

## Time Usage in Frontend

GreyCat `time` type (μs epoch) needs conversion for display.

**Backend:**
```gcl
@volatile type EventView { timestamp: time; }
@expose fn getEvents(): Array<EventView> { ... }
```

**Frontend:**
```typescript
// time is microseconds since epoch
const event = await gc.getEvents()[0];
const date = new Date(event.timestamp / 1000);  // Convert μs → ms
console.log(date.toLocaleString());  // Human-readable
```

**Display Component:**
```typescript
function EventTime({ timestamp }: { timestamp: number }) {
  const date = new Date(timestamp / 1000);
  return <time dateTime={date.toISOString()}>{date.toLocaleString()}</time>;
}
```

## Integration Checklist

- [ ] Install @greycat/web SDK
- [ ] Configure Vite plugin + tsconfig.json
- [ ] Add vite-env.d.ts reference types
- [ ] Initialize SDK before React: `gc.sdk.init().then()`
- [ ] Run `greycat codegen ts` after backend changes
- [ ] Update package.json scripts (types, dev, build)
- [ ] Implement auth service + usePerson hook
- [ ] Create service layer for API calls
- [ ] Setup React Query with QueryClientProvider
- [ ] Serialize enums with `.key!` before sending
- [ ] Test type safety: No `any` types
- [ ] Verify error handling: Try-catch or query callbacks
- [ ] Convert time (μs → ms) for Date display

## Additional Resources

- **Vite Plugin**: Auto-proxies requests to GreyCat backend
- **React Query DevTools**: Add `import { ReactQueryDevtools } from '@tanstack/react-query-devtools'` for debugging
- **GreyCat Explorer**: `http://localhost:8080/explorer` for testing APIs
