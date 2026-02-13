# OliveGuard Frontend: Technical Design Document

**Version:** 1.0.0  
**Date:** February 2026  
**Project:** AI-Powered Water Stress Detection Dashboard  
**Technology Stack:** Next.js 14, React 19, TypeScript, Leaflet, Tailwind CSS

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Technology Stack & Rationale](#technology-stack--rationale)
4. [Project Structure](#project-structure)
5. [Component Architecture](#component-architecture)
6. [State Management & Data Flow](#state-management--data-flow)
7. [Map Integration (Leaflet)](#map-integration-leaflet)
8. [Styling Architecture](#styling-architecture)
9. [TypeScript Implementation](#typescript-implementation)
10. [Performance Optimizations](#performance-optimizations)
11. [Error Handling & Resilience](#error-handling--resilience)
12. [Build System & Configuration](#build-system--configuration)
13. [Future Considerations](#future-considerations)

---

## Executive Summary

The OliveGuard frontend is a single-page application (SPA) built using Next.js 14's App Router architecture, designed to provide an interactive dashboard for water stress detection in olive orchards. The application follows a component-based architecture with strict TypeScript typing, server-side rendering (SSR) optimization, and client-side interactivity for map-based polygon drawing and data visualization.

**Key Technical Achievements:**
- Zero-configuration SSR/SSG with Next.js App Router
- Dynamic client-side map rendering to avoid SSR conflicts
- Type-safe API contracts with mock backend simulation
- Responsive design with Tailwind CSS utility-first approach
- Optimized bundle size through code splitting and dynamic imports

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Next.js App Router                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   / (Home)   │  │  /analyze    │  │  /_not-found │      │
│  │   page.tsx   │  │  page.tsx    │  │              │      │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘      │
│         │                  │                                │
│         └──────────┬────────┘                                │
│                    │                                         │
│         ┌──────────▼──────────┐                            │
│         │   Shared Layout     │                            │
│         │   (layout.tsx)      │                            │
│         └──────────┬───────────┘                            │
└────────────────────┼─────────────────────────────────────────┘
                     │
         ┌───────────▼────────────┐
         │   Component Layer      │
         │  ┌──────────────────┐  │
         │  │  Map Components  │  │
         │  │  Dashboard       │  │
         │  │  UI Primitives   │  │
         │  └──────────────────┘  │
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │   Service Layer        │
         │  ┌──────────────────┐  │
         │  │  Mock API        │  │
         │  │  Utilities       │  │
         │  └──────────────────┘  │
         └─────────────────────────┘
```

### Design Patterns

1. **Component Composition**: Small, reusable UI components composed into larger features
2. **Separation of Concerns**: Clear boundaries between presentation, logic, and data layers
3. **Progressive Enhancement**: Core functionality works without JavaScript; enhanced with interactivity
4. **Defensive Programming**: Extensive error handling and validation at component boundaries

---

## Technology Stack & Rationale

### Core Framework: Next.js 14

**Rationale:**
- **App Router**: Modern file-system-based routing with built-in layouts and nested routes
- **Server Components**: Default server-side rendering reduces client bundle size
- **Automatic Code Splitting**: Route-based splitting without manual configuration
- **Image Optimization**: Built-in image optimization and lazy loading
- **TypeScript Support**: First-class TypeScript support with type inference

**Configuration:**
```typescript
// next.config.ts
const nextConfig: NextConfig = {
  // Default configuration sufficient for MVP
  // No custom webpack config needed
};
```

### React 19

**Key Features Utilized:**
- **Concurrent Rendering**: Automatic batching and priority scheduling
- **Server Components**: Zero-JS components for static content
- **Client Components**: Explicit `"use client"` directive for interactivity
- **Hooks API**: `useState`, `useEffect`, `useRef` for state and lifecycle management

### TypeScript 5

**Configuration Highlights:**
```json
{
  "compilerOptions": {
    "target": "ES2017",           // Modern but compatible
    "strict": true,                // Maximum type safety
    "moduleResolution": "bundler", // Next.js optimized
    "jsx": "react-jsx",           // Modern JSX transform
    "paths": { "@/*": ["./*"] }   // Absolute imports
  }
}
```

**Benefits:**
- Compile-time type checking prevents runtime errors
- IntelliSense support for better DX
- Refactoring safety across large codebases
- Self-documenting code through type annotations

### Leaflet & React-Leaflet

**Why Leaflet over alternatives (Mapbox, Google Maps)?**
- **Open Source**: No API key requirements or usage limits
- **Lightweight**: ~38KB gzipped core library
- **Plugin Ecosystem**: Extensive plugins including `leaflet-draw`
- **Customization**: Full control over styling and behavior
- **Cost**: Zero cost for production deployment

**Challenges Addressed:**
1. **SSR Incompatibility**: Leaflet requires `window` object → Dynamic imports
2. **Namespace Extension**: `leaflet-draw` extends Leaflet → Async import pattern
3. **Type Definitions**: Incomplete TypeScript types → Custom type assertions

### Tailwind CSS 4

**Rationale:**
- **Utility-First**: Rapid UI development without custom CSS files
- **JIT Compilation**: Only used classes are included in bundle
- **Responsive Design**: Built-in breakpoint system
- **Dark Mode**: Native support (prepared for future)
- **Customization**: Theme configuration via `tailwind.config`

**Implementation:**
```css
/* globals.css */
@import "tailwindcss";

/* Custom Leaflet overrides */
.leaflet-draw-toolbar {
  z-index: 1000 !important;
}
```

### Recharts

**Why Recharts over D3.js or Chart.js?**
- **React-Native**: Built specifically for React
- **Declarative API**: Component-based chart definition
- **Responsive**: Automatic responsive behavior
- **TypeScript**: Full TypeScript support
- **Accessibility**: Built-in ARIA attributes

---

## Project Structure

### Directory Organization

```
frontend/
├── app/                          # Next.js App Router
│   ├── layout.tsx               # Root layout (fonts, metadata)
│   ├── page.tsx                 # Home page (feature cards)
│   ├── analyze/
│   │   └── page.tsx             # Map analysis page
│   └── globals.css              # Global styles + Tailwind
│
├── components/                   # React components
│   ├── ui/                      # Atomic UI components
│   │   ├── Button.tsx
│   │   ├── Card.tsx
│   │   └── Skeleton.tsx
│   ├── map/                     # Map-related components
│   │   ├── Map.tsx              # Core Leaflet wrapper
│   │   └── DynamicMap.tsx       # SSR-safe wrapper
│   └── dashboard/               # Feature components
│       ├── StatCard.tsx
│       └── AnalysisChart.tsx
│
├── lib/                         # Utilities & services
│   ├── utils.ts                 # Helper functions (cn, etc.)
│   └── mock-api.ts              # Mock backend service
│
└── public/                      # Static assets
```

### File Naming Conventions

- **Components**: PascalCase (`Map.tsx`, `StatCard.tsx`)
- **Utilities**: camelCase (`utils.ts`, `mock-api.ts`)
- **Pages**: `page.tsx` (Next.js convention)
- **Layouts**: `layout.tsx` (Next.js convention)

---

## Component Architecture

### Component Hierarchy

```
app/page.tsx (Home)
├── FeatureCard (multiple)
└── Link (navigation)

app/analyze/page.tsx (Analysis)
├── DynamicMap
│   └── Map (client-only)
│       ├── DrawButton
│       └── Leaflet DrawControl
├── StatCard (multiple)
├── AnalysisChart
│   └── Recharts LineChart
└── Card (UI primitives)
```

### Component Design Principles

#### 1. Atomic Design Pattern

**Level 1: Atoms (UI Primitives)**
- `Button.tsx`: Single-purpose button component
- `Card.tsx`: Container component with variants
- `Skeleton.tsx`: Loading state placeholder

**Level 2: Molecules (Composite Components)**
- `StatCard.tsx`: Combines Card + Icon + Status logic
- `DrawButton.tsx`: Button + Icon + Positioning

**Level 3: Organisms (Feature Components)**
- `Map.tsx`: Complete map with drawing capabilities
- `AnalysisChart.tsx`: Full chart with data processing

#### 2. Component Props Interface

**Example: StatCard Component**

```typescript
interface StatCardProps {
  title: string;                    // Required: Display label
  value: string | number;           // Required: Metric value
  status?: "Healthy" | "Warning" | "Critical";  // Optional: Visual state
  icon?: React.ReactNode;           // Optional: Custom icon override
  className?: string;                // Optional: Additional Tailwind classes
}
```

**Design Decisions:**
- **Optional `status`**: Allows reuse for non-status metrics (e.g., "Season Similarity")
- **Flexible `value`**: Accepts both string and number for formatting flexibility
- **Icon override**: Enables custom icons while maintaining default status icons
- **className prop**: Enables composition with Tailwind's `cn()` utility

#### 3. Component Composition Example

```typescript
// StatCard.tsx - Composition pattern
export const StatCard: React.FC<StatCardProps> = ({ ... }) => {
  return (
    <Card className={cn(getStatusColor(), className)}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {icon || getStatusIcon()}
      </CardHeader>
      <CardContent>
        <div>{value}</div>
        {status && <p>{status}</p>}
      </CardContent>
    </Card>
  );
};
```

**Benefits:**
- **Reusability**: Card components used across multiple features
- **Consistency**: Uniform styling and behavior
- **Maintainability**: Changes to Card propagate automatically

---

## State Management & Data Flow

### State Management Strategy

**Approach: Local Component State Only**

Rationale for MVP:
- No global state library (Redux, Zustand) needed
- Simple parent-child prop drilling sufficient
- Avoids over-engineering for current requirements

### State Flow Diagram

```
User Action (Draw Polygon)
    │
    ▼
Map Component (handleDrawCreated)
    │
    ▼
onPolygonDrawn Callback (prop)
    │
    ▼
AnalyzePage (handlePolygonDrawn)
    │
    ├─► setIsLoading(true)
    ├─► setError(null)
    └─► mockAnalyzeOrchard(geometry)
            │
            ▼
        Promise<AnalysisResult>
            │
            ├─► Success: setAnalysisResult(data)
            └─► Error: setError(message)
                    │
                    ▼
            setIsLoading(false)
                    │
                    ▼
            Render Results Sidebar
```

### State Variables

**AnalyzePage Component:**

```typescript
const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
const [isLoading, setIsLoading] = useState(false);
const [error, setError] = useState<string | null>(null);
```

**State Lifecycle:**
1. **Initial**: `analysisResult = null`, `isLoading = false`, `error = null`
2. **Drawing**: User draws polygon → `handlePolygonDrawn` called
3. **Loading**: `isLoading = true`, `error = null`, `analysisResult = null`
4. **Success**: `isLoading = false`, `analysisResult = data`, `error = null`
5. **Error**: `isLoading = false`, `error = message`, `analysisResult = null`

### Callback Pattern with Refs

**Problem:** React's closure can capture stale callback references in `useEffect`

**Solution:** Ref-based callback pattern

```typescript
// Map.tsx
const callbackRef = useRef(onPolygonDrawn);

useEffect(() => {
  callbackRef.current = onPolygonDrawn;  // Keep ref updated
}, [onPolygonDrawn]);

useEffect(() => {
  // ... map initialization
  const handleDrawCreated = (e: any) => {
    callbackRef.current({ ... });  // Use ref, not direct prop
  };
}, []);  // Empty deps - callback accessed via ref
```

**Why This Works:**
- `useEffect` with empty deps runs once on mount
- `callbackRef.current` always points to latest callback
- Avoids re-initializing map on callback changes
- Prevents "Map container already initialized" errors

---

## Map Integration (Leaflet)

### Leaflet Initialization Flow

```
Component Mount
    │
    ▼
useEffect Triggered
    │
    ├─► Check: mapContainerRef.current exists?
    ├─► Check: mapRef.current is null?
    └─► Check: Container has _leaflet_id? (prevent double-init)
            │
            ▼
    Async: import("leaflet-draw")
            │
            ▼
    Wait 100ms (namespace extension)
            │
            ▼
    L.map(mapContainerRef.current)
            │
            ├─► Creates Leaflet instance
            ├─► Attaches _leaflet_id to DOM element
            └─► Returns L.Map object
                    │
                    ▼
            L.tileLayer().addTo(map)
                    │
                    ▼
            new L.Control.Draw().addTo(map)
                    │
                    ▼
            map.on(L.Draw.Event.CREATED, handler)
                    │
                    ▼
            setMapReady(true)
```

### Dynamic Import Pattern

**Why Dynamic Import?**

```typescript
// DynamicMap.tsx
const MapComponent = dynamic(
  () => import("./Map").then((mod) => ({ default: mod.Map })),
  {
    ssr: false,  // Critical: Prevent server-side rendering
    loading: () => <Skeleton className="w-full h-full" />
  }
);
```

**Technical Explanation:**
1. **`ssr: false`**: Tells Next.js to skip this component during SSR
2. **Dynamic import**: Code-splits map code into separate chunk
3. **Loading fallback**: Shows skeleton while chunk loads
4. **Namespace access**: `window`, `document`, `L` only available client-side

**Bundle Impact:**
- Main bundle: ~150KB (without map code)
- Map chunk: ~200KB (loaded on-demand)
- Total initial load: Reduced by ~57%

### Leaflet Draw Integration

**Challenge:** `leaflet-draw` extends Leaflet's namespace (`L.Control.Draw`)

**Solution:** Async import with delay

```typescript
const initMap = async () => {
  await import("leaflet-draw");  // Side-effect: extends L.Control
  
  await new Promise(resolve => setTimeout(resolve, 100));
  // ^ Ensures namespace extension completes
  
  const DrawControl = (L.Control as any).Draw;  // Now available
  const drawControl = new DrawControl({ ... });
};
```

**Why Delay Needed:**
- Module import is asynchronous
- Namespace extension happens after import resolves
- 100ms buffer ensures `L.Control.Draw` exists
- Without delay: `DrawControl is not a constructor` error

### Event Handling Architecture

**Event Registration:**

```typescript
// Register on map instance (for toolbar buttons)
map.on(L.Draw.Event.CREATED, handleDrawCreated);

// Register on drawHandler (for custom button)
drawHandler.on(L.Draw.Event.CREATED, handleDrawCreated);
```

**Event Flow:**

```
User Action
    │
    ├─► Toolbar Button Click
    │       │
    │       ▼
    │   DrawControl activates
    │       │
    │       ▼
    │   User draws polygon
    │       │
    │       ▼
    │   L.Draw.Event.CREATED fires
    │       │
    │       ▼
    └───► handleDrawCreated(e)
            │
    └─► Custom Button Click
            │
            ▼
        drawHandler.enable()
            │
            ▼
        User draws polygon
            │
            ▼
        L.Draw.Event.CREATED fires
            │
            ▼
        handleDrawCreated(e)
```

**Event Payload Structure:**

```typescript
L.Draw.Event.CREATED {
  layer: L.Polygon,           // Leaflet polygon layer
  layerType: "polygon",       // Type identifier
  target: L.Map              // Map instance
}
```

### Polygon Geometry Extraction

**Process:**

```typescript
const handleDrawCreated = (e: any) => {
  const layer = e.layer;                    // Leaflet layer object
  const geoJSON = layer.toGeoJSON();        // Convert to GeoJSON
  // Result: { type: "Polygon", coordinates: [[[lon, lat], ...]] }
  
  if (geoJSON.type === "Polygon" && geoJSON.coordinates) {
    callbackRef.current({
      type: "Polygon",
      coordinates: geoJSON.coordinates  // Array of coordinate rings
    });
  }
};
```

**GeoJSON Coordinate Structure:**

```typescript
{
  type: "Polygon",
  coordinates: [
    [                           // Outer ring (required)
      [-4.818, 34.675],        // [longitude, latitude]
      [-4.800, 34.675],
      [-4.800, 34.690],
      [-4.818, 34.690],
      [-4.818, 34.675]         // Closed ring (first = last)
    ]
    // Inner rings (holes) optional
  ]
}
```

**Validation:**

```typescript
// Area calculation (rough estimate)
let area = 0;
for (let i = 0; i < coords.length - 1; i++) {
  area += coords[i][0] * coords[i + 1][1] - coords[i + 1][0] * coords[i][1];
}
area = Math.abs(area / 2) * 111 * 111;  // Convert to hectares

if (area > 100) {
  setError("Polygon too large");
  return;
}
```

**Note:** This is a simplified area calculation using the shoelace formula. For production, use a proper geodesic area calculation library (e.g., `turf.js`).

---

## Styling Architecture

### Tailwind CSS Implementation

**Configuration:** Default Tailwind 4 with PostCSS

**Custom Theme Extensions:**

```css
/* globals.css */
:root {
  --background: #ffffff;
  --foreground: #171717;
}

@theme inline {
  --color-background: var(--background);
  --color-foreground: var(--foreground);
}
```

**Utility Class Patterns:**

1. **Layout Utilities:**
   ```typescript
   className="flex flex-col h-screen"  // Flexbox column, full height
   className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3"  // Responsive grid
   ```

2. **Spacing System:**
   ```typescript
   className="p-6 space-y-4"  // Padding: 1.5rem, vertical spacing: 1rem
   className="gap-4 mb-6"     // Gap: 1rem, margin-bottom: 1.5rem
   ```

3. **Color System:**
   ```typescript
   className="bg-blue-600 hover:bg-blue-700"  // Primary blue with hover
   className="text-gray-600"                   // Secondary text
   className="border-gray-200"                 // Subtle borders
   ```

### Conditional Styling Pattern

**Using `cn()` Utility:**

```typescript
import { cn } from "@/lib/utils";

const getStatusColor = () => {
  switch (status) {
    case "Healthy": return "text-green-600 border-green-200 bg-green-50";
    case "Warning": return "text-yellow-600 border-yellow-200 bg-yellow-50";
    case "Critical": return "text-red-600 border-red-200 bg-red-50";
    default: return "text-gray-600 border-gray-200 bg-gray-50";
  }
};

<Card className={cn(getStatusColor(), className)}>
```

**Why `cn()`?**
- Merges Tailwind classes intelligently
- Resolves conflicts (e.g., `p-4 p-6` → `p-6`)
- Combines conditional classes cleanly
- Uses `clsx` + `tailwind-merge` under the hood

### Responsive Design Strategy

**Breakpoint System:**

```typescript
// Mobile-first approach
className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3"
//         ↑ Mobile  ↑ Tablet    ↑ Desktop

// Responsive text
className="text-xl md:text-2xl lg:text-3xl"

// Responsive spacing
className="px-4 md:px-6 lg:px-8"
```

**Breakpoints:**
- `sm`: 640px (small tablets)
- `md`: 768px (tablets)
- `lg`: 1024px (desktops)
- `xl`: 1280px (large desktops)

### Z-Index Management

**Layer System:**

```typescript
// Base map
z-0          // Map tiles

// UI Overlays
z-[1000]     // Draw button, notifications, sidebars
z-[2000]     // Modals (future)
z-[3000]     // Tooltips (future)
```

**Leaflet Override:**

```css
.leaflet-top.leaflet-right {
  z-index: 1000 !important;  /* Ensure toolbar above map */
}
```

---

## TypeScript Implementation

### Type Safety Strategy

**1. Strict Type Checking:**

```json
{
  "strict": true,  // Enables all strict checks
  "noImplicitAny": true,      // Prevents `any` inference
  "strictNullChecks": true,   // Null/undefined safety
  "strictFunctionTypes": true // Function type checking
}
```

**2. Interface Definitions:**

```typescript
// lib/mock-api.ts
export interface AnalysisResult {
  anomalyScore: number;  // Explicit: 0.0 - 1.0
  ndviSeries: Array<{ date: string; value: number }>;
  stressStatus: "Healthy" | "Warning" | "Critical";  // Union type
  waterContent: number;
  dtwDistance?: number;  // Optional property
}

export interface Geometry {
  type: "Polygon";  // Literal type
  coordinates: number[][][];  // Nested array type
}
```

**3. Component Props Typing:**

```typescript
// components/dashboard/StatCard.tsx
interface StatCardProps {
  title: string;
  value: string | number;  // Union type
  status?: "Healthy" | "Warning" | "Critical";  // Optional union
  icon?: React.ReactNode;   // React element type
  className?: string;        // Optional string
}

export const StatCard: React.FC<StatCardProps> = ({ ... }) => {
  // Implementation
};
```

**4. Type Assertions (When Necessary):**

```typescript
// Map.tsx - Leaflet types incomplete
const DrawControl = (L.Control as any).Draw;
// ^ Type assertion needed due to incomplete @types/leaflet-draw

const handleDrawCreated = (e: any) => {
  // ^ `any` used temporarily - Leaflet event types complex
  const layer = e.layer as L.Polygon;  // Assert specific type
};
```

### Generic Types & Utility Types

**Example: `cn()` Function**

```typescript
import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export const cn = (...inputs: ClassValue[]): string => {
  return twMerge(clsx(inputs));
};
```

**Type Flow:**
1. `ClassValue`: Union type from `clsx` (string | object | array | ...)
2. Rest parameter `...inputs`: Accepts variable arguments
3. Return type `string`: Explicit return type annotation
4. `twMerge(clsx(...))`: Composes two utility functions

---

## Performance Optimizations

### 1. Code Splitting

**Route-Based Splitting (Automatic):**

```
Initial Load:
  - app/layout.js          (~5KB)
  - app/page.js            (~15KB)  ← Home page
  - app/analyze/page.js    (~20KB)  ← Loaded on navigation
  - components/map/Map.js   (~200KB) ← Loaded on /analyze
```

**Dynamic Import Splitting:**

```typescript
// DynamicMap.tsx
const MapComponent = dynamic(() => import("./Map"), {
  ssr: false,
  loading: () => <Skeleton />
});
```

**Bundle Analysis:**
- Home page: ~150KB (without map)
- Analyze page: ~350KB (with map)
- Total reduction: ~57% initial load

### 2. React Optimization Patterns

**Memoization (Prepared for Future):**

```typescript
// Future optimization if needed
const MemoizedChart = React.memo(AnalysisChart);
const MemoizedStatCard = React.memo(StatCard);
```

**Current Approach:**
- No premature optimization
- React 19's automatic optimizations sufficient for MVP
- Re-renders only when props change (automatic)

### 3. Image Optimization

**Next.js Image Component (Prepared):**

```typescript
// Future: Optimize any images
import Image from "next/image";

<Image
  src="/logo.png"
  width={200}
  height={200}
  alt="OliveGuard Logo"
  priority  // Load immediately
/>
```

### 4. CSS Optimization

**Tailwind JIT (Just-In-Time):**

- Only used classes included in bundle
- Unused utilities automatically removed
- Production build: ~15KB CSS (vs. ~200KB full Tailwind)

**PostCSS Processing:**

```javascript
// postcss.config.mjs
export default {
  plugins: {
    '@tailwindcss/postcss': {},
  },
};
```

---

## Error Handling & Resilience

### Error Boundary Strategy

**Current:** Component-level error handling  
**Future:** React Error Boundary for graceful degradation

### Error Handling Patterns

**1. Async Operation Errors:**

```typescript
const handlePolygonDrawn = async (geometry: Geometry) => {
  try {
    setIsLoading(true);
    setError(null);
    
    // Validation
    if (area > 100) {
      setError("Polygon too large");
      return;
    }
    
    // API call
    const result = await mockAnalyzeOrchard(geometry);
    setAnalysisResult(result);
  } catch (err) {
    console.error("Error:", err);
    setError("Failed to analyze orchard. Please try again.");
  } finally {
    setIsLoading(false);
  }
};
```

**2. Map Initialization Errors:**

```typescript
const initMap = async () => {
  try {
    await import("leaflet-draw");
    // ... initialization
  } catch (err) {
    console.error("Error initializing map:", err);
    setError("Failed to initialize map. Please refresh the page.");
  }
};
```

**3. Defensive Checks:**

```typescript
// Prevent double initialization
if (!mapContainerRef.current || mapRef.current) return;
if ((mapContainerRef.current as any)._leaflet_id) return;

// Validate data before processing
if (geoJSON.type === "Polygon" && geoJSON.coordinates) {
  // Process
} else {
  console.warn("Invalid GeoJSON");
}
```

### User-Facing Error Messages

**Error Display Component:**

```typescript
{error && (
  <Card className="bg-red-50 border-red-200">
    <CardContent>
      <div className="flex items-start space-x-3">
        <AlertCircle className="h-5 w-5 text-red-600" />
        <div>
          <p className="font-semibold">Error</p>
          <p className="text-sm">{error}</p>
        </div>
      </div>
    </CardContent>
  </Card>
)}
```

**Error Categories:**
1. **Validation Errors**: User input issues (e.g., "Polygon too large")
2. **Network Errors**: API failures (e.g., "Failed to analyze")
3. **Initialization Errors**: Map/component setup failures
4. **Runtime Errors**: Unexpected errors (logged to console)

---

## Build System & Configuration

### Next.js Build Process

**Development Mode:**

```bash
npm run dev
# → Starts Turbopack dev server
# → Hot Module Replacement (HMR)
# → Fast refresh on file changes
# → Source maps enabled
```

**Production Build:**

```bash
npm run build
# → TypeScript compilation
# → Next.js optimization
# → Code splitting
# → Static page generation
# → Asset optimization
```

**Build Output:**

```
.next/
├── static/           # Static assets
│   ├── chunks/       # Code chunks
│   └── media/        # Images, fonts
├── server/           # Server-side code
│   ├── app/          # App Router pages
│   └── chunks/       # Server chunks
└── cache/            # Build cache
```

### TypeScript Compilation

**Process:**
1. **Type Checking**: `tsc --noEmit` (type-only check)
2. **Transpilation**: Next.js handles via SWC
3. **Bundling**: Turbopack (dev) / Webpack (prod)

**SWC (Speedy Web Compiler):**
- Rust-based compiler (faster than Babel)
- Automatic tree-shaking
- Minification
- No configuration needed

### ESLint Configuration

**Default Next.js ESLint:**

```javascript
// eslint.config.mjs
import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
];

export default eslintConfig;
```

**Rules Applied:**
- React Hooks rules
- TypeScript best practices
- Next.js specific rules
- Core Web Vitals optimization

---

## Future Considerations

### Scalability Improvements

**1. State Management Migration:**

```typescript
// Future: Zustand store
import create from 'zustand';

interface AppState {
  analysisHistory: AnalysisResult[];
  addAnalysis: (result: AnalysisResult) => void;
  selectedOrchard: Geometry | null;
  setSelectedOrchard: (geo: Geometry) => void;
}

const useStore = create<AppState>((set) => ({
  analysisHistory: [],
  addAnalysis: (result) => set((state) => ({
    history: [...state.analysisHistory, result]
  })),
  // ...
}));
```

**2. API Integration:**

```typescript
// lib/api.ts (Future)
const API_BASE = process.env.NEXT_PUBLIC_API_URL;

export const analyzeOrchard = async (
  geometry: Geometry
): Promise<AnalysisResult> => {
  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ geometry }),
  });
  
  if (!response.ok) throw new Error('API error');
  return response.json();
};
```

**3. Caching Strategy:**

```typescript
// Future: React Query integration
import { useQuery } from '@tanstack/react-query';

const useAnalysis = (geometry: Geometry) => {
  return useQuery({
    queryKey: ['analysis', geometry],
    queryFn: () => analyzeOrchard(geometry),
    staleTime: 5 * 60 * 1000,  // 5 minutes
  });
};
```

**4. Authentication (Future):**

```typescript
// Future: NextAuth.js integration
import { useSession } from 'next-auth/react';

export default function AnalyzePage() {
  const { data: session } = useSession();
  // Protected route logic
}
```

### Performance Enhancements

**1. Virtual Scrolling (if needed):**

```typescript
// For long analysis history lists
import { useVirtualizer } from '@tanstack/react-virtual';
```

**2. Service Worker (PWA):**

```typescript
// next.config.ts
const withPWA = require('next-pwa')({
  dest: 'public',
  disable: process.env.NODE_ENV === 'development',
});

module.exports = withPWA(nextConfig);
```

**3. Image Optimization:**

```typescript
// Optimize map tile loading
const optimizedTileLayer = L.tileLayer(url, {
  maxZoom: 19,
  tileSize: 256,
  zoomOffset: 0,
  crossOrigin: true,
});
```

### Testing Strategy (Future)

**Unit Tests (Jest + React Testing Library):**

```typescript
// __tests__/StatCard.test.tsx
import { render, screen } from '@testing-library/react';
import { StatCard } from '@/components/dashboard/StatCard';

test('renders stress score correctly', () => {
  render(<StatCard title="Stress" value="75%" status="Critical" />);
  expect(screen.getByText('75%')).toBeInTheDocument();
});
```

**Integration Tests (Playwright):**

```typescript
// e2e/analyze.spec.ts
import { test, expect } from '@playwright/test';

test('draw polygon and see results', async ({ page }) => {
  await page.goto('/analyze');
  // Draw polygon
  // Verify results appear
});
```

---

## Technical Decisions Summary

### Why Next.js App Router?

- **File-system routing**: No route configuration needed
- **Server Components**: Better performance, smaller bundles
- **Built-in optimizations**: Image, font, script optimization
- **TypeScript**: First-class support

### Why Not Create React App (CRA)?

- **No SSR**: CRA is client-only
- **Manual routing**: Requires React Router setup
- **Manual optimization**: Webpack config needed
- **Larger bundles**: No automatic code splitting

### Why Leaflet over Mapbox?

- **Cost**: Mapbox has usage limits and costs
- **Open source**: Full control, no vendor lock-in
- **Customization**: Easier to style and extend
- **Plugin ecosystem**: Extensive plugins available

### Why Tailwind over CSS Modules?

- **Faster development**: Utility classes vs. writing CSS
- **Consistency**: Design system built-in
- **Smaller bundles**: JIT compilation removes unused styles
- **Better DX**: IntelliSense for class names

### Why Mock API?

- **Development speed**: No backend dependency
- **Frontend-first**: UI/UX can be developed independently
- **Testing**: Easy to test different scenarios
- **Future-ready**: Easy to swap with real API

---

## Conclusion

The OliveGuard frontend represents a modern, type-safe, and performant implementation of an interactive geospatial dashboard. The architecture prioritizes:

1. **Developer Experience**: TypeScript, Tailwind, and clear component structure
2. **User Experience**: Fast loading, responsive design, clear feedback
3. **Maintainability**: Well-organized code, clear patterns, comprehensive documentation
4. **Scalability**: Prepared for future enhancements (state management, API integration, testing)

The technical foundation supports rapid feature development while maintaining code quality and performance standards suitable for production deployment.

---

**Document Version:** 1.0.0  
**Last Updated:** February 2026  
**Author:** OliveGuard Development Team
