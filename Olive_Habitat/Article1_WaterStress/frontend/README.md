# OliveGuard Frontend MVP

Interactive dashboard for AI-powered water stress detection in olive orchards.

## Features

- **Interactive Map**: Draw polygons on satellite imagery to analyze olive orchards
- **Real-time Analysis**: Mock AI analysis with stress scores and NDVI trends
- **Data Visualization**: Charts showing actual vs. ideal NDVI curves
- **Responsive UI**: Modern, clean interface built with Next.js and Tailwind CSS

## Tech Stack

- **Next.js 14** (App Router)
- **TypeScript**
- **Tailwind CSS**
- **Leaflet** (Interactive maps)
- **Recharts** (Data visualization)
- **Lucide React** (Icons)

## Getting Started

### Install Dependencies

```bash
npm install
```

### Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build for Production

```bash
npm run build
npm start
```

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx       # Root layout
│   ├── page.tsx         # Main dashboard
│   └── globals.css      # Global styles
├── components/
│   ├── ui/              # Reusable UI components
│   ├── map/             # Map components
│   └── dashboard/       # Analysis widgets
├── lib/
│   ├── mock-api.ts      # Mock backend service
│   └── utils.ts         # Utility functions
└── public/              # Static assets
```

## Usage

1. Click the polygon drawing tool on the map
2. Draw a polygon around an olive orchard area
3. Wait for the analysis to complete (~1.5s)
4. View results in the sidebar:
   - Stress Score
   - Water Content
   - NDVI Time Series Chart

## Mock API

The frontend currently uses a mock API service (`lib/mock-api.ts`) that simulates backend responses. This will be replaced with real API calls when the backend is ready.
