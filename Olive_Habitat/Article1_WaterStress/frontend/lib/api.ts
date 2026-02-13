/**
 * OliveGuard API Client
 * Calls the FastAPI backend for real analysis.
 */

export interface AnalysisResult {
  anomalyScore: number;
  ndviSeries: Array<{ date: string; value: number }>;
  idealSeries: Array<{ date: string; value: number }>;
  stressStatus: "Healthy" | "Warning" | "Critical";
  waterContent: number;
  dtwDistance?: number;
}

export interface Geometry {
  type: "Polygon";
  coordinates: number[][][];
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function analyzeOrchard(
  geometry: Geometry
): Promise<AnalysisResult> {
  const response = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ geometry }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || `Analysis failed: ${response.status}`);
  }

  return response.json();
}
