/**
 * Mock API Service for OliveGuard Frontend
 * Simulates backend responses for development/testing
 */

export interface AnalysisResult {
  anomalyScore: number; // 0.0 - 1.0 (higher = more stressed)
  ndviSeries: Array<{ date: string; value: number }>;
  idealSeries: Array<{ date: string; value: number }>;
  stressStatus: "Healthy" | "Warning" | "Critical";
  waterContent: number; // 0.0 - 1.0
  dtwDistance?: number; // Dynamic Time Warping distance
}

export interface Geometry {
  type: "Polygon";
  coordinates: number[][][];
}

/**
 * Simulates analyzing an orchard polygon
 * @param geometry - GeoJSON polygon geometry
 * @returns Promise with analysis results
 */
export const mockAnalyzeOrchard = async (
  geometry: Geometry
): Promise<AnalysisResult> => {
  // Simulate network delay
  await new Promise((resolve) => setTimeout(resolve, 1500));

  // Calculate polygon area (rough estimate)
  const coords = geometry.coordinates[0];
  let area = 0;
  for (let i = 0; i < coords.length - 1; i++) {
    area += coords[i][0] * coords[i + 1][1] - coords[i + 1][0] * coords[i][1];
  }
  area = Math.abs(area / 2);

  // Generate mock data based on area (larger areas might have more variation)
  const baseAnomaly = Math.min(0.3 + Math.random() * 0.4, 0.9);
  const anomalyScore = baseAnomaly;

  // Generate NDVI time series (last 12 months)
  const ndviSeries: Array<{ date: string; value: number }> = [];
  const idealSeries: Array<{ date: string; value: number }> = [];
  const now = new Date();
  
  for (let i = 11; i >= 0; i--) {
    const date = new Date(now);
    date.setMonth(date.getMonth() - i);
    const dateStr = date.toISOString().split("T")[0];
    
    // Ideal NDVI curve (seasonal variation)
    const month = date.getMonth();
    const idealValue = 0.5 + 0.2 * Math.sin((month / 12) * 2 * Math.PI - Math.PI / 2);
    idealSeries.push({ date: dateStr, value: idealValue });
    
    // Actual NDVI (with some stress-related deviation)
    const deviation = anomalyScore > 0.5 ? (Math.random() - 0.5) * 0.15 : (Math.random() - 0.5) * 0.05;
    const actualValue = Math.max(0.2, Math.min(0.8, idealValue + deviation));
    ndviSeries.push({ date: dateStr, value: actualValue });
  }

  // Determine stress status
  let stressStatus: "Healthy" | "Warning" | "Critical";
  if (anomalyScore < 0.4) {
    stressStatus = "Healthy";
  } else if (anomalyScore < 0.7) {
    stressStatus = "Warning";
  } else {
    stressStatus = "Critical";
  }

  // Water content (inverse of anomaly score)
  const waterContent = Math.max(0.1, 1 - anomalyScore * 0.8);

  // DTW distance (mock)
  const dtwDistance = anomalyScore * 50 + Math.random() * 10;

  return {
    anomalyScore,
    ndviSeries,
    idealSeries,
    stressStatus,
    waterContent,
    dtwDistance,
  };
};
