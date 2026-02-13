"use client";

import React, { useState } from "react";
import Link from "next/link";
import { DynamicMap } from "@/components/map/DynamicMap";
import { StatCard } from "@/components/dashboard/StatCard";
import { AnalysisChart } from "@/components/dashboard/AnalysisChart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { Button } from "@/components/ui/Button";
import { analyzeOrchard, type AnalysisResult, type Geometry } from "@/lib/api";
import { Droplet, AlertCircle, MapPin, ArrowLeft } from "lucide-react";

export default function AnalyzePage() {
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(
    null
  );
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePolygonDrawn = async (geometry: Geometry) => {
    console.log("Polygon drawn, geometry:", geometry);
    setIsLoading(true);
    setError(null);
    setAnalysisResult(null);

    try {
      // Validate polygon area (rough check)
      const coords = geometry.coordinates[0];
      let area = 0;
      for (let i = 0; i < coords.length - 1; i++) {
        area +=
          coords[i][0] * coords[i + 1][1] - coords[i + 1][0] * coords[i][1];
      }
      area = Math.abs(area / 2) * 111 * 111; // Convert to hectares (rough)

      if (area > 100) {
        setError(
          "Polygon too large. Please draw a smaller area (max 100 hectares)."
        );
        setIsLoading(false);
        return;
      }

      const result = await analyzeOrchard(geometry);
      console.log("Analysis result:", result);
      setAnalysisResult(result);
    } catch (err) {
      console.error("Error in handlePolygonDrawn:", err);
      setError("Failed to analyze orchard. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 shadow-sm">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Link href="/">
              <Button variant="ghost" size="sm">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Home
              </Button>
            </Link>
            <div className="flex items-center space-x-3">
              <MapPin className="h-6 w-6 text-blue-600" />
              <div>
                <h1 className="text-xl font-bold text-gray-900">
                  Orchard Analysis
                </h1>
                <p className="text-xs text-gray-600">
                  Draw a polygon to analyze water stress
                </p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Map Area */}
        <main className="flex-1 relative">
          <DynamicMap onPolygonDrawn={handlePolygonDrawn} />

          {/* Loading Overlay */}
          {isLoading && (
            <div className="absolute inset-0 bg-black/20 z-[1000] flex items-center justify-center">
              <Card className="bg-white p-6">
                <div className="flex flex-col items-center space-y-4">
                  <Skeleton className="h-12 w-12 rounded-full" />
                  <div className="text-center">
                    <p className="font-semibold text-gray-900">
                      Analyzing Orchard...
                    </p>
                    <p className="text-sm text-gray-600 mt-1">
                      Fetching satellite data and detecting anomalies
                    </p>
                  </div>
                </div>
              </Card>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="absolute top-4 right-4 z-[1000] max-w-sm">
              <Card className="bg-red-50 border-red-200">
                <CardContent className="pt-6">
                  <div className="flex items-start space-x-3">
                    <AlertCircle className="h-5 w-5 text-red-600 mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="text-sm font-semibold text-red-800">
                        Error
                      </p>
                      <p className="text-sm text-red-700 mt-1">{error}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </main>

        {/* Results Sidebar */}
        {analysisResult && (
          <aside className="w-96 bg-white border-l border-gray-200 overflow-y-auto">
            <div className="p-6 space-y-6">
              <div>
                <h2 className="text-xl font-bold text-gray-900 mb-4">
                  Analysis Results
                </h2>

                {/* Stat Cards */}
                <div className="grid grid-cols-1 gap-4 mb-6">
                  <StatCard
                    title="Stress Score"
                    value={(analysisResult.anomalyScore * 100).toFixed(1) + "%"}
                    status={analysisResult.stressStatus}
                  />
                  <StatCard
                    title="Water Content"
                    value={(analysisResult.waterContent * 100).toFixed(1) + "%"}
                    icon={<Droplet className="h-5 w-5" />}
                    status={
                      analysisResult.waterContent > 0.6
                        ? "Healthy"
                        : analysisResult.waterContent > 0.4
                        ? "Warning"
                        : "Critical"
                    }
                  />
                  {analysisResult.dtwDistance && (
                    <StatCard
                      title="Season Similarity"
                      value={analysisResult.dtwDistance.toFixed(1)}
                      className="text-gray-600 border-gray-200 bg-gray-50"
                    />
                  )}
                </div>

                {/* Chart */}
                <AnalysisChart
                  actualSeries={analysisResult.ndviSeries}
                  idealSeries={analysisResult.idealSeries}
                />

                {/* Action Hint */}
                <Card className="mt-6 bg-blue-50 border-blue-200">
                  <CardContent className="pt-6">
                    <p className="text-sm text-blue-900">
                      💡 Draw another polygon to analyze a different area
                    </p>
                  </CardContent>
                </Card>
              </div>
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}
