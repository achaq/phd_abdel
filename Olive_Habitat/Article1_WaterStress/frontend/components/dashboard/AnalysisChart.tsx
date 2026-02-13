"use client";

import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";

interface DataPoint {
  date: string;
  value: number;
}

interface AnalysisChartProps {
  actualSeries: DataPoint[];
  idealSeries: DataPoint[];
}

export const AnalysisChart: React.FC<AnalysisChartProps> = ({
  actualSeries,
  idealSeries,
}) => {
  // Combine data for Recharts
  const chartData = actualSeries.map((point, index) => ({
    date: new Date(point.date).toLocaleDateString("en-US", {
      month: "short",
      year: "2-digit",
    }),
    actual: point.value,
    ideal: idealSeries[index]?.value || 0,
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>NDVI Time Series Analysis</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 12 }}
              angle={-45}
              textAnchor="end"
              height={60}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fontSize: 12 }}
              label={{
                value: "NDVI",
                angle: -90,
                position: "insideLeft",
              }}
            />
            <Tooltip
              formatter={(value: number | undefined) =>
                value !== undefined ? value.toFixed(3) : ""
              }
              labelStyle={{ color: "#000" }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#ef4444"
              strokeWidth={2}
              name="Actual"
              dot={{ r: 3 }}
            />
            <Line
              type="monotone"
              dataKey="ideal"
              stroke="#10b981"
              strokeWidth={2}
              strokeDasharray="5 5"
              name="Ideal (Healthy Baseline)"
              dot={{ r: 3 }}
            />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-xs text-gray-500 mt-4">
          The gap between Actual and Ideal curves indicates water stress levels.
        </p>
      </CardContent>
    </Card>
  );
};
