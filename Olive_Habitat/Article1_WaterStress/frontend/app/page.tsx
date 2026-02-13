"use client";

import React from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { MapPin, BarChart3, Database, BookOpen, Settings } from "lucide-react";

const FeatureCard: React.FC<{
  title: string;
  description: string;
  icon: React.ReactNode;
  href?: string;
  disabled?: boolean;
}> = ({ title, description, icon, href, disabled = false }) => {
  const content = (
    <Card
      className={`h-full transition-all duration-200 ${
        disabled
          ? "opacity-60 cursor-not-allowed"
          : "hover:shadow-lg hover:scale-105 cursor-pointer"
      }`}
    >
      <CardHeader>
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-blue-100 rounded-lg text-blue-600">
            {icon}
          </div>
          <CardTitle className="text-xl">{title}</CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-gray-600 text-sm">{description}</p>
        {disabled && (
          <p className="text-xs text-gray-400 mt-2 italic">Coming soon</p>
        )}
      </CardContent>
    </Card>
  );

  if (disabled || !href) {
    return content;
  }

  return <Link href={href}>{content}</Link>;
};

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <MapPin className="h-8 w-8 text-blue-600" />
              <div>
                <h1 className="text-3xl font-bold text-gray-900">OliveGuard</h1>
                <p className="text-sm text-gray-600">
                  AI-Powered Water Stress Detection
                </p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm font-medium text-gray-900">
                Ghafsai, Morocco
              </p>
              <p className="text-xs text-gray-500">Pre-Rif Region</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-12">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Digital Twin for Olive Orchard Management
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Monitor water stress, analyze satellite data, and make
            data-driven decisions for your olive orchards using advanced AI
            and remote sensing technology.
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <FeatureCard
            title="Orchard Analysis"
            description="Draw polygons on satellite imagery to analyze water stress levels in your olive orchards. Get real-time AI-powered insights."
            icon={<MapPin className="h-6 w-6" />}
            href="/analyze"
          />

          <FeatureCard
            title="Historical Trends"
            description="View historical water stress patterns and NDVI trends over time to understand seasonal variations."
            icon={<BarChart3 className="h-6 w-6" />}
            disabled={true}
          />

          <FeatureCard
            title="Data Export"
            description="Export analysis results, time series data, and reports for further analysis or record keeping."
            icon={<Database className="h-6 w-6" />}
            disabled={true}
          />

          <FeatureCard
            title="Documentation"
            description="Learn about the methodology, algorithms, and scientific background behind OliveGuard."
            icon={<BookOpen className="h-6 w-6" />}
            disabled={true}
          />

          <FeatureCard
            title="Settings"
            description="Configure analysis parameters, thresholds, and notification preferences."
            icon={<Settings className="h-6 w-6" />}
            disabled={true}
          />
        </div>

        {/* Info Section */}
        <div className="mt-12 bg-white rounded-lg shadow-sm p-8 border border-gray-200">
          <h3 className="text-2xl font-semibold text-gray-900 mb-4">
            How It Works
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-blue-600 font-bold text-xl">1</span>
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">
                Draw Your Orchard
              </h4>
              <p className="text-sm text-gray-600">
                Use the interactive map to draw a polygon around your olive
                orchard area.
              </p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-blue-600 font-bold text-xl">2</span>
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">
                AI Analysis
              </h4>
              <p className="text-sm text-gray-600">
                Our AI processes satellite data to detect water stress and
                anomalies.
              </p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-blue-600 font-bold text-xl">3</span>
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">
                Get Insights
              </h4>
              <p className="text-sm text-gray-600">
                View detailed analysis results, charts, and recommendations.
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
