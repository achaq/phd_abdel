"use client";

import dynamic from "next/dynamic";
import { Skeleton } from "@/components/ui/Skeleton";

const MapComponent = dynamic(() => import("./Map").then((mod) => ({ default: mod.Map })), {
  ssr: false,
  loading: () => <Skeleton className="w-full h-full min-h-[600px]" />,
});

interface DynamicMapProps {
  onPolygonDrawn: (geometry: {
    type: "Polygon";
    coordinates: number[][][];
  }) => void;
}

export const DynamicMap: React.FC<DynamicMapProps> = (props) => {
  return <MapComponent {...props} />;
};
