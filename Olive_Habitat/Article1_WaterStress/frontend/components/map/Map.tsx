"use client";

import React, { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet-draw/dist/leaflet.draw.css";

// Fix for default marker icons in Next.js
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
  iconUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
  shadowUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
});

interface MapProps {
  onPolygonDrawn: (geometry: {
    type: "Polygon";
    coordinates: number[][][];
  }) => void;
}

// Custom drawing button component
const DrawButton: React.FC<{ onClick: () => void }> = ({ onClick }) => {
  return (
    <button
      onClick={onClick}
      className="absolute top-4 right-4 z-[1000] bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg shadow-lg flex items-center space-x-2 transition-colors"
      aria-label="Start drawing polygon"
    >
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"
        />
      </svg>
      <span>Draw Polygon</span>
    </button>
  );
};

export const Map: React.FC<MapProps> = ({ onPolygonDrawn }) => {
  const mapRef = useRef<L.Map | null>(null);
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const drawControlRef = useRef<any>(null);
  const drawHandlerRef = useRef<L.Draw.Polygon | null>(null);
  const callbackRef = useRef(onPolygonDrawn);
  const [mapReady, setMapReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  // Keep callback ref updated
  useEffect(() => {
    callbackRef.current = onPolygonDrawn;
  }, [onPolygonDrawn]);

  useEffect(() => {
    // Prevent multiple initializations
    if (!mapContainerRef.current || mapRef.current) {
      return;
    }

    // Check if container already has a map instance
    if ((mapContainerRef.current as any)._leaflet_id) {
      console.warn("Map container already initialized, skipping...");
      return;
    }

    // Dynamically import leaflet-draw to ensure it extends Leaflet namespace
    const initMap = async () => {
      try {
        // Import leaflet-draw as a side effect
        await import("leaflet-draw");
        
        // Small delay to ensure namespace is extended
        await new Promise((resolve) => setTimeout(resolve, 100));
        
        // Double-check container is still available and not initialized
        if (!mapContainerRef.current || mapRef.current) {
          return;
        }
        
        if ((mapContainerRef.current as any)._leaflet_id) {
          console.warn("Map container already initialized during async init, skipping...");
          return;
        }
        
        // Now L.Control.Draw should be available
        const DrawControl = (L.Control as any).Draw;
        
        if (!DrawControl) {
          const errorMsg = "Leaflet Draw is not available. Check console for details.";
          console.error("DrawControl not found. L.Control:", L.Control);
          setError(errorMsg);
          return;
        }

        // Initialize map centered on Ghafsai, Morocco
        const map = L.map(mapContainerRef.current!, {
          center: [34.675, -4.818],
          zoom: 13,
        });

        // Add Esri World Imagery tile layer
        L.tileLayer(
          "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
          {
            attribution:
              "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
            maxZoom: 19,
          }
        ).addTo(map);

        mapRef.current = map;

        // Initialize draw control with explicit position
        const drawControl = new DrawControl({
          position: "topright", // Explicitly set position
          draw: {
            polygon: {
              allowIntersection: false,
              showArea: true,
              drawError: {
                color: "#e1e100",
                message: "<strong>Error:</strong> shape edges cannot cross!",
              },
              shapeOptions: {
                color: "#97009c",
                fillColor: "#97009c",
                fillOpacity: 0.2,
              },
            },
            polyline: false,
            rectangle: false,
            circle: false,
            circlemarker: false,
            marker: false,
          },
          edit: {
            featureGroup: new L.FeatureGroup(),
            remove: true,
          },
        });

        map.addControl(drawControl);
        drawControlRef.current = drawControl;

        // Handle polygon creation - this is the main callback (define first)
        const handleDrawCreated = (e: any) => {
          console.log("Draw event CREATED triggered", e);
          try {
            const layer = e.layer;
            const geoJSON = layer.toGeoJSON();
            
            console.log("GeoJSON:", geoJSON);
            
            if (geoJSON.type === "Polygon" && geoJSON.coordinates) {
              console.log("Calling onPolygonDrawn callback with coordinates:", geoJSON.coordinates);
              // Use ref to avoid stale closure
              callbackRef.current({
                type: "Polygon",
                coordinates: geoJSON.coordinates,
              });
            } else {
              console.warn("Invalid GeoJSON type or missing coordinates", geoJSON);
            }

            // Add to map for visualization
            const featureGroup = new L.FeatureGroup([layer]);
            map.addLayer(featureGroup);
            setIsDrawing(false);
          } catch (err) {
            console.error("Error in handleDrawCreated:", err);
            setError("Error processing drawn polygon");
          }
        };

        // Handle draw start
        const handleDrawStart = () => {
          console.log("Draw started");
          setIsDrawing(true);
        };

        // Handle draw stop/cancel
        const handleDrawStop = () => {
          console.log("Draw stopped");
          setIsDrawing(false);
        };

        // Create a polygon draw handler for programmatic drawing
        const drawHandler = new (L.Draw as any).Polygon(map, {
          allowIntersection: false,
          showArea: true,
          shapeOptions: {
            color: "#97009c",
            fillColor: "#97009c",
            fillOpacity: 0.2,
          },
        });
        
        // Also listen to the drawHandler's events (for custom button)
        drawHandler.on(L.Draw.Event.CREATED, handleDrawCreated);
        
        drawHandlerRef.current = drawHandler;

        // Register event listeners on map (for toolbar buttons)
        map.on(L.Draw.Event.CREATED, handleDrawCreated);
        map.on(L.Draw.Event.DRAWSTART, handleDrawStart);
        map.on(L.Draw.Event.DRAWSTOP, handleDrawStop);
        map.on(L.Draw.Event.DELETED, handleDrawStop);
        
        // Also listen for drawend event as backup
        map.on(L.Draw.Event.DRAWVERTEX, () => {
          console.log("Draw vertex added");
        });
        
        setMapReady(true);
        
        // Force map to invalidate size to ensure controls render
        setTimeout(() => {
          map.invalidateSize();
        }, 100);
      } catch (err) {
        console.error("Error initializing map:", err);
        setError("Failed to initialize map. Please refresh the page.");
      }
    };

    initMap();

    // Cleanup function
    return () => {
      if (mapRef.current) {
        try {
          mapRef.current.remove();
        } catch (err) {
          console.warn("Error removing map:", err);
        }
        mapRef.current = null;
      }
      // Clear the leaflet ID from container
      if (mapContainerRef.current && (mapContainerRef.current as any)._leaflet_id) {
        delete (mapContainerRef.current as any)._leaflet_id;
      }
    };
  }, []); // Empty deps - callback is accessed via ref

  const handleDrawButtonClick = () => {
    if (drawHandlerRef.current && mapRef.current) {
      if (isDrawing) {
        // Cancel current drawing
        drawHandlerRef.current.disable();
        setIsDrawing(false);
      } else {
        // Start drawing
        console.log("Enabling draw handler");
        drawHandlerRef.current.enable();
        setIsDrawing(true);
      }
    } else {
      console.warn("Draw handler or map not ready");
    }
  };

  return (
    <div className="relative w-full h-full min-h-[600px]">
      <div
        ref={mapContainerRef}
        className="w-full h-full z-0"
        style={{ position: "relative" }}
      />
      
      {/* Custom Draw Button - Always visible */}
      {mapReady && (
        <DrawButton
          onClick={handleDrawButtonClick}
        />
      )}
      
      {isDrawing && (
        <div className="absolute top-20 right-4 bg-yellow-100 border border-yellow-400 text-yellow-800 px-4 py-2 rounded z-[1000] shadow-lg">
          <p className="text-sm font-semibold">Drawing Mode Active</p>
          <p className="text-xs mt-1">Click on the map to start drawing your polygon</p>
        </div>
      )}
      
      {error && (
        <div className="absolute top-4 left-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded z-[1000] max-w-sm">
          <p className="font-semibold">Error</p>
          <p className="text-sm">{error}</p>
        </div>
      )}
      
    </div>
  );
};
