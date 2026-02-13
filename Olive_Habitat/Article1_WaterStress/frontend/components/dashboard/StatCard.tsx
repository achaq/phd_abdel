import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { cn } from "@/lib/utils";
import { AlertTriangle, CheckCircle, XCircle } from "lucide-react";

interface StatCardProps {
  title: string;
  value: string | number;
  status?: "Healthy" | "Warning" | "Critical";
  icon?: React.ReactNode;
  className?: string;
}

export const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  status,
  icon,
  className,
}) => {
  const getStatusColor = () => {
    switch (status) {
      case "Healthy":
        return "text-green-600 border-green-200 bg-green-50";
      case "Warning":
        return "text-yellow-600 border-yellow-200 bg-yellow-50";
      case "Critical":
        return "text-red-600 border-red-200 bg-red-50";
      default:
        return "text-gray-600 border-gray-200 bg-gray-50";
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case "Healthy":
        return <CheckCircle className="h-5 w-5" />;
      case "Warning":
        return <AlertTriangle className="h-5 w-5" />;
      case "Critical":
        return <XCircle className="h-5 w-5" />;
      default:
        return null;
    }
  };

  return (
    <Card className={cn(getStatusColor(), className)}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {icon || getStatusIcon()}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {status && (
          <p className="text-xs mt-1 capitalize opacity-70">{status}</p>
        )}
      </CardContent>
    </Card>
  );
};
