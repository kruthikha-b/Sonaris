// src/services/api.js

export const mockSurvey = {
  id: "SUR-2026-001",
  name: "Bay Area Survey",
  location: "Eastern Arabian Sea",
  date: "08 Sep 2026",
  status: "Completed",
  imageName: "sample-side-scan-sonar.png",
  imageUrl: "/sample-side-scan-sonar.png",

  detections: [
    {
      id: "AN-001",
      type: "Marine debris",
      confidence: 94,
      priority: "High",
      status: "Uncertain",
      x: 31,
      y: 27,
    },
    {
      id: "AN-002",
      type: "Unknown anomaly",
      confidence: 87,
      priority: "Medium",
      status: "Confirmed",
      x: 66,
      y: 52,
    },
    {
      id: "AN-003",
      type: "Possible rock",
      confidence: 78,
      priority: "Low",
      status: "Rejected",
      x: 48,
      y: 72,
    },
  ],
};

export async function getSurvey() {
  return mockSurvey;
}

export async function getSurveyById(id) {
  if (id === mockSurvey.id) {
    return mockSurvey;
  }

  return null;
}

export async function getDashboardStats() {
  return {
    totalSurveys: 12,
    completedSurveys: 8,
    processingSurveys: 2,
    pendingSurveys: 2,
    totalDetections: mockSurvey.detections.length,
    highPriorityDetections: mockSurvey.detections.filter(
      (item) => item.priority === "High"
    ).length,
  };
}

export async function uploadSurveyImage(file) {
  return {
    success: true,
    message: "Survey image uploaded successfully",
    fileName: file.name,
    imageUrl: URL.createObjectURL(file),
  };
}

export async function processSurvey() {
  return {
    success: true,
    message: "Survey processing started",
    status: "Processing",
  };
}

export async function getProcessingStatus() {
  return {
    success: true,
    status: "Completed",
    progress: 100,
  };
}