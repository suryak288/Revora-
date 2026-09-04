import { DecideRequest, DecideResponse, HealthResponse, ExecuteResponse, AuditEvent, AuditResponse, BatchSimulationReport } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/health`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });
  if (!response.ok) {
    throw new ApiError(response.status, `Health check failed: ${response.statusText}`);
  }
  return response.json();
}

export async function evaluateDecision(request: DecideRequest): Promise<DecideResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/decide`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    let errorMessage = response.statusText;
    try {
      const errorData = await response.json();
      if (errorData.detail) {
        errorMessage = typeof errorData.detail === 'string' 
          ? errorData.detail 
          : JSON.stringify(errorData.detail);
      }
    } catch (e) {
      // Ignore JSON parse error if body is empty or not JSON
    }
    
    if (response.status === 503) {
      throw new ApiError(503, "The ML model is currently unavailable. Please ensure the backend has generated a model artifact.");
    }
    
    throw new ApiError(response.status, `Decision request failed: ${errorMessage}`);
  }

  return response.json();
}

export async function executeRecovery(request: DecideRequest): Promise<ExecuteResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/execute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    let errorMessage = response.statusText;
    try {
      const errorData = await response.json();
      if (errorData.detail) {
        errorMessage = typeof errorData.detail === 'string' 
          ? errorData.detail 
          : JSON.stringify(errorData.detail);
      }
    } catch (e) {
    }
    
    if (response.status === 503) {
      throw new ApiError(503, "The ML model is currently unavailable.");
    }
    
    throw new ApiError(response.status, `Execution request failed: ${errorMessage}`);
  }

  return response.json();
}

export async function getAuditTrail(): Promise<AuditEvent[]> {
  const response = await fetch(`${API_BASE_URL}/api/v1/audit`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    throw new ApiError(response.status, `Audit fetch failed: ${response.statusText}`);
  }

  const data: AuditResponse = await response.json();
  return data.events;
}

export async function simulateBatch(batchSize: number = 1000, seed: number = 2026): Promise<BatchSimulationReport> {
  const response = await fetch(`${API_BASE_URL}/api/v1/evaluation/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ batch_size: batchSize, seed }),
  });

  if (!response.ok) {
    let errorMessage = response.statusText;
    try {
      const errorData = await response.json();
      if (errorData.detail) errorMessage = typeof errorData.detail === 'string' ? errorData.detail : JSON.stringify(errorData.detail);
    } catch (e) {}
    throw new ApiError(response.status, `Batch simulation failed: ${errorMessage}`);
  }

  return response.json();
}
