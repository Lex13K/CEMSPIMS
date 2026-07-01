const API = "";

export type GraphNode = {
  id: string;
  kind: string;
  label: string;
  pipeline: string;
  step_id: string;
  run_id: string | null;
  color: string;
  status: string;
  position: { x: number; y: number };
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  kind: string;
  label?: string;
  parent_run_id?: string;
  child_run_id?: string;
};

export async function fetchGraph() {
  const r = await fetch(`${API}/api/graph`);
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<{
    nodes: GraphNode[];
    edges: GraphEdge[];
    meta: unknown;
    jobs?: unknown[];
    live_by_run?: Record<string, unknown>;
  }>;
}

export async function rescanState() {
  const r = await fetch(`${API}/api/state/rescan`, { method: "POST" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchNodeDetail(id: string) {
  const r = await fetch(`${API}/api/nodes/${encodeURIComponent(id)}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchNodeArtifacts(id: string) {
  const r = await fetch(`${API}/api/nodes/${encodeURIComponent(id)}/artifacts`);
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<{ artifacts: unknown[] }>;
}

export async function fetchNodeConfigFields(id: string) {
  const r = await fetch(`${API}/api/nodes/${encodeURIComponent(id)}/config-fields`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchBranchFields(id: string) {
  const r = await fetch(`${API}/api/nodes/${encodeURIComponent(id)}/branch-fields`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchEdgeDetail(id: string) {
  const r = await fetch(`${API}/api/edges/${id}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchRuns() {
  const r = await fetch(`${API}/api/runs`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchPipelines() {
  const r = await fetch(`${API}/api/pipelines`);
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<{ order: string[] }>;
}

export async function fetchEligibleCompare() {
  const r = await fetch(`${API}/api/compare/eligible`);
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<string[]>;
}

export async function postCompare(body: {
  anchor: string;
  peer_runs: string[];
  comparison_id: string;
  preset?: string;
}) {
  const r = await fetch(`${API}/api/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function postBranch(body: {
  parent: string;
  child: string;
  at_pipeline: string;
  config_patch: Record<string, unknown>;
}) {
  const r = await fetch(`${API}/api/runs/branch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function postBranchAndRun(body: {
  parent: string;
  child: string;
  at_pipeline: string;
  at_step_id?: string;
  config_patch: Record<string, unknown>;
  pipelines: string[];
  overwrite?: string[];
}) {
  const r = await fetch(`${API}/api/runs/branch-and-run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    let msg = await r.text();
    try {
      const parsed = JSON.parse(msg) as { detail?: unknown };
      if (typeof parsed.detail === "string") msg = parsed.detail;
    } catch {
      /* keep raw */
    }
    throw new Error(msg);
  }
  return r.json();
}

export async function extendScope(
  runId: string,
  body: {
    add_pipelines: string[];
    pipelines_to_run?: string[] | null;
    overwrite?: string[];
  }
) {
  const r = await fetch(`${API}/api/runs/${encodeURIComponent(runId)}/extend-scope`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function previewConfigPatch(
  parentRunId: string,
  body: { parent: string; child: string; config_patch: Record<string, unknown> }
) {
  const r = await fetch(`${API}/api/runs/${encodeURIComponent(parentRunId)}/config/preview-patch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<{ diff: string[]; warnings: string[] }>;
}

export async function deleteRun(runId: string) {
  const r = await fetch(`${API}/api/runs/${encodeURIComponent(runId)}`, { method: "DELETE" });
  if (!r.ok) {
    let msg = await r.text();
    try {
      const parsed = JSON.parse(msg) as { detail?: unknown };
      if (typeof parsed.detail === "string") msg = parsed.detail;
    } catch {
      /* keep raw */
    }
    throw new Error(msg);
  }
  return r.json();
}

export async function postJob(body: { run_id: string; pipelines?: string[]; overwrite?: string[] }) {
  const r = await fetch(`${API}/api/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchJobs() {
  const r = await fetch(`${API}/api/jobs`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function cancelJob(id: string) {
  const r = await fetch(`${API}/api/jobs/${id}`, { method: "DELETE" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export function jobStreamUrl(jobId: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/api/jobs/${jobId}/stream`;
}

export async function getConfig(runId: string) {
  const r = await fetch(`${API}/api/runs/${runId}/config`);
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<{ content: string }>;
}

export async function patchConfig(runId: string, content: string) {
  const r = await fetch(`${API}/api/runs/${runId}/config`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
