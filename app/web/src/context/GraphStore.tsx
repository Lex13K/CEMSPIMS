import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";

import { fetchGraph, rescanState } from "../api";
import type { GraphEdge, GraphNode } from "../api";

export type JobRecord = {
  job_id: string;
  run_id: string;
  status: string;
  created_at?: string;
  pipelines?: string[];
};

export type LiveRunState = {
  active_step: { pipeline: string; step_id: string } | null;
  live_log: string[];
  live_headline: string | null;
  active_job_id?: string | null;
};

type GraphState = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  colors: Record<string, string>;
  revision: number;
  loading: boolean;
  error: string | null;
  jobs: JobRecord[];
  liveByRun: Record<string, LiveRunState>;
  refresh: (opts?: { fitTarget?: string; silent?: boolean }) => Promise<void>;
  rescan: () => Promise<void>;
  optimisticRemoveRun: (runId: string) => void;
  lastFitTarget: string | null;
  clearFitTarget: () => void;
};

const GraphContext = createContext<GraphState | null>(null);

type SnapshotPayload = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  meta: { pipeline_colors?: Record<string, string>; revision?: number };
  jobs?: JobRecord[];
  live_by_run?: Record<string, LiveRunState>;
};

function applySnapshot(payload: SnapshotPayload, setters: {
  setNodes: (n: GraphNode[]) => void;
  setEdges: (e: GraphEdge[]) => void;
  setColors: (c: Record<string, string>) => void;
  setRevision: (r: number) => void;
  setJobs: (j: JobRecord[]) => void;
  setLiveByRun: (l: Record<string, LiveRunState>) => void;
  revisionRef: React.MutableRefObject<number>;
}) {
  setters.setNodes(payload.nodes || []);
  setters.setEdges(payload.edges || []);
  const meta = payload.meta || {};
  setters.setColors(meta.pipeline_colors || {});
  const rev = meta.revision ?? 0;
  setters.setRevision(rev);
  setters.revisionRef.current = rev;
  if (payload.jobs) setters.setJobs(payload.jobs);
  if (payload.live_by_run) setters.setLiveByRun(payload.live_by_run);
}

export function GraphProvider({ children }: { children: ReactNode }) {
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [colors, setColors] = useState<Record<string, string>>({});
  const [revision, setRevision] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [liveByRun, setLiveByRun] = useState<Record<string, LiveRunState>>({});
  const [lastFitTarget, setLastFitTarget] = useState<string | null>(null);
  const revisionRef = useRef(0);

  const setters = {
    setNodes,
    setEdges,
    setColors,
    setRevision,
    setJobs,
    setLiveByRun,
    revisionRef,
  };

  const refresh = useCallback(async (opts?: { fitTarget?: string; silent?: boolean }) => {
    if (!opts?.silent) setLoading(true);
    try {
      const g = await fetchGraph();
      applySnapshot(g as SnapshotPayload, setters);
      setError(null);
      if (opts?.fitTarget) setLastFitTarget(opts.fitTarget);
    } catch (e) {
      setError(String(e));
    } finally {
      if (!opts?.silent) setLoading(false);
    }
  }, []);

  const rescan = useCallback(async () => {
    setLoading(true);
    try {
      await rescanState();
      await refresh({ silent: true });
      setError(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [refresh]);

  const optimisticRemoveRun = useCallback((runId: string) => {
    setNodes((prev) => prev.filter((n) => n.run_id !== runId && !n.id.includes(`:${runId}:`)));
    setLiveByRun((prev) => {
      const next = { ...prev };
      delete next[runId];
      return next;
    });
  }, []);

  const clearFitTarget = useCallback(() => setLastFitTarget(null), []);

  useEffect(() => {
    const es = new EventSource("/api/stream");

    es.addEventListener("snapshot", (ev) => {
      try {
        const data = JSON.parse((ev as MessageEvent).data) as SnapshotPayload;
        applySnapshot(data, setters);
        setLoading(false);
        setError(null);
      } catch {
        /* ignore */
      }
    });

    es.addEventListener("node_update", (ev) => {
      try {
        const data = JSON.parse((ev as MessageEvent).data) as {
          run_id: string;
          pipeline: string;
          step_id: string;
          status: string;
          revision?: number;
        };
        const nid =
          data.pipeline === "data.prepare"
            ? `global:${data.pipeline}:${data.step_id}`
            : `run:${data.run_id}:${data.pipeline}:${data.step_id}`;
        setNodes((prev) =>
          prev.map((n) => (n.id === nid ? { ...n, status: data.status } : n))
        );
        if (data.revision != null) {
          setRevision(data.revision);
          revisionRef.current = data.revision;
        }
      } catch {
        /* ignore */
      }
    });

    es.addEventListener("step_log", (ev) => {
      try {
        const data = JSON.parse((ev as MessageEvent).data) as {
          run_id: string;
          line: string;
          live_headline?: string | null;
          live_log?: string[];
          revision?: number;
        };
        setLiveByRun((prev) => {
          const cur = prev[data.run_id] || {
            active_step: null,
            live_log: [],
            live_headline: null,
          };
          const log = [...cur.live_log, data.line].slice(-50);
          return {
            ...prev,
            [data.run_id]: {
              ...cur,
              live_log: data.live_log || log,
              live_headline: data.live_headline ?? cur.live_headline,
            },
          };
        });
        if (data.revision != null) {
          setRevision(data.revision);
          revisionRef.current = data.revision;
        }
      } catch {
        /* ignore */
      }
    });

    es.addEventListener("active_step", (ev) => {
      try {
        const data = JSON.parse((ev as MessageEvent).data) as {
          run_id: string;
          active_step: { pipeline: string; step_id: string } | null;
          status?: string;
          pipeline?: string;
          step_id?: string;
          revision?: number;
        };
        setLiveByRun((prev) => ({
          ...prev,
          [data.run_id]: {
            ...(prev[data.run_id] || { live_log: [], live_headline: null }),
            active_step: data.active_step,
            live_log: data.active_step ? [] : (prev[data.run_id]?.live_log || []),
            live_headline: data.active_step ? null : prev[data.run_id]?.live_headline,
          },
        }));
        if (data.pipeline && data.step_id && data.status) {
          const nid = `run:${data.run_id}:${data.pipeline}:${data.step_id}`;
          setNodes((prev) =>
            prev.map((n) => (n.id === nid ? { ...n, status: data.status! } : n))
          );
        }
      } catch {
        /* ignore */
      }
    });

    es.addEventListener("job_update", (ev) => {
      try {
        const data = JSON.parse((ev as MessageEvent).data) as { job: JobRecord };
        const job = data.job;
        setJobs((prev) => {
          const idx = prev.findIndex((j) => j.job_id === job.job_id);
          if (idx >= 0) {
            const next = [...prev];
            next[idx] = job;
            return next;
          }
          return [job, ...prev];
        });
      } catch {
        /* ignore */
      }
    });

    es.onerror = () => {
      setError("SSE connection lost — reconnecting…");
    };

    return () => es.close();
  }, []);

  return (
    <GraphContext.Provider
      value={{
        nodes,
        edges,
        colors,
        revision,
        loading,
        error,
        jobs,
        liveByRun,
        refresh,
        rescan,
        optimisticRemoveRun,
        lastFitTarget,
        clearFitTarget,
      }}
    >
      {children}
    </GraphContext.Provider>
  );
}

export function useGraph() {
  const ctx = useContext(GraphContext);
  if (!ctx) throw new Error("useGraph requires GraphProvider");
  return ctx;
}
