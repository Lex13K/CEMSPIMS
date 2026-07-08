import { useEffect, useRef, useState } from "react";
import { cancelJob, fetchJobs, fetchPipelines, fetchRuns, jobStreamUrl, postJob } from "../api";
import { useGraph } from "../context/GraphStore";

type Job = {
  job_id: string;
  run_id: string;
  status: string;
  created_at: string;
  pipelines?: string[];
  log_tail?: string[];
};

export default function JobsPage() {
  const { jobs: storeJobs } = useGraph();
  const [jobs, setJobs] = useState<Job[]>([]);
  const [jobsLoading, setJobsLoading] = useState(true);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [runs, setRuns] = useState<{ run_id: string }[]>([]);
  const [runId, setRunId] = useState("");
  const [pipelines, setPipelines] = useState<string[]>([]);
  const [allPipelines, setAllPipelines] = useState<string[]>([]);
  const [overwrite, setOverwrite] = useState(false);
  const [expandedLog, setExpandedLog] = useState<string | null>(null);
  const [liveLog, setLiveLog] = useState("");
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (storeJobs.length) {
      setJobs(storeJobs as Job[]);
      setJobsLoading(false);
    }
  }, [storeJobs]);

  const load = () => {
    setJobsLoading(true);
    return fetchJobs()
      .then((j) => {
        setJobs(j);
        setJobsError(null);
      })
      .catch((e) => setJobsError(String(e)))
      .finally(() => setJobsLoading(false));
  };

  useEffect(() => {
    fetchRuns().then((r) => {
      setRuns(r);
      if (r.length && !runId) setRunId(r[0].run_id);
    });
    fetchPipelines().then((p) => {
      setAllPipelines(p.order);
      setPipelines(p.order);
    });
    load();
  }, [runId]);

  useEffect(() => {
    if (!expandedLog) {
      wsRef.current?.close();
      wsRef.current = null;
      return;
    }
    const ws = new WebSocket(jobStreamUrl(expandedLog));
    wsRef.current = ws;
    ws.onmessage = (ev) => setLiveLog((prev) => prev + ev.data);
    return () => ws.close();
  }, [expandedLog]);

  const startJob = async () => {
    if (!runId) return;
    await postJob({
      run_id: runId,
      pipelines,
      overwrite: overwrite ? pipelines : undefined,
    });
    load();
  };

  return (
    <div className="panel jobs-panel">
      <h2>Pipeline jobs</h2>

      <div className="start-job-form">
        <h3>Start run</h3>
        <div className="form-row">
          <label>Run</label>
          <select value={runId} onChange={(e) => setRunId(e.target.value)}>
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>{r.run_id}</option>
            ))}
          </select>
        </div>
        <div className="pipeline-checklist">
          {allPipelines.map((p) => (
            <label key={p} className="pipeline-check">
              <input
                type="checkbox"
                checked={pipelines.includes(p)}
                onChange={() => {
                  setPipelines(
                    pipelines.includes(p) ? pipelines.filter((x) => x !== p) : [...pipelines, p]
                  );
                }}
              />
              <span>{p}</span>
            </label>
          ))}
        </div>
        <label className="pipeline-check">
          <input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} />
          Overwrite selected pipelines
        </label>
        <button className="btn" onClick={startJob}>Start job</button>
      </div>

      <button className="btn secondary" onClick={load}>Refresh list</button>
      {jobsLoading && <p className="meta-text">Loading jobs…</p>}
      {jobsError && <p className="error-text">{jobsError}</p>}
      {!jobsLoading && jobs.length === 0 && <p>No jobs yet.</p>}
      {jobs.map((j) => (
        <div key={j.job_id} className="job-card">
          <strong>{j.run_id}</strong> — {j.status} <code>{j.job_id}</code>
          {j.pipelines && <span className="meta-text"> ({j.pipelines.join(", ")})</span>}
          {j.status === "running" && (
            <button className="btn danger" style={{ marginLeft: 8 }} onClick={() => cancelJob(j.job_id).then(load)}>
              Stop
            </button>
          )}
          <button
            className="btn secondary"
            style={{ marginLeft: 8 }}
            onClick={() => {
              setExpandedLog(j.job_id);
              setLiveLog((j.log_tail || []).join("\n"));
            }}
          >
            Live log
          </button>
          <pre className="log-box log-box-tall">{(j.log_tail || []).join("\n")}</pre>
        </div>
      ))}

      {expandedLog && (
        <div className="modal-backdrop" onClick={() => setExpandedLog(null)}>
          <div className="modal modal-wide" onClick={(e) => e.stopPropagation()}>
            <h3>Job {expandedLog}</h3>
            <pre className="log-box log-box-live">{liveLog}</pre>
            <button className="btn secondary" onClick={() => setExpandedLog(null)}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
}
