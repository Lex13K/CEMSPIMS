import { useEffect, useState } from "react";
import { fetchRuns, getConfig, patchConfig } from "../api";
import { useGraph } from "../context/GraphStore";

export default function ConfigPage() {
  const { refresh } = useGraph();
  const [runs, setRuns] = useState<{ run_id: string }[]>([]);
  const [runId, setRunId] = useState("");
  const [content, setContent] = useState("");
  const [warnings, setWarnings] = useState<string[]>([]);
  const [msg, setMsg] = useState<string | null>(null);

  useEffect(() => {
    fetchRuns().then((r) => {
      setRuns(r);
      if (r.length && !runId) setRunId(r[0].run_id);
    });
  }, [runId]);

  useEffect(() => {
    if (!runId) return;
    getConfig(runId)
      .then((r) => setContent(r.content))
      .catch((e) => setMsg(String(e)));
  }, [runId]);

  const save = async () => {
    const r = await patchConfig(runId, content);
    setWarnings(r.warnings || []);
    setMsg("Saved");
    await refresh();
  };

  return (
    <div className="panel">
      <h2>Config editor</h2>
      <div className="form-row">
        <label>Run</label>
        <select value={runId} onChange={(e) => setRunId(e.target.value)}>
          {runs.map((r) => (
            <option key={r.run_id} value={r.run_id}>{r.run_id}</option>
          ))}
        </select>
      </div>
      <div className="form-row">
        <textarea rows={24} value={content} onChange={(e) => setContent(e.target.value)} style={{ fontFamily: "monospace", fontSize: 12 }} />
      </div>
      <button className="btn" onClick={save}>Save & validate</button>
      {msg && <p>{msg}</p>}
      {warnings.length > 0 && (
        <ul>{warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
      )}
    </div>
  );
}
