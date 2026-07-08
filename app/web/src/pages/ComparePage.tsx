import { useEffect, useState } from "react";
import { fetchEligibleCompare, postCompare } from "../api";

export default function ComparePage() {
  const [eligible, setEligible] = useState<string[]>([]);
  const [anchor, setAnchor] = useState("");
  const [peers, setPeers] = useState<string[]>([]);
  const [comparisonId, setComparisonId] = useState("v2_ablations");
  const [preset, setPreset] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchEligibleCompare()
      .then((ids) => {
        setEligible(ids);
        if (ids.length && !anchor) setAnchor(ids[0]);
      })
      .catch((e) => setError(String(e)));
  }, [anchor]);

  const togglePeer = (id: string) => {
    setPeers((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]));
  };

  const build = async () => {
    setError(null);
    try {
      const body = preset
        ? { anchor: "", peer_runs: [], comparison_id: comparisonId, preset }
        : { anchor, peer_runs: peers, comparison_id: comparisonId };
      const r = await postCompare(body);
      setResult(JSON.stringify(r, null, 2));
    } catch (e) {
      setError(String(e));
    }
  };

  return (
    <div className="panel">
      <h2>Cross-run comparison</h2>
      <p>Only runs with completed <code>analysis.summarize</code> are listed.</p>
      {error && <p style={{ color: "#f87171" }}>{error}</p>}

      <div className="form-row">
        <label>Preset (optional)</label>
        <select value={preset} onChange={(e) => setPreset(e.target.value)}>
          <option value="">— manual —</option>
          <option value="v2_ablations">v2_ablations</option>
        </select>
      </div>

      {!preset && (
        <>
          <div className="form-row">
            <label>Anchor run</label>
            <select value={anchor} onChange={(e) => setAnchor(e.target.value)}>
              {eligible.map((id) => (
                <option key={id} value={id}>{id}</option>
              ))}
            </select>
          </div>
          <div className="form-row">
            <label>Peer runs</label>
            {eligible.filter((id) => id !== anchor).map((id) => (
              <label key={id} style={{ display: "block" }}>
                <input type="checkbox" checked={peers.includes(id)} onChange={() => togglePeer(id)} />
                {" "}{id}
              </label>
            ))}
          </div>
        </>
      )}

      <div className="form-row">
        <label>Comparison id</label>
        <input value={comparisonId} onChange={(e) => setComparisonId(e.target.value)} />
      </div>

      <button className="btn" onClick={build}>Build comparison</button>

      {result && <pre className="log-box" style={{ marginTop: 16 }}>{result}</pre>}
    </div>
  );
}
