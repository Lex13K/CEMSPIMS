import { useEffect, useState } from "react";
import PipelineChecklist from "./PipelineChecklist";
import { extendScope, fetchPipelines } from "../api";
import type { GraphNode } from "../api";

type Props = {
  node: GraphNode;
  scope: string[] | null;
  onClose: () => void;
  onDone: () => void;
};

export default function ExtendRunWizard({ node, scope, onClose, onDone }: Props) {
  const [allPipelines, setAllPipelines] = useState<string[]>([]);
  const [add, setAdd] = useState<string[]>([]);
  const [runSelected, setRunSelected] = useState<string[]>([]);
  const [forceOverwrite, setForceOverwrite] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runId = node.run_id!;

  useEffect(() => {
    fetchPipelines().then((r) => setAllPipelines(r.order));
  }, []);

  const existing = new Set(scope || allPipelines);
  const available = allPipelines.filter((p) => !existing.has(p));

  const confirm = async () => {
    if (add.length === 0) return;
    setBusy(true);
    try {
      await extendScope(runId, {
        add_pipelines: add,
        pipelines_to_run: runSelected.length ? runSelected : null,
        overwrite: forceOverwrite && runSelected.length ? runSelected : undefined,
      });
      onDone();
      onClose();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  if (available.length === 0) {
    return (
      <div className="modal-backdrop">
        <div className="modal">
          <h2>Extend run</h2>
          <p>All pipelines are already in scope for {runId}.</p>
          <button className="btn secondary" onClick={onClose}>Close</button>
        </div>
      </div>
    );
  }

  return (
    <div className="modal-backdrop">
      <div className="modal modal-wide">
        <h2>Extend run</h2>
        <p className="wizard-sub">Add pipelines to <strong>{runId}</strong>.</p>
        {error && <p className="error-text">{error}</p>}
        <h3>Add to graph</h3>
        <div className="pipeline-checklist">
          {available.map((p) => (
            <label key={p} className="pipeline-check">
              <input
                type="checkbox"
                checked={add.includes(p)}
                onChange={() => {
                  const next = add.includes(p) ? add.filter((x) => x !== p) : [...add, p];
                  setAdd(next);
                  setRunSelected(next);
                }}
              />
              <span>{p}</span>
            </label>
          ))}
        </div>
        {add.length > 0 && (
          <>
            <h3>Run now</h3>
            <PipelineChecklist
              pipelines={add}
              selected={runSelected}
              lockedFrom={add[0]}
              onChange={setRunSelected}
            />
            <label className="pipeline-check">
              <input
                type="checkbox"
                checked={forceOverwrite}
                onChange={() => setForceOverwrite((v) => !v)}
              />
              <span>Overwrite selected (force re-run even if outputs exist)</span>
            </label>
          </>
        )}
        <div className="wizard-actions">
          <button className="btn secondary" onClick={onClose}>Cancel</button>
          <button className="btn" disabled={busy || add.length === 0} onClick={confirm}>
            {busy ? "Saving…" : "Extend & run"}
          </button>
        </div>
      </div>
    </div>
  );
}
