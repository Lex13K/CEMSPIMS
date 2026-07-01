import { useCallback, useEffect, useMemo, useState } from "react";
import ConfigFieldForm, { type ConfigFieldSection } from "./ConfigFieldForm";
import PipelineChecklist from "./PipelineChecklist";
import { fetchBranchFields, postBranchAndRun, previewConfigPatch } from "../api";
import type { GraphNode } from "../api";

type Props = {
  node: GraphNode;
  onClose: () => void;
  onDone: (childId: string) => void;
};

function parseApiError(err: unknown): string {
  const raw = String(err);
  try {
    const parsed = JSON.parse(raw) as { detail?: string | { msg: string }[] };
    if (typeof parsed.detail === "string") return parsed.detail;
    if (Array.isArray(parsed.detail)) {
      return parsed.detail.map((d) => d.msg).join("; ");
    }
  } catch {
    /* use raw */
  }
  return raw;
}

export default function BranchRunWizard({ node, onClose, onDone }: Props) {
  const [childId, setChildId] = useState("");
  const [childIdTouched, setChildIdTouched] = useState(false);
  const [sections, setSections] = useState<ConfigFieldSection[]>([]);
  const [baseline, setBaseline] = useState<Record<string, unknown>>({});
  const [patch, setPatch] = useState<Record<string, unknown>>({});
  const [pipelines, setPipelines] = useState<string[]>([]);
  const [pipelineOptions, setPipelineOptions] = useState<string[]>([]);
  const [diff, setDiff] = useState<string[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const parent = node.run_id!;
  const atPipeline = node.pipeline;
  const atStepId = node.step_id;

  const allFields = useMemo(() => sections.flatMap((s) => s.fields), [sections]);
  const trimmedChildId = childId.trim();
  const childIdMissing = trimmedChildId.length === 0;

  useEffect(() => {
    setLoading(true);
    fetchBranchFields(node.id)
      .then((r) => {
        const secs = (r.fields_by_step as ConfigFieldSection[]) || [];
        setSections(secs);
        setPipelineOptions(r.pipelines_from_here as string[]);
        setPipelines(r.pipelines_from_here as string[]);
        const init: Record<string, unknown> = {};
        for (const sec of secs) {
          for (const f of sec.fields) {
            init[f.dot_key] = f.value;
          }
        }
        setBaseline(init);
        setPatch(init);
      })
      .catch((e) => setError(parseApiError(e)))
      .finally(() => setLoading(false));
  }, [node.id]);

  const patchChanges = useCallback(() => {
    const out: Record<string, unknown> = {};
    for (const f of allFields) {
      const cur = patch[f.dot_key];
      const base = baseline[f.dot_key];
      if (cur === undefined || cur === base) continue;
      if (f.field_type === "int" && typeof cur === "number" && Number(base) === cur) continue;
      if (f.field_type === "float" && typeof cur === "number" && Number(base) === cur) continue;
      out[f.dot_key] = cur;
    }
    return out;
  }, [allFields, patch, baseline]);

  useEffect(() => {
    if (!trimmedChildId) {
      setDiff([]);
      setWarnings([]);
      return;
    }
    const t = setTimeout(() => {
      previewConfigPatch(parent, { parent, child: trimmedChildId, config_patch: patchChanges() })
        .then((r) => {
          setDiff(r.diff || []);
          setWarnings(r.warnings || []);
        })
        .catch(() => {});
    }, 400);
    return () => clearTimeout(t);
  }, [patchChanges, trimmedChildId, parent]);

  const confirm = async () => {
    setChildIdTouched(true);
    if (childIdMissing) {
      setError("Enter a child run id (e.g. v2_test) before creating the branch.");
      return;
    }
    if (pipelines.length === 0) {
      setError("Select at least one pipeline to run.");
      return;
    }
    setBusy(true);
    setError(null);
    setSuccess(null);
    try {
      const result =       await postBranchAndRun({
        parent,
        child: trimmedChildId,
        at_pipeline: atPipeline,
        at_step_id: atStepId,
        config_patch: patchChanges(),
        pipelines,
        overwrite: [atPipeline],
      });
      const jobId = (result.job as { job_id?: string } | undefined)?.job_id;
      setSuccess(
        jobId
          ? `Branch "${trimmedChildId}" created. Job ${jobId} started.`
          : `Branch "${trimmedChildId}" created.`
      );
      onDone(trimmedChildId);
      setTimeout(() => onClose(), 600);
    } catch (e) {
      setError(parseApiError(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="modal-backdrop">
      <div className="modal modal-wide branch-panel">
        <div className="branch-panel-header">
          <h2>Branch from {node.label}</h2>
          <button type="button" className="btn secondary" onClick={onClose}>Close</button>
        </div>
        <p className="wizard-sub">
          Parent <strong>{parent}</strong> · branch at <strong>{atPipeline}/{atStepId}</strong>
        </p>
        {error && <p className="error-banner">{error}</p>}
        {success && <p className="success-banner">{success}</p>}

        <div className="branch-panel-body">
          <div className="form-row">
            <label htmlFor="child-run-id">Child run id (required)</label>
            <input
              id="child-run-id"
              value={childId}
              onChange={(e) => {
                setChildId(e.target.value);
                setChildIdTouched(true);
                setError(null);
              }}
              placeholder="v2_test"
              autoFocus
              className={childIdTouched && childIdMissing ? "input-invalid" : undefined}
            />
            {childIdTouched && childIdMissing && (
              <span className="error-text">Required — pick a new run name (letters, digits, _, -).</span>
            )}
          </div>

          <h3 className="section-heading">Config changes</h3>
          <p className="field-help">
            Only <strong>{node.label}</strong> and later steps are editable here.
          </p>
          {loading ? (
            <p className="meta-text">Loading config fields…</p>
          ) : sections.length === 0 ? (
            <p className="meta-text">No editable fields from this step onward.</p>
          ) : (
            <ConfigFieldForm
              sections={sections}
              values={patch}
              onChange={(k, v) => setPatch((p) => ({ ...p, [k]: v }))}
            />
          )}

          {diff.length > 0 && (
            <>
              <h3 className="section-heading">Diff vs parent</h3>
              <pre className="diff-preview">{diff.join("\n")}</pre>
            </>
          )}
          {warnings.map((w, i) => (
            <p key={i} className="warn-text">{w}</p>
          ))}

          <h3 className="section-heading">Pipelines to run</h3>
          <p className="field-help">Unchecked pipelines are omitted from the child graph.</p>
          <PipelineChecklist
            pipelines={pipelineOptions}
            selected={pipelines}
            lockedFrom={atPipeline}
            onChange={setPipelines}
          />
        </div>

        <div className="wizard-actions branch-panel-footer">
          <button type="button" className="btn secondary" onClick={onClose}>Cancel</button>
          <button
            type="button"
            className="btn"
            disabled={busy}
            onClick={confirm}
          >
            {busy ? "Starting…" : "Create branch & run"}
          </button>
        </div>
      </div>
    </div>
  );
}
