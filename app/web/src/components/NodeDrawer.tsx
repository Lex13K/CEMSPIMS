import { useEffect, useState } from "react";

import { Link } from "react-router-dom";

import ConfigFieldForm, { type ConfigField } from "./ConfigFieldForm";

import { deleteRun, fetchNodeArtifacts, fetchNodeConfigFields, postJob } from "../api";

import { useGraph } from "../context/GraphStore";

import { statusLabel } from "./nodeStyles";

type Detail = Record<string, unknown>;

type Props = {
  detail: Detail;
  onBranch: () => void;
  onExtend: () => void;
  onClose: () => void;
  onJobStarted?: (runId: string) => void;
  onRunDeleted?: (runId: string) => void;
  onRunFocus?: (runId: string) => void;
};

export default function NodeDrawer({
  detail,
  onBranch,
  onExtend,
  onClose,
  onJobStarted,
  onRunDeleted,
  onRunFocus,
}: Props) {
  const { liveByRun } = useGraph();
  const [tab, setTab] = useState<"config" | "artifacts">("config");
  const [artifacts, setArtifacts] = useState<unknown[] | null>(null);
  const [loadingArt, setLoadingArt] = useState(false);
  const [configFields, setConfigFields] = useState<ConfigField[]>([]);
  const [loadingConfig, setLoadingConfig] = useState(false);
  const [configScope, setConfigScope] = useState<string[] | null>(null);
  const [deleting, setDeleting] = useState(false);

  const canBranch = Boolean(detail.can_branch);
  const canRun = Boolean(detail.can_run);
  const canDelete = Boolean(detail.can_delete_run ?? detail.parent_run_id);
  const isShared = Boolean(detail.is_shared);
  const runId = detail.run_id as string | undefined;
  const pipeline = detail.pipeline as string;
  const stepId = detail.step_id as string;
  const scope = (configScope ?? detail.pipeline_scope) as string[] | null | undefined;
  const displayStatus = String(detail.status ?? "missing");
  const nodeId = String(detail.node_id ?? "");

  const live = runId ? liveByRun[runId] : undefined;
  const activeStep = live?.active_step;
  const isActiveStep =
    activeStep?.pipeline === pipeline && activeStep?.step_id === stepId;
  const showLive = isActiveStep && (live?.live_log?.length || live?.live_headline);

  useEffect(() => {
    setTab("config");
    setArtifacts(null);
    setConfigFields([]);
    if (runId) onRunFocus?.(runId);
  }, [nodeId, runId, onRunFocus]);

  useEffect(() => {
    if (!nodeId) return;
    setLoadingConfig(true);
    fetchNodeConfigFields(nodeId)
      .then((r) => {
        setConfigFields((r.fields as ConfigField[]) || []);
        if (r.pipelines_from_here) {
          setConfigScope(r.pipelines_from_here as string[]);
        }
      })
      .catch(() => setConfigFields([]))
      .finally(() => setLoadingConfig(false));
  }, [nodeId]);

  const loadArtifacts = () => {
    if (artifacts !== null || !nodeId) return;
    setLoadingArt(true);
    fetchNodeArtifacts(nodeId)
      .then((r) => setArtifacts(r.artifacts))
      .finally(() => setLoadingArt(false));
  };

  const rerun = async (overwrite: boolean) => {
    if (!runId) return;
    await postJob({
      run_id: runId,
      pipelines: [pipeline],
      overwrite: overwrite ? [pipeline] : undefined,
    });
    onJobStarted?.(runId);
  };

  const removeRun = async () => {
    if (!runId || !canDelete) return;
    const ok = window.confirm(`Delete run "${runId}" and all its data? This cannot be undone.`);
    if (!ok) return;
    setDeleting(true);
    onRunDeleted?.(runId);
    onClose();
    try {
      await deleteRun(runId);
    } catch (e) {
      window.alert(String(e));
    } finally {
      setDeleting(false);
    }
  };

  const readOnlyValues: Record<string, unknown> = {};
  for (const f of configFields) readOnlyValues[f.dot_key] = f.value;

  return (
    <div className="drawer">
      <div className="drawer-header-row">
        <button className="btn secondary" onClick={onClose}>Close</button>
        {canDelete && runId && (
          <button className="btn danger" onClick={removeRun} disabled={deleting}>
            {deleting ? "Deleting…" : `Delete ${runId}`}
          </button>
        )}
      </div>
      <h2>{String(detail.title)}</h2>
      {detail.description ? <p className="drawer-desc">{String(detail.description)}</p> : null}
      <p>
        Status:{" "}
        <span className={`status-pill status-pill-${displayStatus}`}>{statusLabel(displayStatus)}</span>
        {detail.inherited_from ? ` (inherited from ${String(detail.inherited_from)})` : ""}
      </p>
      {detail.error ? <p className="error-text">{String(detail.error)}</p> : null}
      {scope && <p className="meta-text">Scope: {scope.join(" → ")}</p>}

      {showLive && (
        <div className="live-train-box">
          <h3 className="section-heading">Live output</h3>
          {live?.live_headline && (
            <p className="live-train-metric">
              <strong>{live.live_headline}</strong>
            </p>
          )}
          <div className="log-box live-log-scroll">
            {(live?.live_log || []).slice(-12).map((line, i) => (
              <div key={i} className="live-log-line">{line}</div>
            ))}
          </div>
        </div>
      )}

      <div className="tab-row">
        <button className={tab === "config" ? "tab active" : "tab"} onClick={() => setTab("config")}>Config</button>
        <button
          className={tab === "artifacts" ? "tab active" : "tab"}
          onClick={() => {
            setTab("artifacts");
            loadArtifacts();
          }}
        >
          Artifacts
        </button>
      </div>

      {tab === "config" && (
        <>
          {loadingConfig && <p className="meta-text">Loading config…</p>}
          {!loadingConfig && configFields.length > 0 ? (
            <ConfigFieldForm fields={configFields} values={readOnlyValues} onChange={() => {}} readOnly />
          ) : !loadingConfig ? (
            <p className="meta-text">No config fields for this step.</p>
          ) : null}
          {canBranch && (
            <p className="field-help" style={{ marginTop: 8 }}>
              To change values and run a new branch, use Branch below.
            </p>
          )}
        </>
      )}

      {tab === "artifacts" && (
        <div>
          {loadingArt && <p>Loading…</p>}
          {artifacts?.map((a, i) => (
            <ArtifactPreview key={i} artifact={a as Record<string, unknown>} />
          ))}
          {artifacts && artifacts.length === 0 && <p>No artifacts yet.</p>}
        </div>
      )}

      {!isShared && canBranch && (
        <button className="btn" onClick={onBranch}>Branch…</button>
      )}
      {!isShared && canRun && (
        <>
          <button className="btn secondary" onClick={() => rerun(false)}>Re-run pipeline</button>
          <button className="btn secondary" onClick={() => rerun(true)}>Overwrite &amp; re-run</button>
        </>
      )}
      {!isShared && runId && scope && (
        <button className="btn secondary" onClick={onExtend}>Extend run…</button>
      )}
      {runId && (
        <p style={{ marginTop: 12 }}>
          <Link to="/config">Edit full config</Link> · <Link to="/jobs">Jobs</Link>
        </p>
      )}
    </div>
  );
}

function ArtifactPreview({ artifact }: { artifact: Record<string, unknown> }) {
  if (artifact.type === "image" && artifact.data_url) {
    return <img src={String(artifact.data_url)} alt="" style={{ maxWidth: "100%" }} />;
  }
  if (artifact.type === "table" && artifact.columns) {
    const cols = artifact.columns as string[];
    const rows = artifact.rows as unknown[][];
    return (
      <table className="preview">
        <thead>
          <tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows?.map((row, ri) => (
            <tr key={ri}>{row.map((cell, ci) => <td key={ci}>{String(cell)}</td>)}</tr>
          ))}
        </tbody>
      </table>
    );
  }
  if (artifact.path) return <code>{String(artifact.path)}</code>;
  return <pre className="log-box">{JSON.stringify(artifact, null, 2)}</pre>;
}
