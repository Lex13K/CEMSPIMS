import { useCallback, useEffect, useMemo, useState } from "react";

import {

  ReactFlow,

  Background,

  Controls,

  MiniMap,

  useReactFlow,

  type Edge,

  type Node,

  MarkerType,

} from "@xyflow/react";

import "@xyflow/react/dist/style.css";

import { fetchEdgeDetail } from "../api";

import type { GraphEdge, GraphNode } from "../api";

import { useGraph } from "../context/GraphStore";

import ActivityBar from "../components/ActivityBar";

import BranchRunWizard from "../components/BranchRunWizard";

import ExtendRunWizard from "../components/ExtendRunWizard";

import NodeDrawer from "../components/NodeDrawer";

import { statusBadgeColor, toFlowNodes } from "../components/nodeStyles";



function toFlowEdges(edges: GraphEdge[]): Edge[] {

  return edges.map((e) => ({

    id: e.id,

    source: e.source,

    target: e.target,

    type: e.kind === "branch" ? "smoothstep" : "straight",

    animated: e.kind === "branch",

    style: { stroke: e.kind === "branch" ? "#f59e0b" : "#94a3b8", strokeWidth: 2 },

    markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16, color: e.kind === "branch" ? "#f59e0b" : "#94a3b8" },

    data: e,

  }));

}



function shellDetail(node: GraphNode): Record<string, unknown> {
  return {
    node_id: node.id,
    title: node.label,
    description: "",
    pipeline: node.pipeline,
    step_id: node.step_id,
    run_id: node.run_id,
    status: node.status,
    config_fields: [],
    can_branch: node.kind === "run_step" && Boolean(node.run_id),
    can_run: node.kind === "run_step" && Boolean(node.run_id),
    can_delete_run: node.kind === "run_step" && Boolean(node.run_id),
    is_shared: node.kind === "raw" || node.kind === "global_step",
  };
}



function FitViewOnTarget({ targetRunId }: { targetRunId: string | null }) {

  const { fitView } = useReactFlow();

  const { nodes, clearFitTarget } = useGraph();



  useEffect(() => {

    if (!targetRunId) return;

    const ids = nodes.filter((n) => n.run_id === targetRunId).map((n) => ({ id: n.id }));

    if (ids.length) {

      setTimeout(() => {

        fitView({ nodes: ids, padding: 0.3, duration: 400 });

        clearFitTarget();

      }, 100);

    }

  }, [targetRunId, nodes, fitView, clearFitTarget]);

  return null;

}



export default function GraphPage() {

  const { nodes: graphNodes, edges: graphEdges, colors, loading, error, refresh, lastFitTarget, optimisticRemoveRun, rescan } = useGraph();

  const [flowNodes, setFlowNodes] = useState<Node[]>([]);

  const [flowEdges, setFlowEdges] = useState<Edge[]>([]);

  const [selected, setSelected] = useState<string | null>(null);

  const [detail, setDetail] = useState<Record<string, unknown> | null>(null);

  const [edgeDetail, setEdgeDetail] = useState<Record<string, unknown> | null>(null);

  const [wizardNode, setWizardNode] = useState<GraphNode | null>(null);

  const [extendNode, setExtendNode] = useState<GraphNode | null>(null);

  const [activityRun, setActivityRun] = useState<string | null>(null);

  useEffect(() => {
    setFlowNodes(
      toFlowNodes(graphNodes).map((n) => {
        const gn = (n.data as { node: GraphNode }).node;
        const badge = statusBadgeColor(gn.status);
        return {
          ...n,
          data: {
            ...n.data,
            label: (
              <div className="node-label">
                <span className="node-status-dot" style={{ background: badge }} title={gn.status} />
                <span>{gn.label}</span>
              </div>
            ),
          },
        };
      })
    );
    setFlowEdges(toFlowEdges(graphEdges));
  }, [graphNodes, graphEdges]);

  const onNodeClick = useCallback((_: unknown, node: Node) => {
    const gn = (node.data as { node: GraphNode }).node;
    setSelected(node.id);
    setEdgeDetail(null);
    if (gn.run_id) setActivityRun(gn.run_id);
    setDetail(shellDetail(gn));
  }, []);



  const onEdgeClick = useCallback(async (_: unknown, edge: Edge) => {

    try {

      const d = await fetchEdgeDetail(edge.id);

      setEdgeDetail(d);

      setDetail(null);

      setSelected(null);

    } catch (e) {

      setEdgeDetail({ error: String(e) });

    }

  }, []);



  const selectedNode = useMemo(

    () => graphNodes.find((n) => n.id === selected) || null,

    [selected, graphNodes]

  );



  const handleBranchDone = (childId: string) => {

    refresh({ fitTarget: childId, silent: true });

    setActivityRun(childId);

  };



  const handleRerun = (runId: string) => {

    setActivityRun(runId);

  };



  return (

    <>

      <ActivityBar runId={activityRun} />

      <div className="graph-toolbar">
        <button type="button" className="btn secondary" onClick={() => rescan()}>
          Rescan from disk
        </button>
      </div>

      {loading && <div className="graph-loading">Updating graph…</div>}

      <div style={{ height: "100%", width: "100%" }}>

        <ReactFlow

          nodes={flowNodes}

          edges={flowEdges}

          onNodeClick={onNodeClick}

          onEdgeClick={onEdgeClick}

          nodesDraggable={false}

          nodesConnectable={false}

          elementsSelectable

          fitView

          fitViewOptions={{ padding: 0.2 }}

          minZoom={0.15}

          maxZoom={1.5}

          proOptions={{ hideAttribution: true }}

        >

          <FitViewOnTarget targetRunId={lastFitTarget} />

          <Background color="#334155" gap={16} />

          <Controls />

          <MiniMap />

        </ReactFlow>

        <div className="graph-legend">

          {Object.entries(colors).map(([p, c]) => (

            <span key={p} className="legend-item">

              <span className="legend-stripe" style={{ background: c }} /> {p}

            </span>

          ))}

          <span className="legend-item">

            <span className="legend-dot" style={{ background: "#22c55e" }} /> complete

          </span>

          <span className="legend-item">

            <span className="legend-dot" style={{ background: "#3b82f6" }} /> running

          </span>

        </div>

      </div>



      {detail && !edgeDetail && (

        <NodeDrawer

          detail={detail}

          onClose={() => {

            setDetail(null);

            setSelected(null);

          }}

          onBranch={() => setWizardNode(selectedNode)}

          onExtend={() => setExtendNode(selectedNode)}

          onJobStarted={handleRerun}

          onRunDeleted={(rid) => {
            optimisticRemoveRun(rid);
            setDetail(null);
            setSelected(null);
            setActivityRun((cur) => (cur === rid ? null : cur));
          }}

          onRunFocus={setActivityRun}

        />

      )}



      {edgeDetail && (

        <div className="drawer">

          <button className="btn secondary" onClick={() => setEdgeDetail(null)}>Close</button>

          <h2>Branch diff</h2>

          <pre className="log-box">

            {(edgeDetail.diff as string[] | undefined)?.join("\n") || JSON.stringify(edgeDetail, null, 2)}

          </pre>

        </div>

      )}



      {wizardNode && (

        <BranchRunWizard

          node={wizardNode}

          onClose={() => setWizardNode(null)}

          onDone={handleBranchDone}

        />

      )}



      {extendNode && detail && (

        <ExtendRunWizard

          node={extendNode}

          scope={(detail.pipeline_scope as string[] | null) ?? null}

          onClose={() => setExtendNode(null)}

          onDone={() => refresh()}

        />

      )}



      {error && <div className="graph-error">{error}</div>}

    </>

  );

}


