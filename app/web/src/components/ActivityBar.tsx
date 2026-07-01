import { useGraph } from "../context/GraphStore";

type Props = {
  runId: string | null;
};

export default function ActivityBar({ runId }: Props) {
  const { liveByRun, jobs } = useGraph();

  if (!runId) return null;

  const live = liveByRun[runId];
  const jobRunning = jobs.some((j) => j.run_id === runId && j.status === "running");
  const isActive = jobRunning || Boolean(live?.active_step);

  if (!isActive) return null;

  let line = "";
  if (live?.live_headline) {
    line = live.live_headline;
  } else if (live?.active_step?.pipeline) {
    line = `${live.active_step.pipeline} · ${live.active_step.step_id}`;
  } else if (jobRunning) {
    line = "Job running…";
  } else {
    return null;
  }

  return (
    <div className="activity-bar">
      <strong>{runId}</strong>
      <span>{line}</span>
    </div>
  );
}
