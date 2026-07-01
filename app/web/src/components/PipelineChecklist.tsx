type Props = {
  pipelines: string[];
  selected: string[];
  lockedFrom: string;
  onChange: (selected: string[]) => void;
};

export default function PipelineChecklist({ pipelines, selected, lockedFrom, onChange }: Props) {
  const lockedIdx = pipelines.indexOf(lockedFrom);
  const toggle = (p: string) => {
    const idx = pipelines.indexOf(p);
    if (idx < lockedIdx) return;
    if (p === lockedFrom) return;
    if (selected.includes(p)) {
      onChange(selected.filter((x) => x !== p));
    } else {
      onChange([...selected, p].sort((a, b) => pipelines.indexOf(a) - pipelines.indexOf(b)));
    }
  };

  return (
    <div className="pipeline-checklist">
      {pipelines.map((p) => {
        const idx = pipelines.indexOf(p);
        const required = idx >= lockedIdx;
        const checked = selected.includes(p);
        const disabled = p === lockedFrom;
        return (
          <label key={p} className="pipeline-check">
            <input
              type="checkbox"
              checked={checked}
              disabled={disabled}
              onChange={() => toggle(p)}
            />
            <span>
              {p}
              {disabled ? " (required)" : !required ? " (upstream)" : ""}
            </span>
          </label>
        );
      })}
    </div>
  );
}
