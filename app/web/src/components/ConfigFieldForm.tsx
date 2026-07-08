export type ConfigField = {

  dot_key: string;

  section: string;

  key: string;

  label: string;

  help: string;

  field_type: string;

  enum_values: string[] | null;

  value: unknown;

};



export type ConfigFieldSection = {

  pipeline: string;

  step_id: string;

  title: string;

  fields: ConfigField[];

};



type Props = {

  fields?: ConfigField[];

  sections?: ConfigFieldSection[];

  values: Record<string, unknown>;

  onChange: (dotKey: string, value: unknown) => void;

  readOnly?: boolean;

};



export default function ConfigFieldForm({ fields, sections, values, onChange, readOnly }: Props) {

  const groups: ConfigFieldSection[] =

    sections ??

    (fields && fields.length > 0

      ? [{ pipeline: "", step_id: "", title: "", fields }]

      : []);



  if (groups.length === 0 || groups.every((g) => g.fields.length === 0)) {

    return <p className="meta-text">No editable config fields for this step.</p>;

  }



  return (

    <div className="config-fields">

      {groups.map((group) => (

        <div key={`${group.pipeline}:${group.step_id}`} className="config-section">

          {group.title ? <h4 className="config-section-title">{group.title}</h4> : null}

          {group.fields.map((f) => {

            const v = values[f.dot_key] ?? f.value;

            const isBool = f.field_type === "bool";

            return (

              <div key={f.dot_key} className={isBool ? "form-row form-row-bool" : "form-row"}>

                {isBool ? (

                  <>

                    <label title={f.help}>

                      <input

                        type="checkbox"

                        checked={Boolean(v)}

                        disabled={readOnly}

                        onChange={(e) => onChange(f.dot_key, e.target.checked)}

                      />

                      <span>{f.label}</span>

                    </label>

                    <span className="field-help">{f.help}</span>

                  </>

                ) : (

                  <>

                    <label htmlFor={f.dot_key} title={f.help}>{f.label}</label>

                    {f.field_type === "enum" && f.enum_values ? (

                      <select

                        id={f.dot_key}

                        value={String(v ?? "")}

                        disabled={readOnly}

                        onChange={(e) => onChange(f.dot_key, e.target.value)}

                      >

                        {f.enum_values.map((opt) => (

                          <option key={opt} value={opt}>

                            {opt}

                          </option>

                        ))}

                      </select>

                    ) : (

                      <input

                        id={f.dot_key}

                        type={f.field_type === "int" || f.field_type === "float" ? "number" : "text"}

                        value={v === null || v === undefined ? "" : String(v)}

                        disabled={readOnly}

                        onChange={(e) => {

                          let val: unknown = e.target.value;

                          if (f.field_type === "int") val = parseInt(e.target.value, 10);

                          else if (f.field_type === "float") val = parseFloat(e.target.value);

                          onChange(f.dot_key, val);

                        }}

                      />

                    )}

                    <span className="field-help">{f.help}</span>

                  </>

                )}

              </div>

            );

          })}

        </div>

      ))}

    </div>

  );

}


