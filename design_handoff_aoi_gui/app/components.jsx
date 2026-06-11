// ============================================================
// AOI Console — shared React components
// ============================================================

const { useState, useEffect, useRef, useCallback, useMemo } = React;

function Btn({ variant = "secondary", size, icon, children, ...rest }) {
  const cls = ["btn", `btn-${variant}`, size ? `btn-${size}` : ""].join(" ");
  return (
    <button className={cls} {...rest}>
      {icon}{children}
    </button>
  );
}

function IconBtn({ title, children, ...rest }) {
  return (
    <button className="icon-btn" title={title} {...rest}>{children}</button>
  );
}

function Chip({ icon, label, value, empty, onClick, title }) {
  return (
    <button className={"chip" + (empty ? " empty" : "")} onClick={onClick} title={title || value}>
      {icon}
      <span style={{ flexShrink: 0 }}>{label}</span>
      <span className="chip-value">{value}</span>
    </button>
  );
}

function Badge({ kind = "neutral", children }) {
  return <span className={`badge badge-${kind}`}>{children}</span>;
}

function ResultBadge({ result }) {
  if (result === "PASS") return <Badge kind="pass"><IcCheck size={12} strokeWidth={2.4} />PASS</Badge>;
  if (result === "NG") return <Badge kind="ng"><IcX size={12} strokeWidth={2.4} />NG</Badge>;
  return <Badge kind="neutral">—</Badge>;
}

function Segmented({ options, value, onChange }) {
  return (
    <div className="seg" role="tablist">
      {options.map((opt) => (
        <button
          key={opt.value}
          className={"seg-item" + (opt.value === value ? " active" : "")}
          onClick={() => onChange(opt.value)}
        >{opt.label}</button>
      ))}
    </div>
  );
}

function Panel({ title, actions, children, flush, style, className }) {
  return (
    <section className={"panel " + (className || "")} style={style}>
      {title !== undefined && (
        <header className="panel-header">
          <span className="panel-title">{title}</span>
          <div style={{ flex: 1 }}></div>
          {actions}
        </header>
      )}
      <div className={"panel-body" + (flush ? " flush" : "")} style={{ flex: 1 }}>{children}</div>
    </section>
  );
}

function FormGrid({ children }) {
  return <div className="form-grid">{children}</div>;
}

function FRow({ label, children }) {
  return (
    <React.Fragment>
      <label className="form-label">{label}</label>
      <div style={{ minWidth: 0 }}>{children}</div>
    </React.Fragment>
  );
}

function TextField({ mono, ...rest }) {
  return <input className={"field" + (mono ? " mono-field" : "")} {...rest} />;
}

function NumField({ value, onChange, step = 1, min, max, decimals }) {
  const clamp = (v) => {
    if (min !== undefined) v = Math.max(min, v);
    if (max !== undefined) v = Math.min(max, v);
    return decimals !== undefined ? +v.toFixed(decimals) : v;
  };
  const bump = (dir) => onChange(clamp((+value || 0) + dir * step));
  return (
    <div className="num-field">
      <input
        className="field mono-field"
        value={value}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          onChange(isNaN(v) ? e.target.value : v);
        }}
        onBlur={(e) => {
          const v = parseFloat(e.target.value);
          if (!isNaN(v)) onChange(clamp(v));
        }}
      />
      <div className="num-steps">
        <button onClick={() => bump(1)} tabIndex={-1}>▲</button>
        <button onClick={() => bump(-1)} tabIndex={-1}>▼</button>
      </div>
    </div>
  );
}

function Toggle({ value, onChange, disabled }) {
  return (
    <button
      className={"toggle" + (value ? " on" : "")}
      onClick={() => !disabled && onChange(!value)}
      style={disabled ? { opacity: 0.5, cursor: "default" } : null}
      role="switch" aria-checked={value}
    ></button>
  );
}

// 依參數型別自動選擇控件（對應 PySide 的 _make_param_widget）
function ParamControl({ value, onChange, readOnly }) {
  if (typeof value === "boolean") {
    return <Toggle value={value} onChange={onChange} disabled={readOnly} />;
  }
  if (typeof value === "number") {
    if (readOnly) return <TextField mono value={String(value)} readOnly />;
    const isFloat = !Number.isInteger(value) || Math.abs(value) <= 1;
    return <NumField value={value} onChange={onChange} step={isFloat ? 0.01 : 1} decimals={isFloat ? 3 : 0} />;
  }
  return <TextField mono value={String(value)} readOnly={readOnly} onChange={(e) => onChange && onChange(e.target.value)} />;
}

function ProgressBar({ pct }) {
  return (
    <div className="progress-track">
      <div className="progress-fill" style={{ width: `${pct}%` }}></div>
    </div>
  );
}

function Drawer({ title, onClose, children, width }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <React.Fragment>
      <div className="overlay-dim" onClick={onClose}></div>
      <aside className="drawer" style={width ? { width } : null}>
        <header className="drawer-header">
          <span className="drawer-title">{title}</span>
          <IconBtn title="關閉" onClick={onClose}><IcX size={16} /></IconBtn>
        </header>
        <div className="drawer-body">{children}</div>
      </aside>
    </React.Fragment>
  );
}

function EmptyState({ icon, title, hint, action }) {
  return (
    <div className="empty-state">
      <div style={{ opacity: 0.55 }}>{icon}</div>
      <div style={{ fontWeight: 600, color: "var(--text-2)", fontSize: 13 }}>{title}</div>
      {hint && <div style={{ fontSize: "var(--fs-small)", maxWidth: 260, lineHeight: 1.55 }}>{hint}</div>}
      {action}
    </div>
  );
}

Object.assign(window, {
  Btn, IconBtn, Chip, Badge, ResultBadge, Segmented, Panel, FormGrid, FRow,
  TextField, NumField, Toggle, ParamControl, ProgressBar, Drawer, EmptyState,
});
