export default function PageShell({
  title,
  kicker,
  children,
}: {
  title: string;
  kicker?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="sheet-wrap">
      {kicker ? <p className="kicker">{kicker}</p> : null}
      <h1 className="sheet-title">{title}</h1>
      <div className="sheet-body">{children}</div>
    </div>
  );
}
