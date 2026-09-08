import { Download, FileText, FileSpreadsheet, FileJson, Plus } from "lucide-react";
import { Link } from "react-router-dom";
import SectionHeader from "../components/SectionHeader";
import Badge from "../components/Badge";
import { mockSurvey } from "../services/api";

export default function Reports() { return <div><div className="page-heading"><div><span className="eyebrow">REPORTS</span><h1>Survey reports</h1><p>Export and share validated survey findings.</p></div><Link to="/surveys/new" className="primary-btn"><Plus size={18}/> New survey</Link></div><section className="panel"><SectionHeader title="Available reports" subtitle="Generated from processed surveys"/><div className="report-list"><Report icon={FileText} name={`${mockSurvey.name} — Full report`} type="PDF" meta="Generated today · 1.2 MB"/><Report icon={FileSpreadsheet} name={`${mockSurvey.name} — Detection data`} type="CSV" meta="Generated today · 18 KB"/><Report icon={FileJson} name={`${mockSurvey.name} — Machine-readable results`} type="JSON" meta="Generated today · 9 KB"/></div></section></div>; }
function Report({icon:Icon,name,type,meta}) { return <div className="report-row"><div className="report-icon"><Icon size={22}/></div><div><strong>{name}</strong><p>{meta}</p></div><Badge>{type}</Badge><button className="secondary-btn small"><Download size={16}/> Download</button></div>; }