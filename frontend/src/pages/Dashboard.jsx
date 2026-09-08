import { useEffect, useState } from "react";
import { Activity, AlertTriangle, ArrowUpRight, CheckCircle2, Plus, Waves } from "lucide-react";
import { Link } from "react-router-dom";
import StatCard from "../components/StatCard";
import SectionHeader from "../components/SectionHeader";
import Badge from "../components/Badge";
import { getDashboardStats, mockSurvey } from "../services/api";

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  useEffect(() => { getDashboardStats().then(setStats); }, []);
  if (!stats) return <div className="loading">Loading dashboard...</div>;
  return <div>
    <div className="page-heading"><div><span className="eyebrow">OVERVIEW</span><h1>Good morning, analyst.</h1><p>Monitor your underwater surveys and review detected anomalies.</p></div><Link className="primary-btn" to="/surveys/new"><Plus size={18} /> New survey</Link></div>
    <div className="stats-grid"><StatCard icon={Waves} label="Total surveys" value={stats.surveys} detail="+3 this month" tone="blue" /><StatCard icon={AlertTriangle} label="Detected anomalies" value={stats.anomalies} detail="Across all surveys" tone="amber" /><StatCard icon={Activity} label="High priority" value={stats.highPriority} detail="Needs review" tone="red" /><StatCard icon={CheckCircle2} label="Reviewed" value={`${Math.round(stats.reviewed / stats.anomalies * 100)}%`} detail={`${stats.reviewed} anomalies reviewed`} tone="green" /></div>
    <div className="content-grid two-one"><section className="panel"><SectionHeader title="Recent survey" subtitle="Latest processed mission" action={<Link className="text-link" to={`/surveys/${mockSurvey.id}`}>View details <ArrowUpRight size={15} /></Link>} /><div className="survey-row"><div className="survey-thumb"><Waves size={30} /></div><div className="survey-main"><div className="row-between"><strong>{mockSurvey.name}</strong><Badge tone="success">{mockSurvey.status}</Badge></div><p>{mockSurvey.location} · {mockSurvey.date}</p><div className="progress"><span style={{width: "100%"}} /></div><small>Processing complete · {mockSurvey.detections.length} anomalies found</small></div></div></section><section className="panel"><SectionHeader title="Priority queue" subtitle="Anomalies requiring attention" /><div className="priority-list">{mockSurvey.detections.map(item => <Link to={`/anomalies/${item.id}`} className="priority-item" key={item.id}><span className={`priority-dot ${item.priority.toLowerCase()}`} /><div><strong>{item.type}</strong><small>{item.id} · {item.confidence}% confidence</small></div><ArrowUpRight size={16} /></Link>)}</div></section></div>

  </div>;
}