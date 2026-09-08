import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { CalendarDays, ChevronRight, FileImage, MapPin, UploadCloud, X } from "lucide-react";
import SectionHeader from "../components/SectionHeader";

export default function NewSurvey() {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [form, setForm] = useState({ name: "", location: "", date: new Date().toISOString().slice(0,10), notes: "" });
  function chooseFile(e) { const selected = e.target.files?.[0]; if (selected) setFile(Object.assign(selected, { preview: URL.createObjectURL(selected) })); }
  function submit(e) { e.preventDefault(); navigate("/processing", { state: { file, form } }); }
  return <div><div className="page-heading"><div><span className="eyebrow">NEW SURVEY</span><h1>Create a survey</h1><p>Upload side-scan sonar imagery and start an analysis.</p></div></div><form onSubmit={submit} className="form-layout"><section className="panel"><SectionHeader title="Survey information" subtitle="Basic details for this mission" /><div className="form-grid">
    <label>Survey name<input required value={form.name} onChange={e => setForm({...form, name:e.target.value})} placeholder="e.g. Bay Area Survey" />
   <div className="input-icon">
  </div></label><label>Survey date<div className="input-icon"><CalendarDays size={17}/><input type="date" required value={form.date} onChange={e => setForm({...form, date:e.target.value})} /></div></label><label>Notes<textarea value={form.notes} onChange={e => setForm({...form, notes:e.target.value})} placeholder="Optional notes about the mission..." /></label></div></section><section className="panel"><SectionHeader title="Sonar imagery" subtitle="Upload a JPG, PNG, or TIFF file" /><label className={`dropzone ${file ? "has-file" : ""}`}><input type="file" accept="image/*" onChange={chooseFile} />{file ? <><img src={file.preview} alt="Preview" /><div className="file-info"><strong>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(2)} MB</span></div><button type="button" className="icon-btn" onClick={(e) => {e.preventDefault(); setFile(null)}}><X /></button></> : <><UploadCloud size={34}/><strong>Drop sonar image here</strong><span>or click to browse from your computer</span></>}</label><div className="form-actions"><button type="button" className="secondary-btn" onClick={() => navigate("/dashboard")}>Cancel</button><button className="primary-btn" type="submit"><FileImage size={18}/> Start processing <ChevronRight size={17}/></button></div></section></form></div>;
}