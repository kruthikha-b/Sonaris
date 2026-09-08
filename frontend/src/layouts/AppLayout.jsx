import { useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { Menu, X, LogOut, Bell, UserRound, LayoutDashboard, PlusCircle, FileBarChart } from "lucide-react";
import Logo from "../components/Logo";
import { navItems } from "../utils/constants";

const icons = { LayoutDashboard, PlusCircle, FileBarChart };

export default function AppLayout({ onLogout }) {
  const [open, setOpen] = useState(false);
  const location = useLocation();

  return (
    <div className="app-shell">
      <aside className={`sidebar ${open ? "open" : ""}`}>
        <div className="sidebar-top"><Logo /><button className="icon-btn mobile-close" onClick={() => setOpen(false)}><X /></button></div>
        <div className="workspace"><span className="status-dot" /> Demo workspace</div>
        <nav>
          {navItems.map((item) => {
            const Icon = icons[item.icon];
            return <NavLink key={item.path} to={item.path} onClick={() => setOpen(false)} className={({isActive}) => isActive ? "active" : ""}><Icon size={19} />{item.label}</NavLink>;
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="help-card"><strong>Need help?</strong><span>Check the project docs for setup and API details.</span></div>
          <button className="logout-btn" onClick={onLogout}><LogOut size={18} /> Sign out</button>
        </div>
      </aside>
      {open && <div className="overlay" onClick={() => setOpen(false)} />}
      <main className="main-area">
        <header className="topbar">
          <button className="icon-btn menu-btn" onClick={() => setOpen(true)}><Menu /></button>
          <div className="topbar-actions"><button className="icon-btn"><Bell size={19} /></button><div className="user-chip"><div className="avatar">DK</div><span>Survey analyst</span><UserRound size={16} /></div></div>
        </header>
        <div className="page-content"><Outlet /></div>
      </main>
    </div>
  );
}