import { useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import NewSurvey from "./pages/NewSurvey";
import Processing from "./pages/Processing";
import SurveyDetails from "./pages/SurveyDetails";
import AnomalyDetails from "./pages/AnomalyDetails";
import Reports from "./pages/Reports";
import AppLayout from "./layouts/AppLayout";

export default function App() {
  const [loggedIn, setLoggedIn] = useState(
    localStorage.getItem("sonaris_logged_in") === "true"
  );

  function handleLogin() {
    localStorage.setItem("sonaris_logged_in", "true");
    setLoggedIn(true);
  }

  function handleLogout() {
    localStorage.removeItem("sonaris_logged_in");
    setLoggedIn(false);
  }

  if (!loggedIn) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <Routes>
      <Route element={<AppLayout onLogout={handleLogout} />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/surveys/new" element={<NewSurvey />} />
        <Route path="/processing" element={<Processing />} />
        <Route path="/surveys/:surveyId" element={<SurveyDetails />} />
        <Route path="/anomalies/:anomalyId" element={<AnomalyDetails />} />
        <Route path="/reports" element={<Reports />} />
      </Route>
      <Route path="*" element={<Navigate to="/dashboard" replace />} />
    </Routes>
  );
}