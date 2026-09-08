# SONARIS Frontend

A responsive React + Vite frontend prototype for the SONARIS underwater marine debris and anomaly detection system.

## Run locally

```bash
npm install
npm run dev
```

Open the local URL shown by Vite, usually `http://localhost:5173`.

## Included

- Login screen
- Dashboard
- New survey creation
- Sonar image upload with preview
- Processing screen
- Survey details with sonar view, map view, detection table, and analytics
- Anomaly details and review actions
- Reports page
- Responsive sidebar and mobile layout
- Mock data and local browser state

## Backend integration

Replace the functions in `src/services/api.js` with real API calls when the backend is ready. The UI currently uses mock data so the complete flow can be demonstrated without a backend.
