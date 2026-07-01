import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />}>
          <Route index element={<Navigate to="/graph" replace />} />
          <Route path="graph" element={<App.Graph />} />
          <Route path="compare" element={<App.Compare />} />
          <Route path="config" element={<App.Config />} />
          <Route path="jobs" element={<App.Jobs />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
