import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'

import App from './App'
import Pipeline from './pages/Pipeline'
import Explore from './pages/Explore'
import Metrics from './pages/Metrics'
import Attention from './pages/Attention'
import Interpretability from './pages/Interpretability'
import EvaluateDataset from './pages/EvaluateDataset'   // ← aggiungi

// Dentro <Route path="/" element={<App />}>:

import { AppDataProvider } from './context/AppData'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AppDataProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<App />}>
            <Route index element={<Navigate to="/pipeline" replace />} />
            <Route path="pipeline"  element={<Pipeline />} />
            <Route path="explore"   element={<Explore />} />
            <Route path="metrics"   element={<Metrics />} />
            <Route path="dataset"   element={<EvaluateDataset />} /> 
            <Route path="attention" element={<Attention />} />
            <Route path="interpret" element={<Interpretability />} />
                      </Route>
        </Routes>
      </BrowserRouter>
    </AppDataProvider>
  </React.StrictMode>
)