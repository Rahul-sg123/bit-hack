// src/DriftAnalysisChart.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// This component is now reusable for any machine!
export default function DriftAnalysisChart({ machineType, featureName }) {
  const [driftData, setDriftData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      if (!machineType) return;
      try {
        const response = await axios.get(`http://127.0.0.1:8000/drift-analysis/${machineType}`);
        setDriftData(response.data.data);
        setLoading(false);
      } catch (error) {
        console.error(`Failed to fetch drift data for ${machineType}:`, error);
        setLoading(false);
      }
    };
    fetchData();
  }, [machineType]);

  if (loading) {
    return <p className="text-gray-500">Loading Drift Analysis...</p>;
  }

  return (
    // --- STYLING CHANGES FOR WHITE THEME ---
    <div className="bg-white border border-gray-200 rounded-lg shadow-md p-6 mt-8">
      <h3 className="text-xl font-semibold mb-2 text-gray-800">Model Health & Drift Analysis</h3>
      <p className="text-gray-600 mb-4">Comparing live '{featureName}' data against the original training data.</p>
      
      {/* Alert box styled for a light theme */}
      <div className="text-center my-4 p-3 rounded-lg bg-red-50 border border-red-400">
        <p className="font-bold text-red-600 text-lg">ALERT: Data Drift Detected!</p>
        <p className="text-sm text-gray-700">Model accuracy may be degraded. Retraining is recommended.</p>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={driftData}>
          {/* Gradients remain the same */}
          <defs>
            <linearGradient id="colorOriginal" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#8884d8" stopOpacity={0}/>
            </linearGradient>
            <linearGradient id="colorLive" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
            </linearGradient>
          </defs>
          
          {/* Chart elements styled for a light theme */}
          <CartesianGrid stroke="#e5e7eb" strokeDasharray="3 3" />
          <XAxis dataKey="bin_start" stroke="#6b7280" tickFormatter={(tick) => tick.toFixed(1)} label={{ value: featureName, position: 'insideBottom', offset: -10, fill: '#6b7280' }} />
          <YAxis stroke="#6b7280" label={{ value: 'Frequency', angle: -90, position: 'insideLeft', fill: '#6b7280' }}/>
          <Tooltip
            cursor={{ fill: 'rgba(229, 231, 235, 0.5)' }}
            contentStyle={{ background: '#ffffff', border: '1px solid #e5e7eb' }}
          />
          <Legend />
          <Area type="monotone" dataKey="original_distribution" stroke="#8884d8" fillOpacity={1} fill="url(#colorOriginal)" name="Training Data" />
          <Area type="monotone" dataKey="live_distribution" stroke="#ef4444" fillOpacity={1} fill="url(#colorLive)" name="Live Data (Drifted)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}