// src/DriftAnalysisModal.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const DriftAnalysisModal = ({ machineType, featureName, onClose }) => {
  const [driftData, setDriftData] = useState([]);
  const [isDriftDetected, setIsDriftDetected] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get(`http://127.0.0.1:8000/drift-analysis/${machineType}`)
      .then(response => {
        setDriftData(response.data.data);
        setIsDriftDetected(response.data.drift_detected);
        setLoading(false);
      })
      .catch(error => {
        console.error(`Failed to fetch drift data for ${machineType}:`, error);
        setLoading(false);
      });
  }, [machineType]);

  return (
    <div className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50">
      <div className="bg-white p-8 rounded-lg shadow-2xl w-full max-w-3xl">
        <h2 className="text-2xl font-bold mb-2 text-gray-800">Model Health & Drift Analysis</h2>
        <p className="text-gray-600 mb-4">Comparing live '{featureName}' data distribution against the original training data.</p>

        {loading ? (
          <p>Loading analysis...</p>
        ) : (
          <>
            {isDriftDetected ? (
              <div className="text-center my-4 p-3 rounded-lg bg-red-50 border border-red-400">
                <p className="font-bold text-red-600 text-lg">ALERT: Data Drift Detected!</p>
              </div>
            ) : (
              <div className="text-center my-4 p-3 rounded-lg bg-green-50 border border-green-400">
                <p className="font-bold text-green-600 text-lg">✓ Model is Stable</p>
              </div>
            )}
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={driftData}>
                <CartesianGrid stroke="#e5e7eb" />
                <XAxis dataKey="bin_start" stroke="#6b7280" tickFormatter={(tick) => tick.toFixed(1)} />
                <YAxis stroke="#6b7280" />
                <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #e5e7eb' }} />
                <Legend />
                <Area type="monotone" dataKey="original_distribution" stroke="#8884d8" fill="#8884d8" fillOpacity={0.3} name="Training Data" />
                <Area type="monotone" dataKey="live_distribution" stroke="#ef4444" fill="#ef4444" fillOpacity={0.3} name="Live Data" />
              </AreaChart>
            </ResponsiveContainer>
          </>
        )}
        <div className="flex justify-end mt-6">
          <button onClick={onClose} className="px-6 py-2 bg-slate-200 text-slate-800 rounded-md font-semibold hover:bg-slate-300">Close</button>
        </div>
      </div>
    </div>
  );
};

export default DriftAnalysisModal;