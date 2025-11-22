// src/MotorClusteringPage.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Link } from 'react-router-dom';

const clusterColors = ['#28a745', '#ffc107', '#dc3545']; // Green, Yellow, Red

const MotorClusteringPage = () => {
  const [clusterData, setClusterData] = useState([]);

  useEffect(() => {
    // Fetch the motor-specific cluster data
    axios.get('http://127.0.0.1:8000/fleet-clusters/motor')
      .then(response => {
        if (Array.isArray(response.data)) {
          setClusterData(response.data);
        }
      })
      .catch(error => console.error("Failed to fetch motor cluster data:", error));
  }, []);

  return (
    <div className="container mx-auto p-8 max-w-7xl">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-800">Fleet Clustering Insights (Motors)</h1>
        <Link to="/" className="text-indigo-600 hover:underline">&laquo; Back to Fleet Overview</Link>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 bg-white border border-gray-200 rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold mb-2 text-gray-800">Motor Operational Clusters</h3>
          <p className="text-gray-600 mb-4">Motors are grouped by their average operational stress. Each color represents a distinct behavioral pattern.</p>
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid stroke="#e5e7eb" />
              <XAxis type="number" dataKey="avg_torque" name="Avg. Torque" unit="Nm" stroke="#6b7280" />
              <YAxis type="number" dataKey="avg_speed" name="Avg. Speed" unit="rpm" stroke="#6b7280" />
              <ZAxis type="category" dataKey="Product ID" name="Motor ID" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Legend />
              {[0, 1, 2].map(clusterId => (
                <Scatter 
                  key={clusterId}
                  name={`Cluster ${clusterId}`} 
                  data={clusterData.filter(d => d.cluster === clusterId)} 
                  fill={clusterColors[clusterId]} 
                />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-sky-50 border border-sky-200 rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold mb-4 text-gray-800">Cluster Interpretation</h3>
          <div className="space-y-4">
            <div>
              <h4 className="font-bold" style={{color: clusterColors[0]}}>● Cluster 0: Standard Operation</h4>
              <p className="text-sm text-gray-600">These motors are operating within normal torque and speed parameters. Represents a healthy baseline.</p>
            </div>
            <div>
              <h4 className="font-bold" style={{color: clusterColors[1]}}>● Cluster 1: High Stress</h4>
              <p className="text-sm text-gray-600">This group consistently runs at high torque and/or speed. They are at higher risk of premature wear. Recommend load balancing.</p>
            </div>
            <div>
              <h4 className="font-bold" style={{color: clusterColors[2]}}>● Cluster 2: High Wear / Failure Prone</h4>
              <p className="text-sm text-gray-600">These motors show significant tool wear relative to their operational stress. This is a strong indicator of imminent failure.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MotorClusteringPage;