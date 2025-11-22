// src/ClusteringInsights.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Link } from 'react-router-dom';

const clusterColors = ['#28a745', '#ffc107', '#dc3545']; // Green, Yellow, Red

const ClusteringInsights = () => {
  const [clusterData, setClusterData] = useState([]);

  useEffect(() => {
    axios.get('http://127.0.0.1:8000/fleet-clusters/battery')
      .then(response => setClusterData(response.data))
      .catch(error => console.error("Failed to fetch cluster data:", error));
  }, []);

  return (
    <div className="container mx-auto p-8 max-w-7xl">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-800">Fleet Clustering Insights (Batteries)</h1>
        <Link to="/" className="text-indigo-600 hover:underline">&laquo; Back to Fleet Overview</Link>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 bg-white border border-gray-200 rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold mb-2 text-gray-800">Battery Behavior Clusters</h3>
          <p className="text-gray-600 mb-4">This chart groups batteries by their operational behavior. Each color represents a distinct cluster with unique characteristics.</p>
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid stroke="#e5e7eb" />
              <XAxis type="number" dataKey="avg_temp" name="Avg. Temperature" unit="°C" stroke="#6b7280" />
              <YAxis type="number" dataKey="fade_rate" name="Capacity Fade Rate" unit="" stroke="#6b7280" tickFormatter={(tick) => tick.toFixed(3)} />
              <ZAxis type="category" dataKey="battery_id" name="Battery ID" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Legend />
              {/* Create a separate <Scatter> component for each cluster */}
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
              <h4 className="font-bold" style={{color: clusterColors[0]}}>● Cluster 0: Healthy Fleet</h4>
              <p className="text-sm text-gray-600">Low temperature and slow capacity fade. These batteries are operating under optimal conditions.</p>
            </div>
            <div>
              <h4 className="font-bold" style={{color: clusterColors[1]}}>● Cluster 1: Overheating Fleet</h4>
              <p className="text-sm text-gray-600">High operating temperature but moderate fade rate. Recommend checking cooling systems for this group.</p>
            </div>
            <div>
              <h4 className="font-bold" style={{color: clusterColors[2]}}>● Cluster 2: Failing Fleet (Bad Batch)</h4>
              <p className="text-sm text-gray-600">High fade rate regardless of temperature. This indicates a potential manufacturing defect. Isolate and inspect these units immediately.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ClusteringInsights;