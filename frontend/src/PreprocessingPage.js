// src/PreprocessingPage.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Link } from 'react-router-dom';

const PreprocessingPage = () => {
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    axios.get('http://127.0.0.1:8000/visualize/noise-filter')
      .then(response => setChartData(response.data))
      .catch(error => console.error("Failed to fetch noise filter data:", error));
  }, []);

  return (
    <div className="container mx-auto p-8 max-w-7xl">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-800">Real-Time Data Preprocessing</h1>
        <Link to="/" className="text-indigo-600 hover:underline">&laquo; Back to Fleet Overview</Link>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg shadow-md p-6">
        <h3 className="text-xl font-semibold mb-2 text-gray-800">Noise Filtering with Moving Average</h3>
        <p className="text-gray-600 mb-4">
          Real-world sensor data is noisy. Our preprocessing pipeline automatically filters this raw signal to extract the true underlying trend. This ensures our AI makes decisions based on reliable data, preventing false alarms.
        </p>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData}>
            <CartesianGrid stroke="#e5e7eb" />
            <XAxis dataKey="time_step" name="Time" unit="s" stroke="#6b7280" />
            <YAxis name="Sensor Value" stroke="#6b7280" />
            <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #e5e7eb' }} />
            <Legend />
            <Line type="monotone" dataKey="raw_noisy_data" stroke="#ef4444" name="Raw Noisy Signal" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="smoothed_data" stroke="#3b82f6" name="Cleaned Signal (Moving Avg)" dot={false} strokeWidth={3} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default PreprocessingPage;