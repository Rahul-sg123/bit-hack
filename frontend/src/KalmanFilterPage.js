// src/KalmanFilterPage.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Link } from 'react-router-dom';

const KalmanFilterPage = () => {
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    axios.get('http://127.0.0.1:8000/visualize/kalman-filter')
      .then(response => setChartData(response.data))
      .catch(error => console.error("Failed to fetch Kalman filter data:", error));
  }, []);

  return (
    <div className="container mx-auto p-8 max-w-7xl">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-800">Advanced Preprocessing: Kalman Filter</h1>
        <Link to="/" className="text-indigo-600 hover:underline">&laquo; Back to Fleet Overview</Link>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg shadow-md p-6">
        {/* <h3 className="text-xl font-semibold mb-2 text-gray-800">Kalman Filter vs. Moving Average</h3> */}
        <p className="text-gray-600 mb-4">
          A Moving Average is effective but lags behind the true signal. A Kalman Filter uses a predictive model to track the true signal more closely and responsively, allowing for faster and more accurate anomaly detection.
        </p>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData}>
            <CartesianGrid stroke="#e5e7eb" />
            <XAxis dataKey="time_step" name="Time" unit="s" stroke="#6b7280" />
            <YAxis name="Sensor Value" stroke="#6b7280" domain={['auto', 'auto']} />
            <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #e5e7eb' }} />
            <Legend />
            <Line type="monotone" dataKey="raw_noisy_data" stroke="#fca5a5" name="Raw Noisy Signal" dot={false} strokeWidth={1} />
            <Line type="monotone" dataKey="moving_average" stroke="#fb923c" name="Moving Average (Lagging)" dot={false} strokeWidth={3} />
            <Line type="monotone" dataKey="kalman_filter" stroke="#22c55e" name="Kalman Filter (More Accurate)" dot={false} strokeWidth={3} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default KalmanFilterPage;