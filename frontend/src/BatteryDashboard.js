// // src/BatteryDashboard.js
// import React, { useState, useRef } from 'react';
// import ReportGenerator from './ReportGenerator';
// import BusinessImpact from './BusinessImpact';
// import MaintenanceTicketModal from './MaintenanceTicketModal';
// import RecommendationGuide from './RecommendationGuide';
// import DriftAnalysisModal from './DriftAnalysisModal';
// import axios from 'axios';
// import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
// import './App.css';

// function BatteryDashboard() {
//   const [isTicketModalOpen, setIsTicketModalOpen] = useState(false);
//   const [isDriftModalOpen, setIsDriftModalOpen] = useState(false);
//   const reportContentRef = useRef();
//   const [features, setFeatures] = useState({
//     cycle: '',
//     capacity: '',
//     temp_mean: '',
//     voltage_mean: '',
//     current_mean: '',
//     degradation_anomaly_score: ''
//   });
//   const [prediction, setPrediction] = useState(null);
//   const [error, setError] = useState('');

//   const handleChange = (e) => {
//     const { name, value } = e.target;
//     setFeatures(prev => ({ ...prev, [name]: parseFloat(value) || '' }));
//   };

//   const getPrediction = async () => {
//     setError('');
//     setPrediction(null);
//     try {
//       const response = await axios.post('http://127.0.0.1:8000/predict/battery', features);
//       setPrediction(response.data.predicted_RUL);
//     } catch (err) {
//       setError('Failed to get prediction. Make sure the backend server is running.');
//       console.error(err);
//     }
//   };

//   const isAtRisk = prediction !== null && prediction < 30;
//   const status = isAtRisk ? 'At Risk' : 'Normal';

//   // Data for the live sensor values chart
//   const sensorDataForChart = [
//     { name: 'Capacity', value: features.capacity, unit: 'Ah' },
//     { name: 'Temp Mean', value: features.temp_mean, unit: '°C' },
//     { name: 'Voltage Mean', value: features.voltage_mean, unit: 'V' },
//   ];

//   return (
//     <div className="px-8 py-12">
//       {/* --- HEADER SECTION --- */}
//       <div className="flex justify-between items-center mb-6">
//         <h1 className="text-5xl font-black text-slate-900">EV Battery EVB-001</h1>
//         <div className="flex space-x-4">
//             <button
//                 onClick={() => setIsDriftModalOpen(true)}
//                 className="font-bold text-sky-600 hover:text-sky-500 transition-colors"
//             >
//                 Check for Model Drift
//             </button>
//             <ReportGenerator machineId="EV-Battery-EVB-001" reportContentRef={reportContentRef} />
//         </div>
//       </div>

//       {/* --- WRAPPER FOR PDF CONTENT --- */}
//       <div ref={reportContentRef} className="p-4 bg-white">
//         <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

//           {/* --- LEFT COLUMN: CONTROL PANEL --- */}
//           <div className="bg-white border border-slate-200 rounded-lg shadow-md p-6 h-fit space-y-6">
//             <div>
//               <h3 className="text-xl font-bold text-slate-900 mb-4">Predict Remaining Useful Life (RUL)</h3>
//               <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
//                 {Object.keys(features).map(key => (
//                   <div key={key}>
//                     <label className="text-sm font-medium text-slate-700 capitalize">{key.replace(/_/g, ' ')}</label>
//                     <input
//                       type="number" name={key} value={features[key]} onChange={handleChange} step="0.01"
//                       className="w-full mt-1 p-2 bg-slate-50 border border-slate-300 rounded-md focus:ring-2 focus:ring-yellow-500 focus:outline-none"
//                     />
//                   </div>
//                 ))}
//               </div>
//             </div>

//             <button onClick={getPrediction} className="w-full bg-yellow-400 text-black font-bold uppercase py-3 rounded-none text-center hover:bg-yellow-500 transition-colors">
//               Predict RUL
//             </button>

//             {prediction !== null && (
//               <div className={`p-4 rounded-lg border-l-4 ${isAtRisk ? 'border-red-500 bg-red-50' : 'border-green-500 bg-green-50'}`}>
//                 <h3 className="font-bold text-lg text-slate-900">Prediction Result</h3>
//                 <p className="text-3xl font-bold mt-2">
//                   <span className={isAtRisk ? 'text-red-600' : 'text-green-600'}>{prediction}</span>
//                   <span className="text-xl font-semibold text-slate-600"> cycles remaining</span>
//                 </p>
//                 {isAtRisk && (
//                   <button onClick={() => setIsTicketModalOpen(true)} className="mt-4 w-full bg-red-600 text-white font-bold py-2 rounded-md hover:bg-red-500">
//                     🎫 Create Maintenance Ticket
//                   </button>
//                 )}
//               </div>
//             )}
//             {error && <p className="text-red-600 mt-4">{error}</p>}
//           </div>

//           {/* --- RIGHT COLUMN: ANALYSIS & VISUALIZATIONS --- */}
//           <div className="space-y-8">
//             <div className="bg-white border border-slate-200 rounded-lg shadow-md p-6">
//               <h3 className="text-xl font-bold text-slate-900 mb-4">Live Sensor Values</h3>
//               <ResponsiveContainer width="100%" height={250}>
//                   <BarChart data={sensorDataForChart}>
//                       <CartesianGrid stroke="#e5e7eb" />
//                       <XAxis dataKey="name" stroke="#6b7280" />
//                       <YAxis stroke="#6b7280" />
//                       <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #e5e7eb' }} />
//                       <Bar dataKey="value" fill="#0ea5e9" name="Sensor Reading" />
//                   </BarChart>
//               </ResponsiveContainer>
//             </div>
//             {prediction !== null && (
//               <>
//                 <BusinessImpact status={status} costPerHourDowntime={8000} hoursToRepairFailure={24} costOfProactiveMaintenance={150000} />
//                 <RecommendationGuide machineType="battery" status={status} />
//               </>
//             )}
//           </div>
//         </div>
//       </div>

//       {/* --- MODALS --- */}
//       {isDriftModalOpen && <DriftAnalysisModal machineType="battery" featureName="Capacity (Ah)" onClose={() => setIsDriftModalOpen(false)} />}
//       {isTicketModalOpen && (
//         <MaintenanceTicketModal
//           machineId="EVB-001"
//           details={`AI predicts a critically low RUL of ${prediction} cycles. Immediate inspection and replacement planning is recommended.`}
//           onClose={() => setIsTicketModalOpen(false)}
//         />
//       )}
//     </div>
//   );
// }

// // export default BatteryDashboard;

// src/BatteryDashboard.js
import React, { useState, useRef } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import ReportGenerator from "./ReportGenerator";
import BusinessImpact from "./BusinessImpact";
import MaintenanceTicketModal from "./MaintenanceTicketModal";
import RecommendationGuide from "./RecommendationGuide";
import DriftAnalysisModal from "./DriftAnalysisModal";
import axios from "axios";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import "./App.css";

function BatteryDashboard({ isLoggedIn }) {
  const [isTicketModalOpen, setIsTicketModalOpen] = useState(false);
  const [isDriftModalOpen, setIsDriftModalOpen] = useState(false);
  const reportContentRef = useRef();
  const [features, setFeatures] = useState({
    cycle: "",
    capacity: "",
    temp_mean: "",
    voltage_mean: "",
    current_mean: "",
    degradation_anomaly_score: "",
  });
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFeatures((prev) => ({ ...prev, [name]: parseFloat(value) || "" }));
  };

  const getPrediction = async () => {
    setError("");
    setPrediction(null);
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/predict/battery",
        features
      );
      setPrediction(response.data.predicted_RUL);
    } catch (err) {
      setError(
        "Failed to get prediction. Make sure the backend server is running."
      );
      console.error(err);
    }
  };

  const isAtRisk = prediction !== null && prediction < 30;
  const status = isAtRisk ? "At Risk" : "Normal";

  // Data for the live sensor values chart
  const sensorDataForChart = [
    { name: "Capacity", value: features.capacity, unit: "Ah" },
    { name: "Temp Mean", value: features.temp_mean, unit: "°C" },
    { name: "Voltage Mean", value: features.voltage_mean, unit: "V" },
  ];
  const [isDisabled, setIsDisabled] = useState(true);
  return (
    <div className="px-8 py-12">
      {/* --- HEADER SECTION --- */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-5xl font-black text-slate-900">
          EV Battery EVB-001
        </h1>
        <div className="flex space-x-4">
          <button
            onClick={() => isDisabled && setIsDriftModalOpen(true)}
            className={`font-bold ${!isLoggedIn
                ? "text-sky-400 cursor-not-allowed"
                : "text-sky-600 hover:text-sky-500"
              } transition-colors`}
            disabled={!isLoggedIn}
          >
            Check for Model Drift
          </button>
          <ReportGenerator
            machineId="EV-Battery-EVB-001"
            reportContentRef={reportContentRef}
          />
        </div>
      </div>

      {/* --- WRAPPER FOR PDF CONTENT --- */}
      <div ref={reportContentRef} className="p-4 bg-white">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* --- LEFT COLUMN: CONTROL PANEL --- */}
          <div className="bg-white border border-slate-200 rounded-lg shadow-md p-6 h-fit space-y-6">
            <div>
              <h3 className="text-xl font-bold text-slate-900 mb-4">
                Predict Remaining Useful Life (RUL)
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.keys(features).map((key) => (
                  <div key={key}>
                    <label className="text-sm font-medium text-slate-700 capitalize">
                      {key.replace(/_/g, " ")}
                    </label>
                    <input
                      type="number"
                      name={key}
                      value={features[key]}
                      onChange={handleChange}
                      step="0.01"
                      className="w-full mt-1 p-2 bg-slate-50 border border-slate-300 rounded-md focus:ring-2 focus:ring-yellow-500 focus:outline-none"
                    />
                  </div>
                ))}
              </div>
            </div>

            <button
              onClick={getPrediction}
              className="w-full bg-yellow-400 text-black font-bold uppercase py-3 rounded-none text-center hover:bg-yellow-500 transition-colors"
            >
              Predict RUL
            </button>

            {prediction !== null && (
              <div
                className={`p-4 rounded-lg border-l-4 ${isAtRisk
                    ? "border-red-500 bg-red-50"
                    : "border-green-500 bg-green-50"
                  }`}
              >
                <h3 className="font-bold text-lg text-slate-900">
                  Prediction Result
                </h3>
                <p className="text-3xl font-bold mt-2">
                  <span
                    className={isAtRisk ? "text-red-600" : "text-green-600"}
                  >
                    {prediction}
                  </span>
                  <span className="text-xl font-semibold text-slate-600">
                    {" "}
                    cycles remaining
                  </span>
                </p>
                {isAtRisk && (
                  <button
                    onClick={() => setIsTicketModalOpen(true)}
                    className="mt-4 w-full bg-red-600 text-white font-bold py-2 rounded-md hover:bg-red-500"
                  >
                    Create Maintenance Ticket
                  </button>
                )}
              </div>
            )}
            {error && <p className="text-red-600 mt-4">{error}</p>}
          </div>

          {/* --- RIGHT COLUMN: ANALYSIS & VISUALIZATIONS --- */}
          <div className="space-y-8">
            <div className="bg-white border border-slate-200 rounded-lg shadow-md p-6">
              <h3 className="text-xl font-bold text-slate-900 mb-4">
                Live Sensor Values
              </h3>
              <ResponsiveContainer width="100%" height={250}>

                <BarChart data={sensorDataForChart}>
                  <CartesianGrid stroke="#e5e7eb" />
                  <XAxis dataKey="name" stroke="#6b7280" />
                  <YAxis stroke="#6b7280" />
                  <Tooltip
                    contentStyle={{
                      background: "#ffffff",
                      border: "1px solid #e5e7eb",
                    }}
                  />
                  <Bar dataKey="value" fill="#0ea5e9" name="Sensor Reading" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            {isLoggedIn && prediction !== null && (
              <>
                <BusinessImpact
                  status={status}
                  costPerHourDowntime={8000}
                  hoursToRepairFailure={24}
                  costOfProactiveMaintenance={150000}
                />
                <RecommendationGuide machineType="battery" status={status} />
              </>
            )}

          </div>
        </div>
      </div>

      {/* --- MODALS --- */}
      {isDriftModalOpen && (
        <DriftAnalysisModal
          machineType="battery"
          featureName="Capacity (Ah)"
          onClose={() => setIsDriftModalOpen(false)}
        />
      )}
      {isTicketModalOpen && ( <MaintenanceTicketModal machineId="EVB-001"  details={`AI predicts a critically low RUL of ${prediction} cycles.Immediate inspection and replacement planning is recommended.`}onClose={() => setIsTicketModalOpen(false)} />
      )}

      {/* logged check  */}
      {/* {!isLoggedIn && (
        <div className="absolute inset-0 bg-white/70 backdrop-blur-[2px] flex flex-col items-center justify-center rounded-lg">
          <p className="text-gray-700 font-semibold mb-2">
            Login required to access this
          </p>
          <Link
            to="/account"
            className="bg-yellow-400 text-black font-semibold px-4 py-2 rounded-lg hover:bg-yellow-500 transition"
          >
            Go to Login
          </Link>
        </div>

      )} */}
    </div>
  );
}

export default BatteryDashboard;