// // src/MotorDashboard.js
// import React, { useState, useRef } from 'react';
// import BusinessImpact from './BusinessImpact';
// import ReportGenerator from './ReportGenerator';
// import RecommendationGuide from './RecommendationGuide';
// import MaintenanceTicketModal from './MaintenanceTicketModal';
// import LiveDataDisplay from './components/LiveDataDisplay'; // Adjust path if needed
// import DriftAnalysisModal from './DriftAnalysisModal';
// import axios from 'axios';
// import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
// import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
// import './App.css'; // Reuse the same styles

// // Define SEQUENCE_LEN to match the backend/training
// const SEQUENCE_LEN = 10;

// function MotorDashboard(isLoggedIn) {
//     const reportContentRef = useRef();
//     const [features, setFeatures] = useState({
//         air_temperature_k: '',
//         process_temperature_k: '',
//         rotational_speed_rpm: '',
//         torque_nm: '',
//         tool_wear_min: '',
//     });

//     const [prediction, setPrediction] = useState(null);
//     const [error, setError] = useState('');
//     const [shapData, setShapData] = useState(null);
//     const [isTicketModalOpen, setIsTicketModalOpen] = useState(false);
//     const [isDriftModalOpen, setIsDriftModalOpen] = useState(false);
//     const [anomalyScore, setAnomalyScore] = useState(null); // <-- State for Anomaly Score Added

//     const handleChange = (e) => {
//         const { name, value } = e.target;
//         setFeatures(prev => ({ ...prev, [name]: value === '' ? '' : parseFloat(value) }));
//     };

//     const getPrediction = async () => {
//         setError('');
//         setPrediction(null);
//         setShapData(null);
//         setAnomalyScore(null); // <-- Reset anomaly score

//         try {
//             // --- Get Standard Prediction ---
//             const response = await axios.post('http://127.0.0.1:8000/predict/motor', features);
//             const predictionData = response.data; // Store prediction response
//             setPrediction(predictionData);

//             let topFeature = null; // Variable to store top SHAP feature

//             // --- Get SHAP if needed ---
//             if (predictionData.status === 'At Risk') {
//                 try {
//                     const shapResponse = await axios.post('http://127.0.0.1:8000/explain/motor', features);
//                     const shapResult = shapResponse.data;
//                     setShapData(shapResult);
//                     if (shapResult && shapResult.length > 0) {
//                         topFeature = shapResult[0].feature; // Get the top feature name
//                     }
//                 } catch (shapErr) {
//                     console.error("Failed to get SHAP data:", shapErr)
//                     // Optionally set a specific error for SHAP if needed
//                 }
//             }

//             // --- Get Anomaly Score ---
//             try {
//                 // Ensure all features are numbers (or default to 0 for the sequence)
//                 const currentFeatures = [
//                     features.air_temperature_k || 0,
//                     features.process_temperature_k || 0,
//                     features.rotational_speed_rpm || 0,
//                     features.torque_nm || 0,
//                     features.tool_wear_min || 0
//                 ];
//                 // Simulate a sequence by repeating the current input
//                 const sequenceData = Array(SEQUENCE_LEN).fill(currentFeatures);

//                 const anomalyResponse = await axios.post('http://127.0.0.1:8000/anomaly-score/motor', { sequence: sequenceData });
//                 setAnomalyScore(anomalyResponse.data);

//             } catch (anomalyErr) {
//                 console.error("Failed to get anomaly score:", anomalyErr);
//                 // Optionally set a specific error for the anomaly score card
//             }

//             // --- Get Prescriptive Advice (moved here to potentially use topFeature) ---
//             // Assuming you have setRecommendations = useState([]) defined
//             /* // Uncomment if you add setRecommendations state
//             try {
//                 const advicePayload = {
//                     status: predictionData.status,
//                     health_score: parseFloat(predictionData.health_score),
//                     top_feature: topFeature,
//                     current_features: features // Send current features for rule checks
//                 };
//                 const adviceResponse = await axios.post('http://127.0.0.1:8000/prescriptive-advice/motor', advicePayload);
//                 // setRecommendations(adviceResponse.data.recommendations || []); // Update state here
//             } catch (adviceErr) {
//                 console.error("Failed to get prescriptive advice:", adviceErr);
//             }
//             */


//         } catch (err) {
//             setError('Failed to get prediction. Ensure the backend server is running.');
//             console.error(err);
//         }
//     };

//     const sensorDataForChart = [
//       { name: 'Rotational Speed', value: features.rotational_speed_rpm, unit: 'rpm' },
//       { name: 'Torque', value: features.torque_nm, unit: 'Nm' },
//       { name: 'Tool Wear', value: features.tool_wear_min, unit: 'min' },
//     ];
// const [isDisabled, setIsDisabled] = useState(true);
//     return (
//         <div className="px-8 py-12">
//             {/* --- HEADER SECTION --- */}
//             <div className="flex justify-between items-center mb-6">
//                 <h1 className="text-5xl font-black text-slate-900">Motor MOT-007</h1>
//                 <div className="flex space-x-4">
//                     <button onClick={() =>!isDisabled && setIsDriftModalOpen(true)} className={`font-bold ${!isLoggedIn ? "text-sky-400 cursor-not-allowed" : "text-sky-600 hover:text-sky-500"} transition-colors`} disabled={isDisabled}>
//                         Check for Model Drift
//                     </button>
//                     <ReportGenerator machineId="Motor-MOT-007" reportContentRef={reportContentRef} />
//                 </div>
//             </div>

//             {/* --- WRAPPER FOR PDF CONTENT --- */}
//             <div ref={reportContentRef} className="p-4 bg-white">
//                 <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

//                     {/* --- LEFT COLUMN: CONTROL PANEL & RESULTS --- */}
//                     <div className="space-y-6">
//                         {/* Box 1: Control Panel */}
//                         <div className="bg-white border border-slate-200 rounded-lg shadow-md p-6 h-fit">
//                             <h3 className="text-xl font-bold text-slate-900 mb-4">Live Sensor Input</h3>
//                             <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
//                                 {Object.keys(features).map(key => (
//                                     <div key={key}>
//                                         <label className="text-sm font-medium text-slate-700 capitalize">{key.replace(/_/g, ' ')}</label>
//                                         <input
//                                             type="number" name={key} value={features[key]} onChange={handleChange} step="0.1"
//                                             className="w-full mt-1 p-2 bg-slate-50 border border-slate-300 rounded-md focus:ring-2 focus:ring-yellow-500 focus:outline-none"
//                                         />
//                                     </div>
//                                 ))}
//                             </div>
//                             <button onClick={getPrediction} className="mt-6 w-full bg-yellow-400 text-black font-bold uppercase py-3 rounded-none text-center hover:bg-yellow-500 transition-colors">
//                                 Predict Health
//                             </button>
//                         </div>

//                         {/* Box 2: Prediction Result */}
//                         {prediction && (
//                             <div className={`p-6 rounded-lg border-l-4 ${prediction.status === 'At Risk' ? 'border-red-500 bg-red-50' : 'border-green-500 bg-green-50'}`}>
//                                 <h3 className="font-bold text-lg text-slate-900">Prediction Result</h3>
//                                 <p className="text-xl mt-2">
//                                     Status: <span className={`font-extrabold ${prediction.status === 'At Risk' ? 'text-red-600' : 'text-green-600'}`}>{prediction.status}</span>
//                                 </p>
//                                 <p className="text-xl mt-1">
//                                     Health Score: <strong className="text-slate-900">{prediction.health_score} / 100</strong>
//                                 </p>
//                                 {prediction.status === 'At Risk' && (
//                                     <button onClick={() => setIsTicketModalOpen(true)} className="mt-4 w-full bg-red-600 text-white font-bold py-2 rounded-md hover:bg-red-500">
//                                         🎫 Create Maintenance Ticket
//                                     </button>
//                                 )}
//                             </div>
//                         )}

//                         {/* Box 3 & 4: Business Impact and Recommendations */}
//                         {prediction && <BusinessImpact status={prediction.status} costPerHourDowntime={50000} hoursToRepairFailure={8} costOfProactiveMaintenance={75000} />}
//                         {/* {prediction && <PrescriptiveAdvisor recommendations={recommendations} />} Uncomment if you add setRecommendations state */}
//                         {prediction && <RecommendationGuide machineType="motor" status={prediction.status} />}

//                         {error && <p className="text-red-600 mt-4 text-center">{error}</p>}
//                     </div>

//                     {/* --- RIGHT COLUMN: VISUALIZATIONS --- */}
//                     {!isLoggedIn && (
//         <div className="absolute inset-0 bg-white/70 backdrop-blur-[2px] flex flex-col items-center justify-center rounded-lg">
//           <p className="text-gray-700 font-semibold mb-2">
//             Login required to access this
//           </p>
//           <Link
//             to="/account"
//             className="bg-yellow-400 text-black font-semibold px-4 py-2 rounded-lg hover:bg-yellow-500 transition"
//           >
//             Go to Login
//           </Link>
//         </div>

//       )}
      
//                     <div className="space-y-8">
//                         {/* Live Data Chart */}
//                         <div className="bg-white border border-slate-200 rounded-lg shadow-md p-6">
//                            <h3 className="text-xl font-bold text-slate-900 mb-4">Live Operational Data</h3>
//                            <ResponsiveContainer width="100%" height={300}>
//                                 <BarChart data={sensorDataForChart}>
//                                     <CartesianGrid stroke="#e5e7eb" />
//                                     <XAxis dataKey="name" stroke="#6b7280" />
//                                     <YAxis stroke="#6b7280" />
//                                     <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #e5e7eb' }} />
//                                     <Bar dataKey="value" fill="#0ea5e9" name="Sensor Reading" />
//                                 </BarChart>
//                            </ResponsiveContainer>
//                         </div>

//                         {/* --- ADDED: Anomaly Detection Card --- */}
//                         <div className="bg-white border border-slate-200 rounded-lg shadow-md p-6">
//                             <h3 className="text-xl font-bold text-slate-900 mb-4">LSTM Anomaly Score</h3>
//                             {anomalyScore ? (
//                                 <div>
//                                     <p className="text-4xl font-black text-center mb-2" style={{ color: anomalyScore.is_anomaly ? '#ef4444' : '#22c55e' }}>
//                                         {anomalyScore.anomaly_score.toFixed(4)}
//                                     </p>
//                                     <p className="text-sm text-slate-500 text-center mb-4">(Threshold: {anomalyScore.threshold.toFixed(4)})</p>
//                                     {anomalyScore.is_anomaly ? (
//                                         <p className="text-center font-bold text-red-600">Anomaly Detected! Deviation from normal.</p>
//                                     ) : (
//                                         <p className="text-center font-bold text-green-600">Normal Operation Detected.</p>
//                                     )}
//                                 </div>
//                             ) : (
//                                 <p className="text-slate-500 text-center">Run prediction to calculate anomaly score.</p>
//                             )}
//                         </div>
//                         {/* --- END Anomaly Detection Card --- */}

//                         {/* SHAP Chart appears here when needed */}
//                         {shapData && (
//                             <div className="bg-white border border-gray-200 rounded-lg shadow-md p-6">
//                                 <h3 className="text-xl font-semibold mb-2 text-gray-800">Why is it "At Risk"? (XAI)</h3>
//                                 <ResponsiveContainer width="100%" height={300}>
//                                     <BarChart data={shapData} layout="vertical" margin={{ top: 5, right: 20, left: 120, bottom: 5 }}>
//                                         <CartesianGrid stroke="#e5e7eb" />
//                                         <XAxis type="number" stroke="#6b7280" />
//                                         <YAxis type="category" dataKey="feature" stroke="#6b7280" tick={{ fill: '#374151' }} />
//                                         <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #e5e7eb' }} />
//                                         <Bar dataKey="importance" fill="#ef4444" name="Contribution to Risk" />
//                                     </BarChart>
//                                 </ResponsiveContainer>
//                             </div>
//                         )}
//                     </div>
//                 </div>
//             </div>

//             {/* --- MODALS --- */}
//             {isDriftModalOpen && <DriftAnalysisModal machineType="motor" featureName="Torque (Nm)" onClose={() => setIsDriftModalOpen(false)} />}
//             {isTicketModalOpen && <MaintenanceTicketModal machineId="Motor-MOT-007" details={`AI detected 'At Risk' status (Health Score: ${prediction?.health_score}). Suspected cause from XAI: High Torque & Tool Wear.`} onClose={() => setIsTicketModalOpen(false)} />}
//         </div>
//     );
// }

// export default MotorDashboard;

// src/MotorDashboard.js
import React, { useState, useRef } from 'react';
import { Link } from 'react-router-dom'; // Added Link for Login button
import BusinessImpact from './BusinessImpact';
import ReportGenerator from './ReportGenerator';
import RecommendationGuide from './RecommendationGuide';
import MaintenanceTicketModal from './MaintenanceTicketModal';
import DriftAnalysisModal from './DriftAnalysisModal';
import PrescriptiveAdvisor from './PrescriptiveAdvisor';
import LiveDataDisplay from './components/LiveDataDisplay'; // Added LiveDataDisplay
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './App.css'; // Reuse the same styles

// Define SEQUENCE_LEN to match the backend/training
const SEQUENCE_LEN = 10;

function MotorDashboard() {
    const reportContentRef = useRef();
    const [features, setFeatures] = useState({
        air_temperature_k: '',
        process_temperature_k: '',
        rotational_speed_rpm: '',
        torque_nm: '',
        tool_wear_min: '',
    });

    const [prediction, setPrediction] = useState(null);
    const [error, setError] = useState('');
    const [shapData, setShapData] = useState(null);
    const [isTicketModalOpen, setIsTicketModalOpen] = useState(false);
    const [isDriftModalOpen, setIsDriftModalOpen] = useState(false);
    const [anomalyScore, setAnomalyScore] = useState(null);
    const [recommendations, setRecommendations] = useState([]); // Added state for Prescriptive Advisor
    
    // --- State to control login visibility ---
    // Set to 'false' to test the overlay, 'true' to see content
    const [isLoggedIn, setIsLoggedIn] = useState(true); 

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFeatures(prev => ({ ...prev, [name]: value === '' ? '' : parseFloat(value) }));
    };

    const getPrediction = async () => {
        setError('');
        setPrediction(null);
        setShapData(null);
        setAnomalyScore(null);
        setRecommendations([]); // Reset recommendations

        try {
            // --- Get Standard Prediction ---
            const response = await axios.post('http://127.0.0.1:8000/predict/motor', features);
            const predictionData = response.data;
            setPrediction(predictionData);

            let topFeature = null;

            // --- Get SHAP if needed ---
            if (predictionData.status === 'At Risk') {
                try {
                    const shapResponse = await axios.post('http://127.0.0.1:8000/explain/motor', features);
                    const shapResult = shapResponse.data;
                    setShapData(shapResult);
                    if (shapResult && shapResult.length > 0) {
                        topFeature = shapResult[0].feature;
                    }
                } catch (shapErr) {
                    console.error("Failed to get SHAP data:", shapErr)
                }
            }

            // --- Get Anomaly Score ---
            try {
                const currentFeatures = [
                    features.air_temperature_k || 0,
                    features.process_temperature_k || 0,
                    features.rotational_speed_rpm || 0,
                    features.torque_nm || 0,
                    features.tool_wear_min || 0
                ];
                const sequenceData = Array(SEQUENCE_LEN).fill(currentFeatures);
                const anomalyResponse = await axios.post('http://127.0.0.1:8000/anomaly-score/motor', { sequence: sequenceData });
                setAnomalyScore(anomalyResponse.data);
            } catch (anomalyErr) {
                console.error("Failed to get anomaly score:", anomalyErr);
            }

            // --- Get Prescriptive Advice ---
            try {
                const advicePayload = {
                    status: predictionData.status,
                    health_score: parseFloat(predictionData.health_score),
                    top_feature: topFeature,
                    current_features: features
                };
                const adviceResponse = await axios.post('http://127.0.0.1:8000/prescriptive-advice/motor', advicePayload);
                setRecommendations(adviceResponse.data.recommendations || []);
            } catch (adviceErr) {
                console.error("Failed to get prescriptive advice:", adviceErr);
            }

        } catch (err) {
            setError('Failed to get prediction. Ensure the backend server is running.');
            console.error(err);
        }
    };
    
    const sensorDataForChart = [
      { name: 'Rotational Speed', value: features.rotational_speed_rpm, unit: 'rpm' },
      { name: 'Torque', value: features.torque_nm, unit: 'Nm' },
      { name: 'Tool Wear', value: features.tool_wear_min, unit: 'min' },
    ];

    return (
        <div className="px-8 py-12">
            {/* --- HEADER SECTION --- */}
            <div className="flex justify-between items-center mb-6">
                <h1 className="text-5xl font-black text-slate-900">Motor MOT-007</h1>
                <div className="flex space-x-4">
                    <button 
                        onClick={() => isLoggedIn && setIsDriftModalOpen(true)} 
                        className={`font-bold transition-colors ${isLoggedIn ? 'text-sky-600 hover:text-sky-500' : 'text-slate-400 cursor-not-allowed'}`}
                        disabled={!isLoggedIn}
                    >
                        Check for Model Drift
                    </button>
                    <ReportGenerator machineId="Motor-MOT-007" reportContentRef={reportContentRef} />
                </div>
            </div>

            {/* --- WRAPPER FOR PDF CONTENT --- */}
            <div ref={reportContentRef} className="p-4 bg-white">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

                    {/* --- LEFT COLUMN: CONTROL PANEL & RESULTS --- */}
                    <div className="space-y-6">
                        {/* Box 1: Control Panel */}
                        <div className="bg-white border border-slate-200 rounded-lg shadow-md p-6 h-fit">
                            <h3 className="text-xl font-bold text-slate-900 mb-4">Live Sensor Input</h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                {Object.keys(features).map(key => (
                                    <div key={key}>
                                        <label className="text-sm font-medium text-slate-700 capitalize">{key.replace(/_/g, ' ')}</label>
                                        <input
                                            type="number" name={key} value={features[key]} onChange={handleChange} step="0.1"
                                            className="w-full mt-1 p-2 bg-slate-50 border border-slate-300 rounded-md focus:ring-2 focus:ring-yellow-500 focus:outline-none"
                                        />
                                    </div>
                                ))}
                            </div>
                            <button onClick={getPrediction} className="mt-6 w-full bg-yellow-400 text-black font-bold uppercase py-3 rounded-none text-center hover:bg-yellow-500 transition-colors">
                                Predict Health
                            </button>
                        </div>
                        
                        {/* Box 2: Prediction Result */}
                        {prediction && (
                            <div className={`p-6 rounded-lg border-l-4 ${prediction.status === 'At Risk' ? 'border-red-500 bg-red-50' : 'border-green-500 bg-green-50'}`}>
                                <h3 className="font-bold text-lg text-slate-900">Prediction Result</h3>
                                <p className="text-xl mt-2">
                                    Status: <span className={`font-extrabold ${prediction.status === 'At Risk' ? 'text-red-600' : 'text-green-600'}`}>{prediction.status}</span>
                                </p>
                                <p className="text-xl mt-1">
                                    Health Score: <strong className="text-slate-900">{prediction.health_score} / 100</strong>
                                </p>
                                {prediction.status === 'At Risk' && (
                                    <button onClick={() => setIsTicketModalOpen(true)} className="mt-4 w-full bg-red-600 text-white font-bold py-2 rounded-md hover:bg-red-500">
                                        🎫 Create Maintenance Ticket
                                    </button>
                                )}
                            </div>
                        )}

                        {/* Box 3, 4, 5: Business Impact, Recommendations, and Prescriptive Advice */}
                        {prediction && <BusinessImpact status={prediction.status} costPerHourDowntime={50000} hoursToRepairFailure={8} costOfProactiveMaintenance={75000} />}
                        {prediction && <RecommendationGuide machineType="motor" status={prediction.status} />}
                        {prediction && <PrescriptiveAdvisor recommendations={recommendations} />}

                        {error && <p className="text-red-600 mt-4 text-center">{error}</p>}
                    </div>

                    {/* --- RIGHT COLUMN: VISUALIZATIONS (with Login Overlay) --- */}
                    <div className="relative space-y-8"> {/* Added 'relative' for the overlay */}
                        
                        {/* --- LOGIN OVERLAY --- */}
                        {!isLoggedIn && (
                            <div className="absolute inset-0 bg-white/70 backdrop-blur-sm flex flex-col items-center justify-center rounded-lg z-10">
                                <p className="text-gray-700 font-semibold mb-2">
                                    Login required to view analytics
                                </p>
                                <Link
                                    to="/account" // Assuming you have a login route
                                    className="bg-yellow-400 text-black font-semibold px-4 py-2 rounded-lg hover:bg-yellow-500 transition"
                                >
                                    Go to Login
                                </Link>
                            </div>
                        )}

                        {/* --- The content that will be blurred --- */}
                        <div className={!isLoggedIn ? 'blur-sm' : ''}>
                            {/* Live Data Display */}
                            <LiveDataDisplay machineId="MOT-007" />

                            {/* Live Operational Data Chart */}
                            <div className="bg-white border border-slate-200 rounded-lg shadow-md p-6 mt-8">
                                <h3 className="text-xl font-bold text-slate-900 mb-4">Live Operational Data</h3>
                                <ResponsiveContainer width="100%" height={300}>
                                    <BarChart data={sensorDataForChart}>
                                        <CartesianGrid stroke="#e5e7eb" />
                                        <XAxis dataKey="name" stroke="#6b7280" />
                                        <YAxis stroke="#6b7280" />
                                        <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #e5e7eb' }} />
                                        <Bar dataKey="value" fill="#0ea5e9" name="Sensor Reading" />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Anomaly Detection Card */}
                            <div className="bg-white border border-slate-200 rounded-lg shadow-md p-6 mt-8">
                                <h3 className="text-xl font-bold text-slate-900 mb-4">LSTM Anomaly Score</h3>
                                {anomalyScore ? (
                                    <div>
                                        <p className="text-4xl font-black text-center mb-2" style={{ color: anomalyScore.is_anomaly ? '#ef4444' : '#22c55e' }}>
                                            {anomalyScore.anomaly_score.toFixed(4)}
                                        </p>
                                        <p className="text-sm text-slate-500 text-center mb-4">(Threshold: {anomalyScore.threshold.toFixed(4)})</p>
                                        {anomalyScore.is_anomaly ? (
                                            <p className="text-center font-bold text-red-600">Anomaly Detected! Deviation from normal.</p>
                                        ) : (
                                            <p className="text-center font-bold text-green-600">Normal Operation Detected.</p>
                                        )}
                                    </div>
                                ) : (
                                    <p className="text-slate-500 text-center">Run prediction to calculate anomaly score.</p>
                                )}
                            </div>

                            {/* SHAP Chart */}
                            {shapData && (
                                <div className="bg-white border border-gray-200 rounded-lg shadow-md p-6 mt-8">
                                    <h3 className="text-xl font-semibold mb-2 text-gray-800">Why is it "At Risk"? (XAI)</h3>
                                    <ResponsiveContainer width="100%" height={300}>
                                        <BarChart data={shapData} layout="vertical" margin={{ top: 5, right: 20, left: 120, bottom: 5 }}>
                                            <CartesianGrid stroke="#e5e7eb" />
                                            <XAxis type="number" stroke="#6b7280" />
                                            <YAxis type="category" dataKey="feature" stroke="#6b7280" tick={{ fill: '#374151' }} />
                                            <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #e5e7eb' }} />
                                            <Bar dataKey="importance" fill="#ef4444" name="Contribution to Risk" />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* --- MODALS --- */}
            {isDriftModalOpen && <DriftAnalysisModal machineType="motor" featureName="Torque (Nm)" onClose={() => setIsDriftModalOpen(false)} />}
            {isTicketModalOpen && <MaintenanceTicketModal machineId="Motor-MOT-007" details={`AI detected 'At Risk' status (Health Score: ${prediction?.health_score}). Suspected cause from XAI: High Torque & Tool Wear.`} onClose={() => setIsTicketModalOpen(false)} />}
        </div>
    );
}

export default MotorDashboard;