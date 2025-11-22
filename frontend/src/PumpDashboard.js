// // src/PumpDashboard.js
// import React, { useState, useRef } from 'react';
// import BusinessImpact from './BusinessImpact';
// import ReportGenerator from './ReportGenerator';
// import RecommendationGuide from './RecommendationGuide';
// import MaintenanceTicketModal from './MaintenanceTicketModal';
// import DriftAnalysisModal from './DriftAnalysisModal';
// import axios from 'axios';
// import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
// import './App.css'; // Reuse the same styles

// function PumpDashboard() {
//     const reportContentRef = useRef(); // Ref for the PDF content area
//     const [features, setFeatures] = useState({
//         air_temperature_k: 300.2,
//         process_temperature_k: 310.8,
//         rotational_speed_rpm: 1200, // Represents Flow Rate
//         torque_nm: 60.5,           // Represents Pressure
//         tool_wear_min: 25,         // Represents Vibration
//     });

//     const [prediction, setPrediction] = useState(null);
//     const [error, setError] = useState('');
//     const [isTicketModalOpen, setIsTicketModalOpen] = useState(false);
//     const [isDriftModalOpen, setIsDriftModalOpen] = useState(false);

//     const handleChange = (e) => {
//         const { name, value } = e.target;
//         setFeatures(prev => ({ ...prev, [name]: parseFloat(value) }));
//     };

//     const getPrediction = async () => {
//         setError('');
//         setPrediction(null);
//         try {
//             // Still calls the /predict/motor endpoint as a proxy
//             const response = await axios.post('http://127.0.0.1:8000/predict/motor', features);
//             setPrediction(response.data);
//         } catch (err) {
//             setError('Failed to get prediction. Ensure the backend server is running.');
//             console.error(err);
//         }
//     };

//     // Data for the live sensor values chart
//     const sensorDataForChart = [
//       { name: 'Flow Rate', value: features.rotational_speed_rpm, unit: 'L/min' },
//       { name: 'Pressure', value: features.torque_nm, unit: 'PSI' },
//       { name: 'Vibration', value: features.tool_wear_min, unit: 'mm/s' },
//     ];

//     return (
//         <div className="px-8 py-12">
//             {/* --- HEADER SECTION --- */}
//             <div className="flex justify-between items-center mb-6">
//                 <h1 className="text-5xl font-black text-slate-900">Compressor Pump PMP-003</h1>
//                 <div className="flex space-x-4">
//                     <button
//                         onClick={() => setIsDriftModalOpen(true)}
//                         className="font-bold text-sky-600 hover:text-sky-500 transition-colors"
//                     >
//                         Check for Model Drift
//                     </button>
//                     <ReportGenerator machineId="Pump-PMP-003" reportContentRef={reportContentRef} />
//                 </div>
//             </div>

//             {/* --- WRAPPER FOR PDF CONTENT --- */}
//             <div ref={reportContentRef} className="p-4 bg-white">
//                 <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

//                     {/* --- LEFT COLUMN: CONTROL PANEL --- */}
//                     <div className="bg-white border border-slate-200 rounded-lg shadow-md p-6 h-fit space-y-6">
//                         <div>
//                             <h3 className="text-xl font-bold text-slate-900 mb-4">Live Sensor Input</h3>
//                             <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
//                                 <div className="input-group">
//                                     <label className="text-sm font-medium text-slate-700">Air Temperature (K)</label>
//                                     <input type="number" name="air_temperature_k" value={features.air_temperature_k} onChange={handleChange} className="w-full mt-1 p-2 bg-slate-50 border border-slate-300 rounded-md focus:ring-2 focus:ring-yellow-500 focus:outline-none"/>
//                                 </div>
//                                 <div className="input-group">
//                                     <label className="text-sm font-medium text-slate-700">Process Temperature (K)</label>
//                                     <input type="number" name="process_temperature_k" value={features.process_temperature_k} onChange={handleChange} className="w-full mt-1 p-2 bg-slate-50 border border-slate-300 rounded-md focus:ring-2 focus:ring-yellow-500 focus:outline-none"/>
//                                 </div>
//                                 <div className="input-group">
//                                     <label className="text-sm font-medium text-slate-700">Flow Rate (L/min)</label>
//                                     <input type="number" name="rotational_speed_rpm" value={features.rotational_speed_rpm} onChange={handleChange} className="w-full mt-1 p-2 bg-slate-50 border border-slate-300 rounded-md focus:ring-2 focus:ring-yellow-500 focus:outline-none"/>
//                                 </div>
//                                 <div className="input-group">
//                                     <label className="text-sm font-medium text-slate-700">Pressure (PSI)</label>
//                                     <input type="number" name="torque_nm" value={features.torque_nm} onChange={handleChange} className="w-full mt-1 p-2 bg-slate-50 border border-slate-300 rounded-md focus:ring-2 focus:ring-yellow-500 focus:outline-none"/>
//                                 </div>
//                                 <div className="input-group col-span-2">
//                                     <label className="text-sm font-medium text-slate-700">Vibration (mm/s)</label>
//                                     <input type="number" name="tool_wear_min" value={features.tool_wear_min} onChange={handleChange} className="w-full mt-1 p-2 bg-slate-50 border border-slate-300 rounded-md focus:ring-2 focus:ring-yellow-500 focus:outline-none"/>
//                                 </div>
//                             </div>
//                         </div>

//                         <button onClick={getPrediction} className="w-full bg-yellow-400 text-black font-bold uppercase py-3 rounded-none text-center hover:bg-yellow-500 transition-colors">
//                             Predict Health
//                         </button>

//                         {prediction && (
//                             <div className={`p-4 rounded-lg border-l-4 ${prediction.status === 'At Risk' ? 'border-red-500 bg-red-50' : 'border-green-500 bg-green-50'}`}>
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
//                         {error && <p className="text-red-600 mt-4">{error}</p>}
//                     </div>

//                     {/* --- RIGHT COLUMN: VISUALIZATIONS & ANALYSIS --- */}
//                     <div className="space-y-8">
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

//                         {prediction && <BusinessImpact status={prediction.status} costPerHourDowntime={30000} hoursToRepairFailure={6} costOfProactiveMaintenance={40000} />}
//                         {prediction && <RecommendationGuide machineType="motor" status={prediction.status} />}
//                     </div>
//                 </div>
//             </div>

//             {/* --- MODALS --- */}
//             {isDriftModalOpen && <DriftAnalysisModal machineType="pump" featureName="Pressure (PSI)" onClose={() => setIsDriftModalOpen(false)} />}
//             {isTicketModalOpen && <MaintenanceTicketModal machineId="Pump-PMP-003" details={`AI detected 'At Risk' status (Health Score: ${prediction?.health_score}). Recommend checking for high pressure or vibration anomalies.`} onClose={() => setIsTicketModalOpen(false)} />}
//         </div>
//     );
// }

// export default PumpDashboard;

// src/PumpDashboard.js
import React, { useState, useRef } from 'react';
import BusinessImpact from './BusinessImpact';
import ReportGenerator from './ReportGenerator';
import RecommendationGuide from './RecommendationGuide';
import MaintenanceTicketModal from './MaintenanceTicketModal';
import DriftAnalysisModal from './DriftAnalysisModal';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './App.css'; // Reuse the same styles

function PumpDashboard({isLoggedIn}) {
    const reportContentRef = useRef(); // Ref for the PDF content area
    const [features, setFeatures] = useState({
        air_temperature_k: 300.2,
        process_temperature_k: 310.8,
        rotational_speed_rpm: 1200, // Represents Flow Rate
        torque_nm: 60.5,           // Represents Pressure
        tool_wear_min: 25,         // Represents Vibration
    });

    const [prediction, setPrediction] = useState(null);
    const [error, setError] = useState('');
    const [isTicketModalOpen, setIsTicketModalOpen] = useState(false);
    const [isDriftModalOpen, setIsDriftModalOpen] = useState(false);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFeatures(prev => ({ ...prev, [name]: parseFloat(value) }));
    };

    const getPrediction = async () => {
        setError('');
        setPrediction(null);
        try {
            // Still calls the /predict/motor endpoint as a proxy
            const response = await axios.post('http://127.0.0.1:8000/predict/motor', features);
            setPrediction(response.data);
        } catch (err) {
            setError('Failed to get prediction. Ensure the backend server is running.');
            console.error(err);
        }
    };

    // Data for the live sensor values chart
    const sensorDataForChart = [
      { name: 'Flow Rate', value: features.rotational_speed_rpm, unit: 'L/min' },
      { name: 'Pressure', value: features.torque_nm, unit: 'PSI' },
      { name: 'Vibration', value: features.tool_wear_min, unit: 'mm/s' },
    ];
    const [isDisabled, setIsDisabled] = useState(true);

    return (
        <div className="px-8 py-12">
            {/* --- HEADER SECTION --- */}
            <div className="flex justify-between items-center mb-6">
                <h1 className="text-5xl font-black text-slate-900">Compressor Pump PMP-003</h1>
                <div className="flex space-x-4">
                    <button
                onClick={() => !isDisabled && setIsDriftModalOpen(true)}
                className={`font-bold ${!isLoggedIn ? "text-sky-400 cursor-not-allowed" : "text-sky-600 hover:text-sky-500"} transition-colors`} disabled={isDisabled}
            >
                Check for Model Drift
            </button>
                    <ReportGenerator machineId="Pump-PMP-003" reportContentRef={reportContentRef} />
                </div>
            </div>

            {/* --- WRAPPER FOR PDF CONTENT --- */}
            <div ref={reportContentRef} className="p-4 bg-white">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

                    {/* --- LEFT COLUMN: CONTROL PANEL --- */}
                    <div className="bg-white border border-slate-200 rounded-lg shadow-md p-6 h-fit space-y-6">
                        <div>
                            <h3 className="text-xl font-bold text-slate-900 mb-4">Live Sensor Input</h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div className="input-group">
                                    <label className="text-sm font-medium text-slate-700">Air Temperature (K)</label>
                                    <input type="number" name="air_temperature_k" value={features.air_temperature_k} onChange={handleChange} className="w-full mt-1 p-2 bg-slate-50 border border-slate-300 rounded-md focus:ring-2 focus:ring-yellow-500 focus:outline-none"/>
                                </div>
                                <div className="input-group">
                                    <label className="text-sm font-medium text-slate-700">Process Temperature (K)</label>
                                    <input type="number" name="process_temperature_k" value={features.process_temperature_k} onChange={handleChange} className="w-full mt-1 p-2 bg-slate-50 border border-slate-300 rounded-md focus:ring-2 focus:ring-yellow-500 focus:outline-none"/>
                                </div>
                                <div className="input-group">
                                    <label className="text-sm font-medium text-slate-700">Flow Rate (L/min)</label>
                                    <input type="number" name="rotational_speed_rpm" value={features.rotational_speed_rpm} onChange={handleChange} className="w-full mt-1 p-2 bg-slate-50 border border-slate-300 rounded-md focus:ring-2 focus:ring-yellow-500 focus:outline-none"/>
                                </div>
                                <div className="input-group">
                                    <label className="text-sm font-medium text-slate-700">Pressure (PSI)</label>
                                    <input type="number" name="torque_nm" value={features.torque_nm} onChange={handleChange} className="w-full mt-1 p-2 bg-slate-50 border border-slate-300 rounded-md focus:ring-2 focus:ring-yellow-500 focus:outline-none"/>
                                </div>
                                <div className="input-group col-span-2">
                                    <label className="text-sm font-medium text-slate-700">Vibration (mm/s)</label>
                                    <input type="number" name="tool_wear_min" value={features.tool_wear_min} onChange={handleChange} className="w-full mt-1 p-2 bg-slate-50 border border-slate-300 rounded-md focus:ring-2 focus:ring-yellow-500 focus:outline-none"/>
                                </div>
                            </div>
                        </div>

                        <button onClick={getPrediction} className="w-full bg-yellow-400 text-black font-bold uppercase py-3 rounded-none text-center hover:bg-yellow-500 transition-colors">
                            Predict Health
                        </button>

                        {prediction && (
                            <div className={`p-4 rounded-lg border-l-4 ${prediction.status === 'At Risk' ? 'border-red-500 bg-red-50' : 'border-green-500 bg-green-50'}`}>
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
                        {error && <p className="text-red-600 mt-4">{error}</p>}
                    </div>

                    {/* --- RIGHT COLUMN: VISUALIZATIONS & ANALYSIS --- */}
                    <div className="space-y-8">
                        <div className="bg-white border border-slate-200 rounded-lg shadow-md p-6">
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

                        {prediction && <BusinessImpact status={prediction.status} costPerHourDowntime={30000} hoursToRepairFailure={6} costOfProactiveMaintenance={40000} />}
                        {prediction && <RecommendationGuide machineType="motor" status={prediction.status} />}
                    </div>
                </div>
            </div>

            {/* --- MODALS --- */}
            {isDriftModalOpen && <DriftAnalysisModal machineType="pump" featureName="Pressure (PSI)" onClose={() => setIsDriftModalOpen(false)} />}
            {isTicketModalOpen && <MaintenanceTicketModal machineId="Pump-PMP-003" details={`AI detected 'At Risk' status (Health Score: ${prediction?.health_score}). Recommend checking for high pressure or vibration anomalies.`} onClose={() => setIsTicketModalOpen(false)} />}
        </div>
    );
}

export default PumpDashboard;