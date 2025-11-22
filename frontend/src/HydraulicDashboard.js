// // src/HydraulicDashboard.js
// import React, { useState, useRef } from 'react';
// import BusinessImpact from './BusinessImpact';
// import ReportGenerator from './ReportGenerator';
// import RecommendationGuide from './RecommendationGuide';
// import MaintenanceTicketModal from './MaintenanceTicketModal';
// import DriftAnalysisModal from './DriftAnalysisModal';
// import axios from 'axios';
// import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
// import './App.css'; // Reuse the same styles

// function HydraulicDashboard() {
//     const reportContentRef = useRef(); // Ref for the PDF content area
//     const [features, setFeatures] = useState({
//         PS1: '', PS2: '', PS3: '', PS4: '', PS5: '', PS6: '',
//         EPS1: '', FS1: '', TS1: '', TS2: '', TS3: '', TS4: '',
//         VS1: '', CE: '', CP: '', SE: ''
//     });

//     const [prediction, setPrediction] = useState(null);
//     const [error, setError] = useState('');
//     const [isTicketModalOpen, setIsTicketModalOpen] = useState(false);
//     const [isDriftModalOpen, setIsDriftModalOpen] = useState(false);

//     const handleChange = (e) => {
//         const { name, value } = e.target;
//         setFeatures(prev => ({ ...prev, [name]: parseFloat(value) || '' }));
//     };

//     const getPrediction = async () => {
//         setError('');
//         setPrediction(null);
//         try {
//             const response = await axios.post('http://127.0.0.1:8000/predict/hydraulic', features);
//             setPrediction(response.data);
//         } catch (err) {
//             setError('Failed to get prediction. Ensure the backend server is running.');
//             console.error(err);
//         }
//     };

//     const isAtRisk = prediction?.status === 'Close to Total Failure';
//     const statusForGuides = isAtRisk ? 'At Risk' : 'Normal';

//     // Data for the live sensor values chart
//     const sensorDataForChart = [
//         { name: 'Pressure 1', value: features.PS1, unit: 'bar' },
//         { name: 'Pressure 2', value: features.PS2, unit: 'bar' },
//         { name: 'Temp 1', value: features.TS1, unit: '°C' },
//         { name: 'Temp 2', value: features.TS2, unit: '°C' },
//     ];

//     return (
//         <div className="px-8 py-12">
//             {/* --- HEADER SECTION --- */}
//             <div className="flex justify-between items-center mb-6">
//                 <h1 className="text-5xl font-black text-slate-900">Hydraulic System HTS-004</h1>
//                 <div className="flex space-x-4">
//                     <button
//                         onClick={() => setIsDriftModalOpen(true)}
//                         className="font-bold text-sky-600 hover:text-sky-500 transition-colors"
//                     >
//                         Check for Model Drift
//                     </button>
//                     <ReportGenerator machineId="Hydraulic-HTS-004" reportContentRef={reportContentRef} />
//                 </div>
//             </div>

//             {/* --- WRAPPER FOR PDF CONTENT --- */}
//             <div ref={reportContentRef} className="p-4 bg-white">
//                 <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

//                     {/* --- LEFT COLUMN: CONTROL PANEL --- */}
//                     <div className="bg-white border border-slate-200 rounded-lg shadow-md p-6 h-fit space-y-6">
//                         <div>
//                             <h3 className="text-xl font-bold text-slate-900 mb-4">Live Sensor Input</h3>
//                             <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
//                                 {Object.keys(features).map(key => (
//                                     <div key={key}>
//                                         <label className="text-sm font-medium text-slate-700">{key}</label>
//                                         <input
//                                             type="number" name={key} value={features[key]} onChange={handleChange} step="0.01"
//                                             className="w-full mt-1 p-2 bg-slate-50 border border-slate-300 rounded-md focus:ring-2 focus:ring-yellow-500 focus:outline-none"
//                                         />
//                                     </div>
//                                 ))}
//                             </div>
//                         </div>

//                         <button onClick={getPrediction} className="w-full bg-yellow-400 text-black font-bold uppercase py-3 rounded-none text-center hover:bg-yellow-500 transition-colors">
//                             Predict Health
//                         </button>

//                         {prediction && (
//                             <div className={`p-4 rounded-lg border-l-4 ${isAtRisk ? 'border-red-500 bg-red-50' : 'border-green-500 bg-green-50'}`}>
//                                 <h3 className="font-bold text-lg text-slate-900">Prediction Result</h3>
//                                 <p className="text-xl mt-2">
//                                     Status: <span className={`font-extrabold ${isAtRisk ? 'text-red-600' : 'text-green-600'}`}>{prediction.status}</span>
//                                 </p>
//                                 <p className="text-xl mt-1">
//                                     Health Score: <strong className="text-slate-900">{prediction.health_score} / 100</strong>
//                                 </p>
//                                 {isAtRisk && (
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
//                            <h3 className="text-xl font-bold text-slate-900 mb-4">Key Operational Data</h3>
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

//                         {prediction && <BusinessImpact status={statusForGuides} costPerHourDowntime={100000} hoursToRepairFailure={12} costOfProactiveMaintenance={200000} />}
//                         {prediction && <RecommendationGuide machineType="motor" status={statusForGuides} />}
//                     </div>
//                 </div>
//             </div>

//             {/* --- MODALS --- */}
//             {isDriftModalOpen && <DriftAnalysisModal machineType="hydraulic" featureName="System Temperature (K)" onClose={() => setIsDriftModalOpen(false)} />}
//             {isTicketModalOpen && <MaintenanceTicketModal machineId="Hydraulic-HTS-004" details={`AI detected '${prediction?.status}' (Health Score: ${prediction?.health_score}). Immediate inspection of hydraulic pressure and cooling system is recommended.`} onClose={() => setIsTicketModalOpen(false)} />}
//         </div>
//     );
// }

// export default HydraulicDashboard;

// src/HydraulicDashboard.js
import React, { useState, useRef } from 'react';
import BusinessImpact from './BusinessImpact';
import ReportGenerator from './ReportGenerator';
import RecommendationGuide from './RecommendationGuide';
import MaintenanceTicketModal from './MaintenanceTicketModal';
import DriftAnalysisModal from './DriftAnalysisModal';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './App.css'; // Reuse the same styles

function HydraulicDashboard() {
    const reportContentRef = useRef(); // Ref for the PDF content area
    const [features, setFeatures] = useState({
        PS1: '', PS2: '', PS3: '', PS4: '', PS5: '', PS6: '',
        EPS1: '', FS1: '', TS1: '', TS2: '', TS3: '', TS4: '',
        VS1: '', CE: '', CP: '', SE: ''
    });

    const [prediction, setPrediction] = useState(null);
    const [error, setError] = useState('');
    const [isTicketModalOpen, setIsTicketModalOpen] = useState(false);
    const [isDriftModalOpen, setIsDriftModalOpen] = useState(false);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFeatures(prev => ({ ...prev, [name]: parseFloat(value) || '' }));
    };

    const getPrediction = async () => {
        setError('');
        setPrediction(null);
        try {
            const response = await axios.post('http://127.0.0.1:8000/predict/hydraulic', features);
            setPrediction(response.data);
        } catch (err) {
            setError('Failed to get prediction. Ensure the backend server is running.');
            console.error(err);
        }
    };

    const isAtRisk = prediction?.status === 'Close to Total Failure';
    const statusForGuides = isAtRisk ? 'At Risk' : 'Normal';

    // Data for the live sensor values chart
    const sensorDataForChart = [
        { name: 'Pressure 1', value: features.PS1, unit: 'bar' },
        { name: 'Pressure 2', value: features.PS2, unit: 'bar' },
        { name: 'Temp 1', value: features.TS1, unit: '°C' },
        { name: 'Temp 2', value: features.TS2, unit: '°C' },
    ];
    const [isDisabled, setIsDisabled] = useState(true);
    return (
        <div className="px-8 py-12">
            {/* --- HEADER SECTION --- */}
            <div className="flex justify-between items-center mb-6">
                <h1 className="text-5xl font-black text-slate-900">Hydraulic System HTS-004</h1>
                <div className="flex space-x-4">
                    <button
                onClick={() => !isDisabled && setIsDriftModalOpen(true)}
                className={`font-bold ${isDisabled ? "text-sky-400 cursor-not-allowed" : "text-sky-600 hover:text-sky-500"} transition-colors`} disabled={isDisabled}
            >
                Check for Model Drift
            </button>
                    <ReportGenerator machineId="Hydraulic-HTS-004" reportContentRef={reportContentRef} />
                </div>
            </div>

            {/* --- WRAPPER FOR PDF CONTENT --- */}
            <div ref={reportContentRef} className="p-4 bg-white">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

                    {/* --- LEFT COLUMN: CONTROL PANEL --- */}
                    <div className="bg-white border border-slate-200 rounded-lg shadow-md p-6 h-fit space-y-6">
                        <div>
                            <h3 className="text-xl font-bold text-slate-900 mb-4">Live Sensor Input</h3>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                {Object.keys(features).map(key => (
                                    <div key={key}>
                                        <label className="text-sm font-medium text-slate-700">{key}</label>
                                        <input
                                            type="number" name={key} value={features[key]} onChange={handleChange} step="0.01"
                                            className="w-full mt-1 p-2 bg-slate-50 border border-slate-300 rounded-md focus:ring-2 focus:ring-yellow-500 focus:outline-none"
                                        />
                                    </div>
                                ))}
                            </div>
                        </div>

                        <button onClick={getPrediction} className="w-full bg-yellow-400 text-black font-bold uppercase py-3 rounded-none text-center hover:bg-yellow-500 transition-colors">
                            Predict Health
                        </button>

                        {prediction && (
                            <div className={`p-4 rounded-lg border-l-4 ${isAtRisk ? 'border-red-500 bg-red-50' : 'border-green-500 bg-green-50'}`}>
                                <h3 className="font-bold text-lg text-slate-900">Prediction Result</h3>
                                <p className="text-xl mt-2">
                                    Status: <span className={`font-extrabold ${isAtRisk ? 'text-red-600' : 'text-green-600'}`}>{prediction.status}</span>
                                </p>
                                <p className="text-xl mt-1">
                                    Health Score: <strong className="text-slate-900">{prediction.health_score} / 100</strong>
                                </p>
                                {isAtRisk && (
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
                           <h3 className="text-xl font-bold text-slate-900 mb-4">Key Operational Data</h3>
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

                        {prediction && <BusinessImpact status={statusForGuides} costPerHourDowntime={100000} hoursToRepairFailure={12} costOfProactiveMaintenance={200000} />}
                        {prediction && <RecommendationGuide machineType="motor" status={statusForGuides} />}
                    </div>
                </div>
            </div>

            {/* --- MODALS --- */}
            {isDriftModalOpen && <DriftAnalysisModal machineType="hydraulic" featureName="System Temperature (K)" onClose={() => setIsDriftModalOpen(false)} />}
            {isTicketModalOpen && <MaintenanceTicketModal machineId="Hydraulic-HTS-004" details={`AI detected '${prediction?.status}' (Health Score: ${prediction?.health_score}). Immediate inspection of hydraulic pressure and cooling system is recommended.`} onClose={() => setIsTicketModalOpen(false)} />}
        </div>
    );
}

export default HydraulicDashboard;