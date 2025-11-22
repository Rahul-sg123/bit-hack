// src/ModelOpsCenter.js
import React, { useState } from 'react';
import axios from 'axios';
import DriftAnalysisChart from './DriftAnalysisChart';
import { Link } from 'react-router-dom';

const ModelOpsCenter = () => {
  const [isRetraining, setIsRetraining] = useState(false);
  const [logMessages, setLogMessages] = useState([]);
  const [retrainingComplete, setRetrainingComplete] = useState(false);

  const handleRetrainClick = async () => {
    setIsRetraining(true);
    setRetrainingComplete(false);
    setLogMessages([]); // Clear previous logs

    // Simulate a live log stream
    const logs = [
      "[INFO] Retraining request received for 'Motor' model...",
      "[DATA] Fetching new data batches from the last 7 days...",
      "[DATA] Data validation complete. 5,000 new records found.",
      "[TRAIN] Initializing retraining pipeline with new data...",
      "[TRAIN] Epoch 1/100 - Loss: 0.15, Accuracy: 95.2%",
      "[TRAIN] Epoch 50/100 - Loss: 0.08, Accuracy: 97.8%",
      "[TRAIN] Epoch 100/100 - Loss: 0.05, Accuracy: 98.5%",
      "[VALIDATE] New model version v1.1 validation complete.",
      "[DEPLOY] Deploying new model to production environment...",
      "[SUCCESS] Model v1.1 is now live. System is using the updated model."
    ];

    // Call the backend to "officially" start the process
    await axios.get('http://127.0.0.1:8000/retrain/motor');

    // Add each log message with a delay to simulate a real process
    logs.forEach((log, index) => {
      setTimeout(() => {
        setLogMessages(prev => [...prev, log]);
      }, index * 700); // 700ms delay between each log line
    });

    // After all logs are displayed, mark as complete
    setTimeout(() => {
      setIsRetraining(false);
      setRetrainingComplete(true);
    }, logs.length * 700);
  };

  return (
    <div className="container mx-auto p-8 max-w-7xl">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-800">Model Operations Center</h1>
        <Link to="/" className="text-indigo-600 hover:underline">&laquo; Back to Fleet Overview</Link>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Column: Drift Analysis */}
        <DriftAnalysisChart machineType="motor" featureName="Torque (Nm)" />

        {/* Right Column: Retraining Actions */}
        <div className="bg-white border border-gray-200 rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold mb-4 text-gray-800">Model Retraining</h3>
          <p className="text-gray-600 mb-4">
            Since data drift has been detected, the model's accuracy may be affected. Initiate a retraining pipeline to update the model with the latest data.
          </p>
          <button 
            onClick={handleRetrainClick} 
            disabled={isRetraining}
            className="w-full bg-blue-600 text-white font-bold py-3 rounded-md hover:bg-blue-500 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {isRetraining ? 'Retraining in Progress...' : 'Initiate Retraining for Motor Model'}
          </button>

          {/* Live Log Terminal */}
          {(isRetraining || retrainingComplete) && (
            <div className="mt-6 bg-gray-800 text-white font-mono text-sm rounded-lg p-4 h-72 overflow-y-auto">
              {logMessages.map((msg, index) => (
                <p key={index} className="animate-fadeIn">{msg}</p>
              ))}
              {isRetraining && <div className="animate-pulse">_</div>}
              {retrainingComplete && <p className="text-green-400 font-bold mt-2">PROCESS COMPLETE</p>}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ModelOpsCenter;