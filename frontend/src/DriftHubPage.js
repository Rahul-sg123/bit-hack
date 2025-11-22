// src/DriftHubPage.js
import React from 'react';
import DriftAnalysisChart from './DriftAnalysisChart';
import { Link } from 'react-router-dom';

const DriftHubPage = () => {
  return (
    <div className="px-8 py-12">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-5xl font-black text-slate-900">Model Health & Drift Hub</h1>
        <Link to="/" className="font-bold text-sky-600 hover:text-sky-500 transition-colors">
          &laquo; Back to Fleet Overview
        </Link>
      </div>
      <p className="text-xl text-slate-600 mb-10">
        Continuously monitor the statistical distribution of live data against the original training data to ensure model reliability.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* You can now place a chart for each machine type */}
        <DriftAnalysisChart machineType="battery" featureName="Capacity (Ah)" />
        <DriftAnalysisChart machineType="motor" featureName="Torque (Nm)" />
        <DriftAnalysisChart machineType="pump" featureName="Pressure (PSI)" />
        <DriftAnalysisChart machineType="hydraulic" featureName="System Temperature (K)" />
      </div>
    </div>
  );
};

export default DriftHubPage;