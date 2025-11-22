// src/BusinessImpact.js
import React from 'react';

const BusinessImpact = ({ status, costPerHourDowntime, hoursToRepairFailure, costOfProactiveMaintenance }) => {
  const totalCostOfFailure = costPerHourDowntime * hoursToRepairFailure;
  const predictedSavings = totalCostOfFailure - costOfProactiveMaintenance;

  if (status !== 'At Risk') {
    return null;
  }

  return (
    // --- CHANGES ---
    // Removed background, border, and shadow. Added a top border for separation.
    <div className="pt-8 mt-8 border-t border-slate-200">
      <h3 className="text-xl font-bold mb-4 text-slate-800">Business Impact Analysis</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-center">
        <div className="bg-red-100 p-4 rounded-lg">
          <p className="text-sm text-red-700 font-bold">Estimated Cost of Failure</p>
          <p className="text-3xl font-bold text-red-900">₹{totalCostOfFailure.toLocaleString('en-IN')}</p>
        </div>
        <div className="bg-green-100 p-4 rounded-lg">
          <p className="text-sm text-green-700 font-bold">Predicted Savings</p>
          <p className="text-3xl font-bold text-green-900">₹{predictedSavings.toLocaleString('en-IN')}</p>
        </div>
      </div>
    </div>
  );
};

export default BusinessImpact;