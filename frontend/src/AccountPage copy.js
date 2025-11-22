// src/AccountPage.js
import React from "react";
import { motion } from "framer-motion";
import ReportGenerator from './ReportGenerator';


const AccountPage = () => {
  // You can later fetch real user data from backend or localStorage
  const user = {
    name: "Admin User",
    email: "admin@example.com",
    lastLogin: "Oct 31, 2025, 3:45 AM",
  };

  // Example: predicted reports data (could come from MongoDB later)
  const reports = [
    {
      id: "REP-2025-001",
      model: "Battery Health Predictor",
      accuracy: "97.5%",
      timestamp: "2025-10-28 16:20",
      status: "Successful",
      reportpage: <ReportGenerator machineId="EV-Battery-EVB-001" />
    },
    {
      id: "REP-2025-002",
      model: "Motor Drift Analyzer",
      accuracy: "93.2%",
      timestamp: "2025-10-29 11:10",
      status: "Successful",
      reportpage: <ReportGenerator machineId="REP-2025-002" />
    },
    {
      id: "REP-2025-003",
      model: "Pump Failure Predictor",
      accuracy: "95.7%",
      timestamp: "2025-10-30 08:50",
      status: "Warning",
      reportPage: <ReportGenerator machineId="REP-2025-003" />
    },
  ];

  return (
    <div className="min-h-screen flex items-center justify-center py-10 px-4">
      <motion.div
        className="bg-white rounded-2xl p-10 w-full max-w-3xl"
      >
        <h1 className="text-3xl font-extrabold text-gray-800 mb-6 text-center">
          Account Overview
        </h1>

        {/* --- User Info --- */}
        <div className="bg-gray-100 rounded-xl p-5 mb-8">
          <h2 className="text-lg font-semibold text-gray-700 mb-2">
            Account Info
          </h2>
          <p className="text-gray-600">
            <strong>Name:</strong> {user.name}
          </p>
          <p className="text-gray-600">
            <strong>Email:</strong> {user.email}
          </p>
          <p className="text-gray-600">
            <strong>Last Login:</strong> {user.lastLogin}
          </p>
        </div>

        {/* --- Predicted Reports --- */}
        <h2 className="text-xl font-bold text-gray-800 mb-4">
          Predicted Reports
        </h2>
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white border border-gray-200 rounded-xl shadow-sm">
            <thead>
              <tr className="bg-indigo-100 text-gray-700 text-left">
                <th className="p-3">Report ID</th>
                <th className="p-3">Model Name</th>
                <th className="p-3">Accuracy</th>
                <th className="p-3">Timestamp</th>
                <th className="p-3">Status</th>
                <th className="p-3">Report</th>
              </tr>
            </thead>
            <tbody>
              {reports.map((r) => (
                <tr key={r.id} className="border-t hover:bg-gray-50">
                  <td className="p-3">{r.id}</td>
                  <td className="p-3">{r.model}</td>
                  <td className="p-3 font-semibold text-green-600">
                    {r.accuracy}
                  </td>
                  <td className="p-3">{r.timestamp}</td>
                  <td
                    className={`p-3 font-bold ${
                      r.status === "Warning"
                        ? "text-yellow-600"
                        : "text-green-600"
                    }`}
                  >
                    {r.status}
                  </td>
                  <td className="">
                    <ReportGenerator machineId="EV-Battery-EVB-001" reportContentRef={r.report} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>
    </div>
  );
};

export default AccountPage;