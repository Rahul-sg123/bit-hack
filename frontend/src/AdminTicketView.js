// src/AdminTicketView.js
import React from 'react';
import { useTickets } from './TicketContext';
import { Link } from 'react-router-dom';

const AdminTicketView = () => {
  const { tickets } = useTickets();

  return (
    <div className="container mx-auto p-8 max-w-7xl">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-800">Maintenance Ticket Dashboard</h1>
        <Link to="/" className="text-indigo-600 hover:underline">&laquo; Back to Fleet Overview</Link>
      </div>
      <div className="bg-white border border-gray-200 rounded-lg shadow-md overflow-hidden">
        <table className="w-full text-left">
          <thead className="bg-gray-50">
            <tr>
              <th className="p-4 font-semibold text-gray-700">Ticket ID</th>
              <th className="p-4 font-semibold text-gray-700">Asset ID</th>
              <th className="p-4 font-semibold text-gray-700">Details</th>
              <th className="p-4 font-semibold text-gray-700">Status</th>
              <th className="p-4 font-semibold text-gray-700">Date Raised</th>
            </tr>
          </thead>
          <tbody>
            {tickets.map(ticket => (
              <tr key={ticket.id} className="border-t border-gray-200">
                <td className="p-4 font-mono">#{ticket.id}</td>
                <td className="p-4 font-semibold">{ticket.machineId}</td>
                <td className="p-4 text-gray-600">{ticket.details}</td>
                <td className="p-4">
                  <span className="bg-red-100 text-red-800 font-medium px-2.5 py-0.5 rounded-full">{ticket.status}</span>
                </td>
                <td className="p-4 text-gray-600">{ticket.date}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default AdminTicketView;