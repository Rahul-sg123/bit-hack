// src/MaintenanceTicketModal.js
import React from 'react';
import { useTickets } from './TicketContext'; // <-- Import the hook

const MaintenanceTicketModal = ({ machineId, details, onClose }) => {
  const { addTicket } = useTickets(); // <-- Get the addTicket function

  const handleSubmit = () => {
    console.log("Submit button clicked. Adding ticket:", { machineId, details }); // Debug log
    addTicket({ machineId, details }); // <-- Call the function to add the ticket

    alert(`✅ Maintenance ticket for ${machineId} created!`);
    onClose(); // Close the modal
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white p-8 rounded-lg shadow-2xl w-full max-w-lg">
        <h2 className="text-2xl font-bold mb-4 text-gray-800">Create New Maintenance Ticket</h2>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Asset ID</label>
            <input type="text" readOnly value={machineId} className="w-full mt-1 p-2 bg-gray-100 border border-gray-300 rounded-md" />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Issue Details (AI Generated)</label>
            <textarea readOnly value={details} rows="4" className="w-full mt-1 p-2 bg-gray-100 border border-gray-300 rounded-md" />
          </div>
          <div className="flex justify-end space-x-4 pt-4">
            <button onClick={onClose} className="px-4 py-2 bg-gray-200 rounded-md">Cancel</button>
            <button onClick={handleSubmit} className="px-4 py-2 bg-indigo-600 text-white rounded-md">Submit Ticket</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MaintenanceTicketModal;