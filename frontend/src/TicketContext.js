// src/TicketContext.js
import React, { createContext, useState, useContext } from 'react';

const TicketContext = createContext();

// Custom hook to easily use the ticket context
export const useTickets = () => useContext(TicketContext);

// Provider component that holds the state and functions
export const TicketProvider = ({ children }) => {
  const [tickets, setTickets] = useState([
    // Optional: Start with some dummy data
    { id: 1, machineId: 'PMP-003', details: "Initial Check: AI detected 'Caution' status. Vibration levels higher than normal.", status: 'Open', date: new Date().toLocaleDateString() },
  ]);

  // Function to add a new ticket to the list
  const addTicket = (ticketData) => {
    const newTicket = { 
      ...ticketData, 
      id: tickets.length + 1, // Simple ID generation
      status: 'Open', 
      date: new Date().toLocaleDateString() // Add current date
    };
    // Add the new ticket to the beginning of the array
    setTickets(prevTickets => [newTicket, ...prevTickets]); 
    console.log("Ticket Added:", newTicket); // Add console log for debugging
    console.log("Current Tickets:", [newTicket, ...tickets]); // Show updated list
  };

  // Provide the tickets list and the addTicket function to children
  return (
    <TicketContext.Provider value={{ tickets, addTicket }}>
      {children}
    </TicketContext.Provider>
  );
};