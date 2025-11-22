// src/PrescriptiveAdvisor.js
import React from 'react';
import { motion } from 'framer-motion';

// A small library of SVG icons for our steps
const icons = {
  safety: (
    <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
  ),
  inspect: (
    <svg className="w-6 h-6 text-sky-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
  ),
  analyze: (
    <svg className="w-6 h-6 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" /></svg>
  ),
  schedule: (
    <svg className="w-6 h-6 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
  ),
};

const PrescriptiveAdvisor = ({ recommendations }) => {
  if (!recommendations || recommendations.length === 0) {
    return null;
  }

  return (
    <div className="pt-8 mt-8 border-t border-slate-200">
      <h3 className="text-xl font-bold text-slate-800 flex items-center mb-4">
        <span className="text-2xl mr-3">👨‍🔧</span> Actionable Recommendations
      </h3>
      <div className="space-y-4">
        {recommendations.map((rec, index) => (
          <motion.div
            key={index}
            className={`flex items-start p-4 border rounded-lg shadow-sm ${
                rec.priority === 1 ? 'bg-red-50 border-red-200' :
                rec.priority === 2 ? 'bg-amber-50 border-amber-200' :
                'bg-sky-50 border-sky-200'
            }`}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
          >
            <div className="flex-shrink-0 mt-0.5">
               <span className={`font-bold mr-2 text-xl ${
                   rec.priority === 1 ? 'text-red-600' :
                   rec.priority === 2 ? 'text-amber-600' :
                   'text-sky-600'
               }`}>
                   {rec.priority === 1 ? '❗' : rec.priority === 2 ? '⚠️' : 'ℹ️'}
               </span>
            </div>
            <div>
              <h4 className={`font-bold ${
                  rec.priority === 1 ? 'text-red-800' :
                  rec.priority === 2 ? 'text-amber-800' :
                  'text-sky-800'
              }`}>Priority {rec.priority}:</h4>
              <p className="text-sm text-slate-700">{rec.action}</p>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

export default PrescriptiveAdvisor;