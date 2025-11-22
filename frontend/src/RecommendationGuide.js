// src/RecommendationGuide.js
import React from 'react';
import { motion } from 'framer-motion';

// --- SVG Icons ---
const icons = {
  safety: (
    <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
    </svg>
  ),
  inspect: (
    <svg className="w-6 h-6 text-sky-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
    </svg>
  ),
  analyze: (
    <svg className="w-6 h-6 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" />
    </svg>
  ),
  schedule: (
    <svg className="w-6 h-6 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
    </svg>
  ),
};

// --- Steps Data ---
const recommendationSteps = {
  battery: {
    'At Risk': [
      { icon: 'inspect', title: 'Immediate Diagnostics', description: 'Schedule a full diagnostic test for the battery pack to assess cell health.' },
      { icon: 'safety', title: 'Physical Inspection', description: 'Check for physical signs of swelling, leaking, or corrosion on terminals.' },
      { icon: 'analyze', title: 'Review Usage Patterns', description: 'Analyze recent high-drain activities that could be accelerating degradation.' },
      { icon: 'schedule', title: 'Plan Replacement', description: 'Based on the predicted RUL, schedule a replacement to avoid unplanned downtime.' },
    ],
  },
  motor: {
    'At Risk': [
      { icon: 'safety', title: 'Safety First', description: 'Isolate the machine from power before performing any physical inspection.' },
      { icon: 'analyze', title: 'Review XAI Insights', description: 'Check the "Why is it at risk?" chart to identify the root cause (e.g., high torque, tool wear).' },
      { icon: 'inspect', title: 'Perform Sensory Check', description: 'Listen for unusual noises, check for overheating, and perform a vibration analysis.' },
      { icon: 'schedule', title: 'Schedule Maintenance', description: 'Based on the XAI insights, schedule the appropriate action (e.g., replace tool, adjust load).' },
    ],
  },
};

const RecommendationGuide = ({ machineType, status }) => {
  const steps = recommendationSteps[machineType]?.[status];

  if (!steps) return null;

  return (
    <div className="pt-8 mt-8 border-t border-slate-200">
      <h3 className="text-xl font-bold text-slate-800 flex items-center mb-4">
        Actionable Recommendation Guide
      </h3>

      {/* 2 columns grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {steps.map((step, index) => (
          <motion.div
            key={index}
            className="flex items-start p-4 bg-slate-50 border border-slate-200 rounded-lg transition-all duration-300 hover:shadow-md hover:border-sky-500"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
          >
            <div className="flex-shrink-0 mr-4 mt-1">
              {icons[step.icon] || icons['analyze']}
            </div>
            <div>
              <h4 className="font-bold text-slate-900">{step.title}</h4>
              <p className="text-sm text-slate-600">{step.description}</p>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

export default RecommendationGuide;
