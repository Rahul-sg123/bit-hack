// src/PatternMinerPage.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

const PatternMinerPage = () => {
  const [patterns, setPatterns] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get('http://127.0.0.1:8000/maintenance-patterns')
      .then(response => {
        setPatterns(response.data.patterns || []);
        setLoading(false);
      })
      .catch(error => {
        console.error("Failed to fetch patterns:", error);
        setLoading(false);
      });
  }, []);

  return (
    <div className="px-8 py-12">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-5xl font-black text-slate-900">Maintenance Pattern Miner</h1>
        <Link to="/" className="font-bold text-sky-600 hover:text-sky-500">&laquo; Back to Fleet</Link>
      </div>
      <p className="text-xl text-slate-600 mb-10">
        Discovering hidden relationships between sensor readings and failure events from historical data.
      </p>

      {loading ? (
        <p className="text-slate-500">Loading patterns...</p>
      ) : (
        <div className="space-y-6">
          {patterns.map((rule, index) => (
            <motion.div
              key={rule.id}
              className="bg-white border border-slate-200 rounded-lg shadow-md p-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <h3 className="text-xl font-bold text-slate-800 mb-3">Discovered Rule #{rule.id}</h3>
              <div className="text-lg text-slate-700 space-y-2">
                <p>
                  <strong className="font-semibold text-slate-900">IF</strong>{' '}
                  {rule.conditions.map((cond, i) => (
                    <React.Fragment key={i}>
                      <span className="bg-slate-100 px-2 py-1 rounded mx-1 font-mono text-sm">{cond}</span>
                      {i < rule.conditions.length - 1 && <strong className="mx-1">AND</strong>}
                    </React.Fragment>
                  ))}
                </p>
                <p>
                  <strong className="font-semibold text-slate-900">THEN</strong> probability of{' '}
                  <span className="bg-red-100 text-red-700 px-2 py-1 rounded mx-1 font-mono text-sm font-bold">{rule.outcome}</span>{' '}
                  increases by approximately <strong className="text-red-600">{(rule.confidence * 100).toFixed(0)}%</strong>.
                </p>
                <p className="text-sm text-slate-500 pt-1">
                  (This pattern is <strong className="text-slate-600">{rule.lift.toFixed(1)}x</strong> more likely than random chance.)
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
};

export default PatternMinerPage;