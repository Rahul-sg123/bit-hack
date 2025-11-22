// // src/App.js
// import React, { useState } from 'react';
// import { motion } from 'framer-motion';
// import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
// import BatteryDashboard from './BatteryDashboard';
// import PatternMinerPage from './PatternMinerPage';
// import MotorDashboard from './MotorDashboard';
// import PumpDashboard from './PumpDashboard';
// import HydraulicDashboard from './HydraulicDashboard';
// import AdminTicketView from './AdminTicketView';
// import ModelOpsCenter from './ModelOpsCenter';
// import ClusteringInsights from './ClusteringInsights';
// import MotorClusteringPage from './MotorClusteringPage';
// import KalmanFilterPage from './KalmanFilterPage';
// import HeroBanner from './components/HeroBanner';
// import DriftHubPage from './DriftHubPage'; // Ensure this is imported
// import './App.css';

// // --- Hardcoded fleet data (ID added for React keys) ---
// const fleetData = [
//   { id: 'EVB-001', type: 'EV Battery', link: '/machine/battery/EVB-001', image: '/images/battery.jpg' },
//   { id: 'MOT-007', type: 'Electric Motor', link: '/machine/motor/MOT-007', image: '/images/motor.jpg' },
//   { id: 'PMP-003', type: 'Compressor Pump', link: '/machine/pump/PMP-003', image: '/images/pump.jpg' },
//   { id: 'UPS-SRV-2', type: 'Server UPS', link: '/machine/battery/UPS-SRV-2', image: '/images/ups.jpg' },
//   { id: 'CNV-BELT-A', type: 'Conveyor', link: '/machine/motor/CNV-BELT-A', image: '/images/conveyor.jpg' },
//   { id: 'HTS-004', type: 'Hydraulic System', link: '/machine/hydraulic/HTS-004', image: '/images/hydraulic.jpg' },
// ];

// // --- JCB-THEMED FleetOverview component ---
// const FleetOverview = () => (
//   <div id="fleet-overview" className="px-8 py-12">
//     <h1 className="text-5xl font-black text-slate-900 mb-2">Machine Health Hub</h1>
//     <p className="text-xl text-slate-600 mb-10">Unified monitoring for diverse industrial assets.</p>

//     <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-8">
//       {fleetData.map((machine, index) => (
//         <motion.div
//           key={machine.id} // Using the unique ID for the key
//           className="bg-white border border-slate-200 rounded-none shadow-md flex flex-col overflow-hidden transition-shadow duration-300 hover:shadow-2xl"
//           initial={{ opacity: 0, y: 20 }}
//           animate={{ opacity: 1, y: 0 }}
//           transition={{ duration: 0.5, delay: index * 0.05 }}
//         >
//           <div className="h-48 bg-slate-100 flex items-center justify-center p-4 border-b-4 border-yellow-400">
//             {machine.image && (
//               <img src={machine.image} alt={machine.type} className="max-h-full max-w-full object-contain" />
//             )}
//           </div>

//           <div className="p-6 flex-grow flex flex-col">
//             <h2 className="text-xl font-bold text-slate-900">{machine.type}</h2>
//             <div className="flex-grow" />
//             <Link to={machine.link} className="mt-4 bg-yellow-400 text-black font-bold uppercase py-3 rounded-none text-center hover:bg-yellow-500 transition-colors block w-full">
//               Analyze
//             </Link>
//           </div>
//         </motion.div>
//       ))}
//     </div>
//   </div>
// );

// // --- MAIN App Component ---
// function App() {
//   const [isMenuOpen, setIsMenuOpen] = useState(false);

//   // All navigation links, now including the Drift Hub
//   const navLinks = (
//     <>
//       <Link to="/kalman-filter" className="font-bold text-slate-700 hover:text-yellow-500 transition-colors">Noise Filter</Link>
//       {/* --- ADDED THIS LINK --- */}
      
//       <Link to="/patterns" className="font-bold text-slate-700 hover:text-yellow-500 transition-colors">Failure Patterns</Link>
//       {/* <Link to="/drift-hub" className="font-bold text-slate-700 hover:text-yellow-500 transition-colors">Drift Hub</Link> */}
//       <div className="group relative">
//         <button className="font-bold text-slate-700 hover:text-yellow-500 transition-colors inline-flex items-center">
//           <span>Clustering Insights</span>
//           <svg className="fill-current h-4 w-4 ml-1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" /> </svg>
//         </button>
//         <ul className="absolute hidden text-slate-700 pt-1 group-hover:block w-48 bg-white border border-slate-200 rounded-md shadow-lg z-20">
//           <li><Link to="/clustering-insights/battery" className="block px-4 py-2 hover:bg-slate-100">Batteries</Link></li>
//           <li><Link to="/clustering-insights/motor" className="block px-4 py-2 hover:bg-slate-100">Motors</Link></li>
//         </ul>
//       </div>
//       <Link to="/model-ops" className="font-bold text-slate-700 hover:text-yellow-500 transition-colors">Model Ops</Link>
//       <Link to="/admin/tickets" className="font-bold text-slate-700 hover:text-yellow-500 transition-colors">Ticket Admin</Link>
//     </>
//   );

//   return (
//     <Router>
//       <div className="App">
//         <header className="bg-white border-b border-slate-200 p-4 sticky top-0 z-50 shadow-md">
//           <div className="px-4 sm:px-6 lg:px-8 flex justify-between items-center">
//             <Link to="/" className="text-3xl font-black text-slate-900">
//               Smart<span className="text-yellow-500">Predict</span>
//             </Link>
//             <nav className="hidden md:flex items-center space-x-8">{navLinks}</nav>
//             <div className="md:hidden">
//               <button onClick={() => setIsMenuOpen(!isMenuOpen)}>
//                 <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16m-7 6h7" /></svg>
//               </button>
//             </div>
//           </div>
//           {isMenuOpen && (
//             <motion.div className="md:hidden" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
//               <nav className="flex flex-col items-center space-y-4 p-4">{navLinks}</nav>
//             </motion.div>
//           )}
//         </header>
//         <main>
//           <Routes>
//             <Route path="/" element={<><HeroBanner /><FleetOverview /></>} />
//             <Route path="/machine/battery/:id" element={<BatteryDashboard />} />
//             <Route path="/machine/motor/:id" element={<MotorDashboard />} />
//             <Route path="/machine/pump/:id" element={<PumpDashboard />} />
//             <Route path="/machine/hydraulic/:id" element={<HydraulicDashboard />} />
//             <Route path="/admin/tickets" element={<AdminTicketView />} />
//             <Route path="/model-ops" element={<ModelOpsCenter />} />
//             <Route path="/clustering-insights/battery" element={<ClusteringInsights />} />
//             <Route path="/clustering-insights/motor" element={<MotorClusteringPage />} />
//             <Route path="/kalman-filter" element={<KalmanFilterPage />} />
//             <Route path="/patterns" element={<PatternMinerPage />} />
//             {/* --- ADDED THIS ROUTE --- */}
//             <Route path="/drift-hub" element={<DriftHubPage />} />
//           </Routes>
//         </main>
//       </div>
//     </Router>
//   );
// }

// export default App;

import React, { useState } from "react";
import { motion } from "framer-motion";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import BatteryDashboard from "./BatteryDashboard";
import MotorDashboard from "./MotorDashboard";
import PumpDashboard from "./PumpDashboard";
import HydraulicDashboard from "./HydraulicDashboard";
import AdminTicketView from "./AdminTicketView";
import ModelOpsCenter from "./ModelOpsCenter";
import ClusteringInsights from "./ClusteringInsights";
import MotorClusteringPage from "./MotorClusteringPage";
import KalmanFilterPage from "./KalmanFilterPage";
import HeroBanner from "./components/HeroBanner";
import DriftHubPage from "./DriftHubPage";
import "./App.css";
import loginImg from "./loginimg.jpg";
import Why from "./why.js";
import Footer from "./footer.js";
import AccountPage from "./AccountPage";
import PatternMinerPage from './PatternMinerPage';
// import LoginPage from "./LoginPage"

const fleetData = [
  { id: "EVB-001", type: "EV Battery", link: "/machine/battery/EVB-001", image: "/images/battery.jpg" },
  { id: "MOT-007", type: "Electric Motor", link: "/machine/motor/MOT-007", image: "/images/motor.jpg" },
  { id: "PMP-003", type: "Compressor Pump", link: "/machine/pump/PMP-003", image: "/images/pump.jpg" },
  { id: "UPS-SRV-2", type: "Server UPS", link: "/machine/battery/UPS-SRV-2", image: "/images/ups.jpg" },
  { id: "CNV-BELT-A", type: "Conveyor", link: "/machine/motor/CNV-BELT-A", image: "/images/conveyor.jpg" },
  { id: "HTS-004", type: "Hydraulic System", link: "/machine/hydraulic/HTS-004", image: "/images/hydraulic.jpg" },
];

// FleetOverview Component (unchanged)
const FleetOverview = ({ isLoggedIn }) => (
  <div className="relative overflow-hidden">
    <div className="absolute inset-0 bg-[url('./bgg.jpg')] bg-cover bg-center"></div>
    <div className="absolute inset-0 bg-black/60 z-0"></div>
    <div id="fleet-overview" className="px-8 py-12 relative z-10 max-w-7xl mx-auto">
      <h1 className="text-5xl font-black text-white mb-2">Machine Health Hub</h1>
      <p className="text-xl text-white mb-10">Unified monitoring for diverse industrial assets.</p>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-8">
        {fleetData.map((machine, index) => (
          <motion.div
            key={machine.id}
            className="relative bg-white/70 border border-slate-200 rounded-lg shadow-md flex flex-col overflow-hidden hover:shadow-3xl"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.05 }}
          >
            <div className="h-48 flex items-center justify-center p-4 border-b-4 border-yellow-400">
              {machine.image && (
                <img src={machine.image} alt={machine.type} className="max-h-full max-w-full object-contain" />
              )}
            </div>
            <div className="p-6 flex-grow flex flex-col">
              <h2 className="text-xl font-bold text-slate-900">{machine.type}</h2>
              <div className="flex-grow" />
              <Link
                to={machine.link}
                className="mt-4 bg-yellow-400 text-black font-bold uppercase py-3 rounded-md text-center hover:bg-yellow-500 transition-colors block w-full"
              >
                Analyze
              </Link>
            </div>

            {/* Overlay Example for Non-Logged User */}
            {!isLoggedIn && (
              <div className="absolute inset-0 bg-white/70 backdrop-blur-[2px] flex flex-col items-center justify-center rounded-lg">
                <p className="text-gray-700 font-semibold mb-2">Login required to access this</p>
                <Link
                  to="/account"
                  className="bg-yellow-400 text-black font-semibold px-4 py-2 rounded-lg hover:bg-yellow-500 transition"
                >
                  Go to Login
                </Link>
              </div>
            )}
          </motion.div>
        ))}
      </div>
    </div>
  </div>
);

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const navLinks = (
    <>
      <Link to="/about" className="text-xl font-bold text-slate-700 hover:text-yellow-500 transition-colors">
        About
      </Link>

      <div className="group relative">
        <button className="text-xl font-bold text-slate-700 hover:text-yellow-500 inline-flex items-center">
          <span>Features</span>
          <svg className="fill-current h-4 w-4 ml-1" viewBox="0 0 20 20">
            <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
          </svg>
        </button>
        <ul className="absolute hidden group-hover:block bg-white border rounded-md shadow-lg z-20 w-48">
          <li className="relative">
            <Link to="/model-ops" className="block px-4 py-2 hover:bg-slate-100">
              Model Ops
            </Link>
            {/* Overlay for restricted feature */}
            {!isLoggedIn && (
              <div className="absolute inset-0 bg-white/70 flex items-center justify-center rounded-md">
                <p className="text-gray-600 text-sm font-semibold">Login to access</p>
              </div>
            )}
          </li>
          <li>
            <Link to="/admin/tickets" className="block px-4 py-2 hover:bg-slate-100">
              Ticket Admin
            </Link>
          </li>
          <li>
            <Link to="/kalman-filter" className="block px-4 py-2 hover:bg-slate-100">
              Noise Filter
            </Link>
          </li>
          <li>
            <Link to="/patterns" className="font-bold text-slate-700 hover:text-yellow-500 transition-colors">Failure Patterns</Link>
          </li>
        </ul>
      </div>

      <div className="group relative">
        <button className="text-xl font-bold text-slate-700 hover:text-yellow-500 inline-flex items-center">
          <span>Clustering Insights</span>
          <svg className="fill-current h-4 w-4 ml-1" viewBox="0 0 20 20">
            <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
          </svg>
        </button>
        <ul className="absolute hidden group-hover:block bg-white border rounded-md shadow-lg z-20 w-48">
          <li><Link to="/clustering-insights/battery" className="block px-4 py-2 hover:bg-slate-100">Batteries</Link></li>
          <li><Link to="/clustering-insights/motor" className="block px-4 py-2 hover:bg-slate-100">Motors</Link></li>
        </ul>
      </div>

      <Link
        to="/account"
        className="flex items-center justify-center w-10 h-10 rounded-full bg-gray-200 text-slate-700 hover:bg-gray-300"
        title="My Account"
      >
        <img src={loginImg} alt="account" className="w-20 h-20 object-contain" />
      </Link>

      <button
        onClick={() => setIsLoggedIn(!isLoggedIn)}
        className="bg-yellow-400 text-black font-semibold px-3 py-1 rounded-lg hover:bg-yellow-500"
      >
        {isLoggedIn ? "Logout" : "Login"}
      </button>
    </>
  );

// if (!isLoggedIn) {
//   return <LoginPage onLogin={() => setIsLoggedIn(true)} />;
// }

  return (
    <Router>
      <div className="App">
        {/* --- Navbar --- */}
        <header className="bg-white border-b border-slate-200 p-2 sticky top-0 z-50 shadow-md">
          <div className="px-4 sm:px-6 lg:px-8 flex justify-between items-center">
            <Link to="/" className="text-3xl font-black text-slate-900">
              Smart<span className="text-yellow-500">Predict</span>
            </Link>
            <nav className="hidden md:flex items-center space-x-8">{navLinks}</nav>
            <div className="md:hidden">
              <button onClick={() => setIsMenuOpen(!isMenuOpen)}>
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16m-7 6h7" />
                </svg>
              </button>
            </div>
          </div>

          {isMenuOpen && (
            <motion.div className="md:hidden" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <nav className="flex flex-col items-center space-y-4 p-4">{navLinks}</nav>
            </motion.div>
          )}
        </header>

        {/* --- Main Routes --- */}
        <main>
          <Routes>
            <Route path="/" element={<><HeroBanner /><Why /></>} />
            <Route path="/fleet-overview" element={<FleetOverview isLoggedIn={isLoggedIn} />} />
            <Route path="/machine/battery/:id" element={<BatteryDashboard isLoggedIn={isLoggedIn}/>} />
            <Route path="/machine/motor/:id" element={<MotorDashboard isLoggedIn={isLoggedIn}/>} />
            <Route path="/machine/pump/:id" element={<PumpDashboard isLoggedIn={isLoggedIn}/>} />
            <Route path="/machine/hydraulic/:id" element={<HydraulicDashboard isLoggedIn={isLoggedIn}/>} />
            <Route path="/admin/tickets" element={<AdminTicketView />} />
            <Route path="/model-ops" element={<ModelOpsCenter />} />
            <Route path="/clustering-insights/battery" element={<ClusteringInsights />} />
            <Route path="/clustering-insights/motor" element={<MotorClusteringPage />} />
            <Route path="/kalman-filter" element={<KalmanFilterPage />} />
            <Route path="/patterns" element={<PatternMinerPage />} />
            <Route path="/drift-hub" element={<DriftHubPage />} />
            <Route path="/account" element={<AccountPage />} />

            {/* <Route path="/machine/battery/:id" element={<BatteryDashboard isLoggedIn={isLoggedIn} />}/>
            <Route path="/machine/motor/:id" element={<MotorDashboard isLoggedIn={isLoggedIn} />} />
            <Route path="/machine/pump/:id" element={<PumpDashboard isLoggedIn={isLoggedIn} />} />
            <Route path="/machine/hydraulic/:id" element={<HydraulicDashboard isLoggedIn={isLoggedIn} />} /> */}
          </Routes>

          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            <Link
              to="/fleet-overview"
              className="inline-block bg-yellow-400 text-slate-900 font-bold uppercase px-8 py-3 mt-10 mb-10 text-lg rounded-full transition-all hover:bg-yellow-500 hover:shadow-lg"
            >
              Explore Assets
            </Link>
          </motion.div>
        </main>

        <Footer />
      </div>
    </Router>
  );
}

export default App;