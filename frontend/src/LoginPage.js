// // LoginPage.js
// import React from "react";
// import { useNavigate } from "react-router-dom";

// function LoginPage() {
//   const navigate = useNavigate();

//   const handleLogin = () => {
//     // your login validation logic here
//     localStorage.setItem("isLoggedIn", "true");
//     navigate("/"); // go back to main/home page after login
//   };

//   return (
//     <div className="flex flex-col items-center justify-center h-screen">
//       <h1 className="text-3xl font-bold mb-4">Login</h1>
//       <button
//         onClick={handleLogin}
//         className="px-6 py-2 bg-sky-600 text-white rounded-lg hover:bg-sky-500"
//       >
//         Login
//       </button>
//     </div>
//   );
// }

// export default LoginPage;


// src/LoginPage.js
// import React, { useState } from "react";

// const LoginPage = ({ onLogin }) => {
//   const [email, setEmail] = useState("");
//   const [password, setPassword] = useState("");
//   const [error, setError] = useState("");

//   const handleSubmit = (e) => {
//     e.preventDefault();

//     // Example: Replace this with real authentication logic
//     if (email === "admin@example.com" && password === "123456") {
//       onLogin();
//     } else {
//       setError("Invalid credentials. Try again!");
//     }
//   };

//   return (
//     <div className="flex items-center justify-center min-h-screen bg-gray-50">
//       <div className="bg-white shadow-xl rounded-2xl p-10 w-[90%] sm:w-[400px]">
//         <h2 className="text-3xl font-bold text-center text-gray-800 mb-6">
//           Welcome Back 
//         </h2>
//         {error && (
//           <div className="bg-red-100 text-red-700 p-2 rounded mb-3 text-center">
//             {error}
//           </div>
//         )}
//         <form onSubmit={handleSubmit} className="flex flex-col">
//           <label className="text-gray-600 mb-1 font-medium">Email</label>
//           <input
//             type="email"
//             placeholder="Enter your email"
//             value={email}
//             onChange={(e) => setEmail(e.target.value)}
//             className="p-3 border rounded-lg mb-4 focus:outline-none focus:ring-2 focus:ring-indigo-400"
//             required
//           />

//           <label className="text-gray-600 mb-1 font-medium">Password</label>
//           <input
//             type="password"
//             placeholder="Enter your password"
//             value={password}
//             onChange={(e) => setPassword(e.target.value)}
//             className="p-3 border rounded-lg mb-6 focus:outline-none focus:ring-2 focus:ring-indigo-400"
//             required
//           />

//           <button
//             type="submit"
//             className="bg-indigo-600 text-white py-3 rounded-lg font-semibold hover:bg-indigo-700 transition-all duration-300"
//           >
//             Login
//           </button>
//         </form>

//         <p className="text-center text-gray-500 text-sm mt-5">
//           Don’t have an account?{" "}
//           <span className="text-indigo-600 hover:underline cursor-pointer">
//             Sign up
//           </span>
//         </p>
//       </div>
//     </div>
//   );
// };

// export default LoginPage;


import React, { useState } from "react";

const LoginPage = ({ onLogin }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleLogin = (e) => {
    e.preventDefault();

    // You can add validation or API login here
    if (email && password) {
      onLogin(); // tells App.js "user logged in"
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <div className="bg-white p-8 rounded-lg shadow-lg w-96">
        <h1 className="text-2xl font-bold text-center mb-6 text-gray-800">
          Login to SmartPredict
        </h1>
        <form onSubmit={handleLogin}>
          <input
            type="email"
            placeholder="Email"
            className="w-full p-3 border rounded mb-4"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <input
            type="password"
            placeholder="Password"
            className="w-full p-3 border rounded mb-4"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <button
            type="submit"
            className="w-full bg-yellow-400 text-gray-900 font-bold py-2 rounded hover:bg-yellow-500 transition-colors"
          >
            Login
          </button>
        </form>
      </div>
    </div>
  );
};

export default LoginPage;