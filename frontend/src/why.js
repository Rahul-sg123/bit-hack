import React from "react";

import { motion } from "framer-motion";

import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";

import loginImg from "./loginimg.jpg";

import Predict from "./predict.jpg";

import Prevent from "./prevent.jpg";

import Perform from "./perform.jpg";



const why = () => {

  return (

    <>

      <div className="flex flex-col mb-10 mt-10">

        <span className="text-4xl text-black font-bold mt-10">

          Why SmartPredict AI?

        </span>

        <div className="flex flex-row items-center">

          <motion.div

            className="flex flex-col items-center p-10"

            initial={{ opacity: 0, x: -100 }}

          whileInView={{ opacity: 1, x: 0 }}

          transition={{ duration: 0.8 }}

          viewport={{ once: true, amount: 0.3 }}

          >

            <img

              src={Predict}

              alt="predict"

              className="max-h-[250px] max-w-[250px] m-10"

            />

            <span className="text-2xl text-black">Predict</span>

            <p className="text-l">

              Foreest equipment failure and supply chain botimatite with

              industry leading, Stop prediction, Stop reacting

            </p>

          </motion.div>

          <motion.div

            className="flex flex-col items-center p-10"

            initial={{ opacity: 0, y: 100 }}

          whileInView={{ opacity: 1, y: 0 }}

          transition={{ duration: 0.8 }}

          viewport={{ once: true, amount: 0.3 }}

          >

            <img

              src={Prevent}

              alt="predict"

              className="max-h-[250px] max-w-[250px] m-10"

            />

            <span className="text-2xl text-black">Prevent</span>

            <p className="text-l">

              Foreest equipment failure and supply chain botimatite with

              industry leading, Stop prediction, Stop reacting

            </p>

          </motion.div>

          <motion.div

            className="flex flex-col items-center p-10"

            initial={{ opacity: 0, x: 100 }}

          whileInView={{ opacity: 1, x: 0 }}

          transition={{ duration: 0.8 }}

          viewport={{ once: true, amount: 0.3 }}

          >

            <img

              src={Perform}

              alt="predict"

              className="max-h-[250px] max-w-[250px] m-10"

            />

            <span className="text-2xl text-black">Perform</span>

            <p className="text-l">

              Foreest equipment failure and supply chain botimatite with

              industry leading, Stop prediction, Stop reacting

            </p>

          </motion.div>

        </div>

      </div>

    </>

  );

};

export default why;