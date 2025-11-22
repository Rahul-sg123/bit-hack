// src/MachineModel3D.js
import React from 'react';
import { Canvas } from '@react-three/fiber';
import { useGLTF, OrbitControls, Stage } from '@react-three/drei';

function Model({ path }) {
  const { scene } = useGLTF(path);
  return <primitive object={scene} />;
}

export default function MachineModel3D({ modelPath }) {
  return (
    <div className="h-80 bg-gray-900/50 backdrop-blur-sm border border-gray-700 rounded-lg shadow-lg p-2">
      <Canvas camera={{ fov: 45 }}>
        <Stage environment="city" intensity={0.6}>
          <Model path={modelPath} />
        </Stage>
        <OrbitControls autoRotate enableZoom={false} />
      </Canvas>
    </div>
  );
}