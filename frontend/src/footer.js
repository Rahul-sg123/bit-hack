import React from "react";
import linkdin from "./linkdin.jpg";
import twitter from "./twitter.jpg";
import github from "./github.jpg";
import medium from "./medium.jpg";
import reddit from "./reddit.jpg";
import instagram from "./insta.jpg";

const Footer = () => {
  const social = [
    { img: linkdin, alt: "LinkedIn", href: "#" },
    { img: twitter, alt: "Twitter", href: "#" },
    { img: github, alt: "GitHub", href: "#" },
    { img: medium, alt: "Medium", href: "#" },
    { img: instagram, alt: "instagram", href: "#" },
  ];

  return (
    <footer className="flex justify-center items-center bg-white text-l mt-5 min-h-[250px] text-left w-full">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 p-10 w-full max-w-6xl">
        {/* Column 1 */}
        <div>
          <span className="text-2xl font-bold mb-12">SmartPredict</span>
          <p className="mt-4">
            JSS Boys Hostel, Vishnuvardhana Road, Uttarahalli, Kengeri,
            Bengaluru - 560060
          </p>
          <p className="mb-10">Karnataka</p>
          <div>
            <span className="font-bold">Contact Us</span>
            <div className="w-[200px] h-[1px] bg-black mt-4 mb-4"></div>
            <div className="flex flex-row space-x-4 mt-2 mb-4">
              {social.map((item, index) => (
                <a href={item.href} key={index}>
                  <img
                    src={item.img}
                    alt={item.alt}
                    className="w-10 h-10 object-contain hover:opacity-75 transition-opacity "
                  />
                </a>
              ))}
            </div>
          </div>
          <span>Ⓒ 2025 SmartPredict. All rights reserved.</span>
        </div>

        {/* Column 2 */}
        <div>
          <span className="font-bold">SMARTPREDICT</span>
          <div className="flex flex-col space-y-2 mt-2">
            <a href="#" className="text-grey-400 hover:text-black">About Us</a>
            <a href="#" className="text-grey-400 hover:text-black">Pricing</a>
            <a href="#" className="text-grey-400 hover:text-black">Help & Support</a>
          </div>
        </div>

        {/* Column 3 */}
        <div>
          <span className="font-bold">SERVICES</span>
          <div className="flex flex-col space-y-2 mt-2">
            <a href="#" className="text-grey-100 hover:text-black">EV Battery</a>
            <a href="#" className="text-grey-100 hover:text-black">Electric Motor</a>
            <a href="#" className="text-grey-100 hover:text-black">Compressor Pump</a>
            <a href="#" className="text-grey-100 hover:text-black">Server UPS</a>
            <a href="#" className="text-grey-100 hover:text-black">Conveyor</a>
            <a href="#" className="text-grey-100 hover:text-black">Hydraulic System</a>
          </div>
          <span className="block mt-4">Version 0.0.2</span>
        </div>

        {/* Bottom Line */}
        <div className="col-span-full text-center mt-4">
          @SmartPredict | 2025 | <span className="text-yellow-500">DriveX</span>
        </div>
      </div>
    </footer>
  );
};

export default Footer;