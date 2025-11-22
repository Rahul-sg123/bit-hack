// src/ReportGenerator.js
import React from 'react';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

const ReportGenerator = ({ machineId, reportContentRef }) => {
  const handleDownloadPdf = async () => {
    const input = reportContentRef.current;
    if (!input) {
      console.error("Report content ref is not available.");
      return;
    }

    // Use html2canvas to capture the div as an image
    const canvas = await html2canvas(input, { scale: 2 });
    const imgData = canvas.toDataURL('image/png');

    // Create a new PDF document
    // 'p' for portrait, 'mm' for millimeters, 'a4' for A4 size
    const pdf = new jsPDF('p', 'mm', 'a4');
    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
    const generationDate = new Date().toLocaleDateString('en-IN', { timeZone: 'Asia/Kolkata' });

    // Add header
    pdf.setFontSize(20);
    pdf.text(`SmartPredict Health Report: ${machineId}`, 15, 20);

    // Add the captured image
    pdf.addImage(imgData, 'PNG', 10, 30, pdfWidth - 20, pdfHeight - 20);

    // Add footer
    pdf.setFontSize(10);
    pdf.text(`Generated on: ${generationDate}`, 15, pdf.internal.pageSize.getHeight() - 10);

    // Download the PDF
    pdf.save(`Report-${machineId}.pdf`);
  };

  return (
    <button
      onClick={handleDownloadPdf}
      className="bg-green-600 text-white font-bold px-4 py-2 rounded-md hover:bg-green-500 transition-colors"
    >
      📄 Download Report
    </button>
  );
};

export default ReportGenerator;