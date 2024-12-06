import React from 'react';
import { Button } from '@chakra-ui/react';
import { MdDownload } from 'react-icons/md';

export const DocumentDownloader = ({ reviewData, fileId }) => {
  const handleDownload = async () => {
    try {
      const downloadUrl = `${import.meta.env.VITE_API_URL}/api/documents/${fileId}`;
      console.log('Attempting download from:', downloadUrl);
      console.log('FileId:', fileId);
      console.log('Full request details:', {
        url: downloadUrl,
        method: 'GET',
        headers: {
          'Accept': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }
      });
      
      const response = await fetch(downloadUrl, {
        method: 'GET',
        headers: {
          'Accept': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        },
      });
      
      console.log('Response status:', response.status);
      console.log('Response headers:', Object.fromEntries(response.headers.entries()));
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Server response:', {
          status: response.status,
          statusText: response.statusText,
          error: errorText
        });
        throw new Error(`Failed to download document: ${response.status} ${response.statusText}`);
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const originalFilename = reviewData?.filename || 'document.docx';
      const downloadFilename = `reviewed_${fileId}_${originalFilename}`;
      
      a.download = downloadFilename;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download failed:', error);
      alert(`Download failed: ${error.message}`);
    }
  };

  return (
    <Button
      leftIcon={<MdDownload />}
      colorScheme="blue"
      onClick={handleDownload}
    >
      Download Reviewed Document
    </Button>
  );
};
