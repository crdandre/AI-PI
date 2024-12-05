import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Progress,
  Select,
  VStack,
  Text,
  Container,
  useToast,
} from '@chakra-ui/react';
import axios from 'axios';
import { CheckIcon } from '@chakra-ui/icons';

const apiHost = 'localhost';
const apiPort = 8001;
const apiUrl = `http://${apiHost}:${apiPort}`;

export const DocumentUpload = () => {
  const [file, setFile] = useState(null);
  const [model, setModel] = useState('gpt-4o-mini');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  const [processingStatus, setProcessingStatus] = useState('');
  const [apiConnected, setApiConnected] = useState(false);
  const toast = useToast();

  // Check API connection on component mount
  useEffect(() => {
    const checkApiConnection = async () => {
      console.log(`Attempting to connect to API at ${apiUrl}/`);
      try {
        // Try without withCredentials first
        const response = await axios.get(`${apiUrl}/`, {
          headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
          },
          timeout: 5000,
          // Remove withCredentials for initial testing
          // withCredentials: true
        });
        
        console.log('API Response:', response);
        
        if (response.data && (response.status === 200 || response.status === 201)) {
          setApiConnected(true);
          toast.closeAll();
        } else {
          throw new Error('API returned unexpected response');
        }
      } catch (error) {
        // Add more specific error handling
        let errorMessage = 'Cannot connect to API server. ';
        
        if (error.code === 'ECONNABORTED') {
          errorMessage += 'Connection timed out. Is the server running?';
        } else if (error.code === 'ERR_NETWORK') {
          errorMessage += 'Network error. Check if the server is running on the correct port.';
        } else {
          errorMessage += error.message;
        }
        
        console.error('API connection failed:', {
          message: error.message,
          code: error.code,
          response: error.response,
          url: `${apiUrl}/`
        });
        
        setApiConnected(false);
        
        if (!toast.isActive('api-error')) {
          toast({
            id: 'api-error',
            title: 'API Connection Failed',
            description: errorMessage,
            status: 'error',
            duration: null,
            isClosable: true,
          });
        }
      }
    };

    checkApiConnection();
    const interval = setInterval(checkApiConnection, 60000);
    return () => clearInterval(interval);
  }, [toast]);

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);

    setIsLoading(true);
    setUploadProgress(0);
    setIsComplete(false);
    setProcessingStatus('Uploading file...');
    
    try {
      const response = await axios.post(`${apiUrl}/api/review`, formData, {
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const progress = (progressEvent.loaded / progressEvent.total) * 100;
            setUploadProgress(progress);
            if (progress === 100) {
              setProcessingStatus('Processing document...');
            }
          } else {
            setProcessingStatus('Uploading file... (size unknown)');
          }
        },
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // If we get here, the request was successful
      const blob = new Blob([response.data]);
      
      // Create download link for processed document
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `reviewed_${file.name}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);

      setIsComplete(true);
      toast({
        title: 'Success',
        description: 'Document processed successfully',
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
    } catch (error) {
      console.error('Upload error:', error);
      
      // Improved error handling
      let errorMessage = 'Failed to process document';
      
      if (error.response) {
        // The server responded with an error
        if (error.response.data instanceof Blob) {
          // If the error response is a blob, read it
          const text = await error.response.data.text();
          try {
            const errorData = JSON.parse(text);
            errorMessage = errorData.detail || errorData.message || text;
          } catch {
            errorMessage = text;
          }
        } else {
          // Regular JSON error response
          errorMessage = error.response.data?.detail || 
                        error.response.data?.message || 
                        `Server error: ${error.response.status}`;
        }
      } else if (error.request) {
        // The request was made but no response was received
        errorMessage = 'No response received from server';
      } else {
        // Something happened in setting up the request
        errorMessage = error.message;
      }

      toast({
        title: 'Error',
        description: errorMessage,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setProcessingStatus('');
      setIsLoading(false);
    }
  };

  return (
    <Container maxW="container.md" py={8}>
      <VStack spacing={6} align="stretch">
        <Text fontSize="2xl" fontWeight="bold" textAlign="center">
          AI-PI Document Review
        </Text>

        {!apiConnected && (
          <Box p={4} bg="red.100" color="red.700" borderRadius="md">
            ⚠️ API server is not connected. Please ensure it is running on port 8000.
          </Box>
        )}
        
        <Box borderWidth={2} borderRadius="lg" p={4} borderStyle="dashed">
          <input
            type="file"
            accept=".docx"
            onChange={(e) => setFile(e.target.files?.[0])}
            style={{ width: '100%' }}
          />
        </Box>

        <Select value={model} onChange={(e) => setModel(e.target.value)}>
          <option value="gpt-4o-mini">4o Mini</option>
          <option value="o1-mini">o1 Mini</option>
        </Select>

        <Button
          colorScheme="blue"
          onClick={handleUpload}
          isLoading={isLoading}
          disabled={!file || !apiConnected}
          size="lg"
        >
          {!apiConnected ? 'API Not Connected' : 'Process Document'}
        </Button>

        {isLoading && (
          <Box>
            <Progress 
              size="xs" 
              value={uploadProgress} 
              colorScheme="blue"
              hasStripe
              isAnimated
            />
            <Text mt={2} textAlign="center">
              {processingStatus}
              {processingStatus === 'Uploading file...' && ` ${Math.round(uploadProgress)}%`}
            </Text>
          </Box>
        )}

        {isComplete && !isLoading && (
          <Box textAlign="center" color="green.500">
            <CheckIcon w={6} h={6} />
            <Text>Processing complete! Download should start automatically.</Text>
          </Box>
        )}
      </VStack>
    </Container>
  );
};