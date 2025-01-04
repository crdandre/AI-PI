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
  Checkbox,
  Heading,
} from '@chakra-ui/react';
import axios from 'axios';
import { CheckIcon } from '@chakra-ui/icons';

export const DocumentUpload = ({ onDocumentProcessed, onApiStatusChange, onApiUrlChange }) => {
  const [file, setFile] = useState(null);
  const [model, setModel] = useState('gpt-4o-mini');
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  const [uploadedFileId, setUploadedFileId] = useState(null);
  const [processingStatus, setProcessingStatus] = useState('');
  const [apiConnected, setApiConnected] = useState(false);
  const [testMode, setTestMode] = useState(false);
  const toast = useToast();
  const [apiHost, setApiHost] = useState('localhost');
  const [apiPort, setApiPort] = useState(8001);
  const apiUrl = `http://${apiHost}:${apiPort}`;

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
          onApiStatusChange(true);
          onApiUrlChange(apiUrl);
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
        onApiStatusChange(false);
        onApiUrlChange(apiUrl);
        
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
  }, [toast, apiUrl, onApiStatusChange, onApiUrlChange]);

  const handleUpload = async (file) => {
    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      console.log('Uploading file...', file.name);
      const uploadResponse = await fetch(`${apiUrl}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) {
        const errorData = await uploadResponse.json();
        throw new Error(`Upload failed: ${errorData.detail}`);
      }

      const uploadData = await uploadResponse.json();
      console.log('Upload response:', uploadData);

      if (!uploadData.fileId) {
        throw new Error('No fileId received from server');
      }

      setUploadedFileId(uploadData.fileId);
      toast({
        title: 'Upload Successful',
        description: `${file.name} has been uploaded successfully`,
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
    } catch (error) {
      console.error('Upload error:', error);
      handleError(error, 'Failed to upload document');
    } finally {
      setIsUploading(false);
    }
  };

  const clearUpload = () => {
    setFile(null);
    setUploadedFileId(null);
    setIsComplete(false);
  };

  const handleProcess = async () => {
    if (!uploadedFileId) return;

    setIsProcessing(true);
    setProcessingStatus('Processing document...');

    try {
      const endpoint = testMode ? 'processing_test' : 'process';
      const response = await axios.post(
        `${apiUrl}/api/${endpoint}`,
        {
          fileId: uploadedFileId,
          model: model
        }
      );

      setIsComplete(true);
      onDocumentProcessed({
        ...response.data,
        fileId: uploadedFileId
      });
    } catch (error) {
      handleError(error, 'Failed to process document');
    } finally {
      setProcessingStatus('');
      setIsProcessing(false);
    }
  };

  const handleError = async (error, defaultMessage) => {
    console.error('Operation error:', error);
    
    let errorMessage = defaultMessage;
    
    if (error.response) {
      if (error.response.data instanceof Blob) {
        const text = await error.response.data.text();
        try {
          const errorData = JSON.parse(text);
          errorMessage = errorData.detail || errorData.message || text;
        } catch {
          errorMessage = text;
        }
      } else {
        errorMessage = error.response.data?.detail || 
                      error.response.data?.message || 
                      `Server error: ${error.response.status}`;
      }
    } else if (error.request) {
      errorMessage = 'No response received from server';
    } else {
      errorMessage = error.message;
    }

    toast({
      title: 'Error',
      description: errorMessage,
      status: 'error',
      duration: 5000,
      isClosable: true,
    });
  };

  return (
    <Container maxW="container.md" py={8}>
      <VStack spacing={6} align="stretch">
        <Box 
          p={8} 
          bg="white" 
          boxShadow="xl" 
          borderRadius="md" 
          borderWidth="1px" 
          borderColor="cambridge.blue"
        >
          <Heading 
            as="h3" 
            fontSize="xl" 
            fontFamily="Crimson Text, Georgia, serif" 
            color="cambridge.darkBlue" 
            textAlign="center" 
            mb={6}
          >
            Document Submission Portal
          </Heading>
        </Box>

        {!apiConnected && (
          <Box p={4} bg="red.100" color="red.700" borderRadius="md">
            API server is not connected. Please ensure it is running on port 8000.
          </Box>
        )}
        
        <Box borderWidth={2} borderRadius="lg" p={4} borderStyle="dashed">
          <input
            type="file"
            accept=".docx"
            onChange={(e) => {
              const selectedFile = e.target.files?.[0];
              if (selectedFile) {
                setFile(selectedFile);
                handleUpload(selectedFile);
              }
            }}
            style={{ width: '100%' }}
          />
          {file && (
            <Text mt={2} color="green.500">
              Selected file: {file.name}
            </Text>
          )}
        </Box>

        <Select value={model} onChange={(e) => setModel(e.target.value)}>
          <option value="gpt-4o-mini">4o Mini</option>
          <option value="o1-mini">o1 Mini</option>
        </Select>

        <Checkbox 
          isChecked={testMode} 
          onChange={(e) => setTestMode(e.target.checked)}
        >
          Test Mode (Skip Processing)
        </Checkbox>

        <Button
          colorScheme={file ? "red" : "blue"}
          onClick={() => file ? clearUpload() : null}
          isLoading={isUploading}
          disabled={!apiConnected}
          size="lg"
        >
          {!apiConnected ? 'API Not Connected' : (file ? 'Clear Upload' : 'Upload Document')}
        </Button>

        <Button
          colorScheme="green"
          onClick={handleProcess}
          isLoading={isProcessing}
          disabled={!uploadedFileId || isUploading}
          size="lg"
        >
          Process Document
        </Button>

        {(isUploading || isProcessing) && (
          <Box>
            {isUploading && (
              <Progress 
                size="xs" 
                value={uploadProgress} 
                colorScheme="blue"
                hasStripe
                isAnimated
              />
            )}
            <Text mt={2} textAlign="center">
              {processingStatus}
              {processingStatus === 'Uploading file...' && ` ${Math.round(uploadProgress)}%`}
            </Text>
          </Box>
        )}

        {isComplete && !isUploading && !isProcessing && (
          <Box textAlign="center" color="green.500">
            <CheckIcon w={6} h={6} />
            <Text>Processing complete! Download should start automatically.</Text>
          </Box>
        )}
      </VStack>
    </Container>
  );
  
};