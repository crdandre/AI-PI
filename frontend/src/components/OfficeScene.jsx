import React, { useState, useEffect } from 'react';
import { Box, Image, VStack, Text, useToast, HStack } from '@chakra-ui/react';
import { motion } from 'framer-motion';

const MotionBox = motion(Box);

const useTypewriter = (text, speed = 50) => {
  const [displayedText, setDisplayedText] = useState('');
  const [isTyping, setIsTyping] = useState(true);

  useEffect(() => {
    setIsTyping(true);
    setDisplayedText('');
    
    let i = 0;
    const timer = setInterval(() => {
      if (i < text.length) {
        setDisplayedText(prev => prev + text.charAt(i));
        i++;
      } else {
        setIsTyping(false);
        clearInterval(timer);
      }
    }, speed);

    return () => clearInterval(timer);
  }, [text, speed]);

  return { displayedText, isTyping };
};

export const OfficeScene = ({ onDocumentProcessed, isTestMode }) => {
  const [documentState, setDocumentState] = useState('idle');
  const [statusMessage, setStatusMessage] = useState('    Upload a document to get started...');
  const [fileData, setFileData] = useState(null);
  const { displayedText, isTyping } = useTypewriter(statusMessage, 40);
  const toast = useToast();
  const [reviewProgress, setReviewProgress] = useState(0);

  // Calculate typing duration based on message length
  const getTypingDuration = (message) => {
    return message.length * 50 + 500; // 50ms per character + 500ms buffer
  };

  const handleProfessorClick = async () => {
    if (documentState === 'onDesk' && fileData) {
      try {
        setDocumentState('reviewing');
        setStatusMessage('    Professor is reviewing your document...');
        
        // Reset and start progress animation
        setReviewProgress(0);
        const startTime = Date.now();
        const duration = 120000; // 120 seconds = 2 minutes

        const progressInterval = setInterval(() => {
          const elapsed = Date.now() - startTime;
          const progress = Math.min(elapsed / duration, 1);
          setReviewProgress(progress);
        }, 50); // Update every 50ms for smooth animation

        const endpoint = isTestMode ? 'processing_test' : 'process';
        const response = await fetch(`${import.meta.env.VITE_API_URL}/api/${endpoint}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ fileId: fileData.fileId }),
        });
        
        clearInterval(progressInterval);
        
        if (!response.ok) throw new Error('Processing failed');
        
        const processedData = await response.json();
        setDocumentState('complete');
        setStatusMessage('    Review complete! Click to download.');
        setFileData({ ...fileData, ...processedData });
      } catch (error) {
        setReviewProgress(0);
        toast({
          title: 'Processing failed',
          description: error.message,
          status: 'error',
          duration: 5000,
        });
        setDocumentState('onDesk');
        setStatusMessage('    Error occurred. Please try again.');
      }
    }
  };

  const handleFileUpload = async (e) => {
    if (e.target.files?.[0]) {
      try {
        const formData = new FormData();
        formData.append('file', e.target.files[0]);
        
        const response = await fetch(`${import.meta.env.VITE_API_URL}/api/upload`, {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) throw new Error('Upload failed');
        
        const data = await response.json();
        setFileData(data);
        setDocumentState('onDesk');
        setStatusMessage('    Document placed on desk. Click the professor to start review.');
      } catch (error) {
        toast({
          title: 'Upload failed',
          description: error.message,
          status: 'error',
          duration: 5000,
        });
      }
    }
  };

  return (
    <Box 
      position="relative" 
      h="600px" 
      w="100%" 
      bg="inherit"
      display="block"
      margin="0 auto"
    >
      {/* Professor - Top of screen */}
      <MotionBox
        position="absolute"
        top="100px"
        left="50%"
        style={{ x: '-50%' }}
        whileHover={{ scale: 1.02 }}
        transition={{ duration: 0.2 }}
        cursor={documentState === 'onDesk' ? 'pointer' : 'default'}
        onClick={handleProfessorClick}
        zIndex={1}
        _before={{
          content: '""',
          position: 'absolute',
          top: '-20px',
          left: '-20px',
          right: '-20px',
          bottom: '-20px',
          borderRadius: 'full',
          background: documentState === 'reviewing' 
            ? `rgba(${Math.round(0 + (75 * reviewProgress))}, ${Math.round(149 - (149 * reviewProgress))}, ${Math.round(255 - (125 * reviewProgress))}, ${reviewProgress * 3})`
            : 'transparent',
          filter: 'blur(20px)',
          transition: 'background 0.3s ease',
          zIndex: -1,
        }}
      >
        <Image
          src={`/professor-avatar${documentState === 'complete' ? '-complete' : ''}.png`}
          alt="Professor"
          w="140px"
          h="180px"
          objectFit="contain"
          filter="drop-shadow(0px 4px 4px rgba(0, 0, 0, 0.25))"
        />
      </MotionBox>

      {/* Desk in middle */}
      <Box
        position="absolute"
        top="50%"
        left="50%"
        transform="translate(-50%, -50%)"
        zIndex={2}
      >
        {/* Desk top */}
        <Box
          w="300px"
          h="80px"
          bgGradient="linear(to-b, #2D1810, #1F110B)"
          borderRadius="md"
          boxShadow="0 8px 30px rgba(0, 0, 0, 0.7)"
          cursor={documentState === 'idle' ? 'pointer' : 'default'}
          transform="perspective(500px) rotateX(30deg)"
          onClick={() => {
            if (documentState === 'idle') {
              document.getElementById('fileInput').click();
            }
          }}
          position="relative"
        >
          {/* Upload indicator (left side, only shown in idle state) */}
          {documentState === 'idle' && (
            <VStack
              position="absolute"
              top="50%"
              left="25%"
              transform="translate(-50%, -50%) rotateX(-30deg)"
              spacing={1}
              opacity={0.7}
            >
              <Box
                w="32px"
                h="40px"
                bg="whiteAlpha.900"
                borderRadius="sm"
                position="relative"
                display="flex"
                justifyContent="center"
                alignItems="center"
              >
                <Box
                  as="span"
                  fontSize="24px"
                  color="gray.500"
                  position="absolute"
                >
                  +
                </Box>
              </Box>
              <Text
                fontSize="xs"
                color="whiteAlpha.900"
                fontWeight="medium"
              >
                Upload Document
              </Text>
            </VStack>
          )}

          {/* Document on desk - left side */}
          {documentState !== 'idle' && (
            <MotionBox
              position="absolute"
              top="-30px"
              left="25%"
              transform="translateX(-50%)"
              w="40px"
              h="56px"
              bg="white"
              borderRadius="sm"
              boxShadow="md"
              zIndex={3}
              animate={{
                y: documentState === 'reviewing' ? [-150, 0] : 0,
                scale: documentState === 'reviewing' ? [1, 0.8] : 1,
                rotate: documentState === 'reviewing' ? [0, 360] : 0
              }}
              transition={{
                duration: 1,
                repeat: documentState === 'reviewing' ? Infinity : 0,
                repeatType: "reverse"
              }}
            >
              {/* Document icon/lines */}
              <Box
                p={1}
                display="flex"
                flexDirection="column"
                gap={1}
              >
                <Box h="2px" w="80%" bg="gray.300" />
                <Box h="2px" w="60%" bg="gray.300" />
                <Box h="2px" w="70%" bg="gray.300" />
              </Box>
            </MotionBox>
          )}

          {/* Reviewed Document on desk - middle position */}
          {documentState === 'complete' && (
            <MotionBox
              position="absolute"
              top="-30px"
              left="50%"
              transform="translateX(-50%)"
              w="40px"
              h="56px"
              bg="white"
              borderRadius="sm"
              boxShadow="md"
              zIndex={3}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
            >
              {/* Document icon/lines with checkmark */}
              <Box
                p={1}
                display="flex"
                flexDirection="column"
                gap={1}
                position="relative"
              >
                <Box h="2px" w="80%" bg="gray.300" />
                <Box h="2px" w="60%" bg="gray.300" />
                <Box h="2px" w="70%" bg="gray.300" />
                <Box
                  position="absolute"
                  top="50%"
                  left="50%"
                  transform="translate(-50%, -50%)"
                  color="green.500"
                  fontSize="24px"
                >
                  ✓
                </Box>
              </Box>
            </MotionBox>
          )}

          {/* Download indicator - right side */}
          {documentState === 'complete' && (
            <VStack
              position="absolute"
              top="50%"
              left="75%"
              transform="translate(-50%, -50%) rotateX(-30deg)"
              spacing={1}
              opacity={0.7}
              cursor="pointer"
              onClick={async (e) => {
                e.stopPropagation();
                try {
                  const downloadUrl = `${import.meta.env.VITE_API_URL}/api/documents/${fileData.fileId}`;
                  const response = await fetch(downloadUrl, {
                    method: 'GET',
                    headers: {
                      'Accept': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                    },
                  });
                  
                  if (!response.ok) {
                    throw new Error(`Download failed: ${response.status} ${response.statusText}`);
                  }
                  
                  const blob = await response.blob();
                  const url = window.URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  const originalFilename = fileData.filename || 'document.docx';
                  const downloadFilename = `reviewed_${fileData.fileId}_${originalFilename}`;
                  
                  a.download = downloadFilename;
                  a.click();
                  window.URL.revokeObjectURL(url);
                } catch (error) {
                  toast({
                    title: 'Download failed',
                    description: error.message,
                    status: 'error',
                    duration: 5000,
                  });
                }
              }}
            >
              <Box
                w="32px"
                h="40px"
                bg="whiteAlpha.900"
                borderRadius="sm"
                position="relative"
                display="flex"
                justifyContent="center"
                alignItems="center"
              >
                <Box
                  as="span"
                  fontSize="24px"
                  color="green.500"
                  position="absolute"
                >
                  ↓
                </Box>
              </Box>
              <Text
                fontSize="xs"
                color="whiteAlpha.900"
                fontWeight="medium"
              >
                Download Review
              </Text>
            </VStack>
          )}
        </Box>
      </Box>

      {/* Student - Bottom of screen */}
      <MotionBox
        position="absolute"
        bottom="120px"
        left="50%"
        style={{ x: '-50%' }}
        whileHover={{ scale: 1.02 }}
        transition={{ duration: 0.2 }}
        zIndex={2}
      >
        <Image
          src="/student-avatar.png"
          alt="Student"
          w="140px"
          h="180px"
          objectFit="contain"
          filter="drop-shadow(0px 4px 4px rgba(0, 0, 0, 0.25))"
        />
      </MotionBox>

      {/* Game Boy style dialogue box with typing animation */}
      <MotionBox
        position="absolute"
        bottom="20px"
        left="50%"
        style={{ x: '-50%' }}
        bg="white"
        border="4px solid"
        borderColor="gray.200"
        color="gray.700"
        w="400px"
        p={4}
        borderRadius="lg"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        boxShadow="0 4px 6px rgba(0, 0, 0, 0.1)"
        zIndex={1}
      >
        <HStack spacing={3} align="flex-start">
          <Box
            w="8px"
            h="8px"
            borderRadius="full"
            bg="green.500"
            mt={1}
          />
          <Text 
            fontSize="xs"
            fontWeight="medium"
            color="gray.600"
            whiteSpace="pre-wrap"
          >
            {displayedText || statusMessage}
            {isTyping && (
              <Box as="span" animation="blink 1s step-end infinite">
                ▊
              </Box>
            )}
          </Text>
        </HStack>
      </MotionBox>

      {/* Add this style for the blinking cursor */}
      <style jsx global>{`
        @keyframes blink {
          0% { opacity: 1; }
          50% { opacity: 0; }
          100% { opacity: 1; }
        }
      `}</style>

      {/* Hidden file input */}
      <input
        id="fileInput"
        type="file"
        accept=".pdf,.doc,.docx"
        style={{ display: 'none' }}
        onChange={handleFileUpload}
      />
    </Box>
  );
}; 