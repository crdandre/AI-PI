import React from 'react';
import {
  Box,
  VStack,
  Container,
  Heading,
  List,
  ListItem,
  ListIcon,
  Button,
} from '@chakra-ui/react';
import { MdCheckCircle, MdWarning, MdInfo, MdDownload } from 'react-icons/md';
import axios from 'axios';

export const DocumentViewer = ({ reviewData, fileId }) => {
  const handleDownload = async () => {
    try {
      console.log('Starting download with fileId:', fileId);
      console.log('API URL:', import.meta.env.VITE_API_URL);
      
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/documents/${fileId}`, {
        method: 'GET',
        headers: {
          'Accept': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        },
      });
      
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
      console.log('Blob received:', {
        size: blob.size,
        type: blob.type
      });
      
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const originalFilename = reviewData?.filename || 'document.docx';
      const downloadFilename = `reviewed_${fileId}_${originalFilename}`;
      console.log('Download filename:', downloadFilename);
      
      a.download = downloadFilename;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download failed:', error);
      // You could also add a toast notification here
      alert(`Download failed: ${error.message}`);
    }
  };

  const renderHighLevelReview = () => {
    if (!reviewData?.high_level_review) return null;
    const { 
      overall_assessment, 
      key_strengths, 
      key_weaknesses, 
      recommendations,
      communication_review 
    } = reviewData.high_level_review;

    return (
      <VStack align="stretch" spacing={4}>
        <Box>
          <Heading size="sm" mb={2}>Overall Assessment</Heading>
          <Box bg="gray.50" p={3} borderRadius="md">{overall_assessment}</Box>
        </Box>

        <Box>
          <Heading size="sm" mb={2}>Key Strengths</Heading>
          <List spacing={2}>
            {key_strengths.map((strength, idx) => (
              <ListItem key={idx} display="flex" alignItems="center">
                <ListIcon as={MdCheckCircle} color="green.500" />
                {strength}
              </ListItem>
            ))}
          </List>
        </Box>

        <Box>
          <Heading size="sm" mb={2}>Areas for Improvement</Heading>
          <List spacing={2}>
            {key_weaknesses.map((weakness, idx) => (
              <ListItem key={idx} display="flex" alignItems="center">
                <ListIcon as={MdWarning} color="orange.500" />
                {weakness}
              </ListItem>
            ))}
          </List>
        </Box>

        <Box>
          <Heading size="sm" mb={2}>Recommendations</Heading>
          <List spacing={2}>
            {recommendations.map((rec, idx) => (
              <ListItem key={idx} display="flex" alignItems="center">
                <ListIcon as={MdInfo} color="blue.500" />
                {rec}
              </ListItem>
            ))}
          </List>
        </Box>

        {communication_review && (
          <Box>
            <Heading size="sm" mb={2}>Writing Assessment</Heading>
            <Box bg="gray.50" p={3} borderRadius="md">
              {communication_review.writing_assessment}
            </Box>
          </Box>
        )}
      </VStack>
    );
  };

  // Add a console log to verify props
  console.log('DocumentViewer props:', { reviewData, fileId });

  return (
    <Container maxW="container.xl" py={4}>
      <VStack spacing={6} align="stretch">
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Heading size="md">Document Review</Heading>
          <Button
            leftIcon={<MdDownload />}
            colorScheme="blue"
            onClick={handleDownload}
          >
            Download Reviewed Document
          </Button>
        </Box>
        {renderHighLevelReview()}
      </VStack>
    </Container>
  );
};
