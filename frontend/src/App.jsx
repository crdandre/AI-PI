import { ChakraProvider, CSSReset, Container, VStack, HStack, Checkbox, Spacer } from '@chakra-ui/react';
import { OfficeScene } from './components/OfficeScene';
import { DocumentDownloader } from './components/DocumentDownload';
import { useState } from 'react';
import theme from './theme';

function App() {
  const [reviewData, setReviewData] = useState(null);
  const [isTestMode, setIsTestMode] = useState(false);

  const handleDocumentProcessed = (data) => {
    if (!data || !data.fileId) {
      console.error('Invalid document data received:', data);
      return;
    }
    setReviewData(data);
  };

  return (
    <ChakraProvider theme={theme}>
      <CSSReset />
      <Container maxW="container.xl" py={12}>
        <VStack spacing={8} align="stretch">
          <HStack>
            <Spacer />
            <Checkbox 
              isChecked={isTestMode}
              onChange={(e) => setIsTestMode(e.target.checked)}
              colorScheme="blue"
            >
              Test Mode
            </Checkbox>
          </HStack>
          <OfficeScene 
            onDocumentProcessed={handleDocumentProcessed} 
            isTestMode={isTestMode}
          />
          {reviewData && (
            <DocumentDownloader 
              reviewData={reviewData} 
              fileId={reviewData.fileId} 
            />
          )}
        </VStack>
      </Container>
    </ChakraProvider>
  );
}

export default App;