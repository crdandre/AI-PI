import { ChakraProvider, CSSReset, Container, VStack, HStack, Checkbox, Spacer } from '@chakra-ui/react';
import { OfficeScene } from './components/OfficeScene';
import { DocumentDownloader } from './components/DocumentDownload';
import { useState } from 'react';
import theme from './theme';
import { ApiStatusIndicator } from './components/ApiStatusIndicator';

function App() {
  const [reviewData, setReviewData] = useState(null);
  const [isTestMode, setIsTestMode] = useState(false);
  const [apiConnected, setApiConnected] = useState(false);
  const [apiUrl, setApiUrl] = useState('http://localhost:8001');

  const handleDocumentProcessed = (data) => {
    if (!data || !data.fileId) {
      console.error('Invalid document data received:', data);
      return;
    }
    setReviewData(data);
  };

  const handleApiUrlChange = (url) => {
    console.log('App: Updating API URL to', url);
    setApiUrl(url);
  };

  return (
    <ChakraProvider theme={theme}>
      <CSSReset />
      <Container maxW="container.xl" py={12}>
        <VStack spacing={8} align="stretch">
          <VStack align="flex-end">
            <Checkbox 
              isChecked={isTestMode}
              onChange={(e) => setIsTestMode(e.target.checked)}
              colorScheme="blue"
            >
              Test Mode
            </Checkbox>
            {console.log('App: Rendering ApiStatusIndicator with URL:', apiUrl)}
            <ApiStatusIndicator 
              isConnected={apiConnected} 
              apiUrl={apiUrl} 
            />
          </VStack>
          <OfficeScene 
            onDocumentProcessed={handleDocumentProcessed} 
            isTestMode={isTestMode}
            onApiStatusChange={setApiConnected}
            onApiUrlChange={handleApiUrlChange}
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