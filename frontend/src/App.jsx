import { ChakraProvider, CSSReset, Container, VStack } from '@chakra-ui/react';
import { DocumentUpload } from './components/DocumentUpload';
import { DocumentViewer } from './components/DocumentViewer';
import { useState } from 'react';

function App() {
  const [reviewData, setReviewData] = useState(null);
  const [fileId, setFileId] = useState(null);

  const handleDocumentProcessed = (data) => {
    setReviewData(data);
    setFileId(data.fileId);
  };

  return (
    <ChakraProvider>
      <CSSReset />
      <Container maxW="container.xl" py={8}>
        <VStack spacing={8} align="stretch">
          <DocumentUpload onDocumentProcessed={handleDocumentProcessed} />
          {reviewData && <DocumentViewer reviewData={reviewData} fileId={fileId} />}
        </VStack>
      </Container>
    </ChakraProvider>
  );
}

export default App;
