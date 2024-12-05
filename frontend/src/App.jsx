import { ChakraProvider, CSSReset } from '@chakra-ui/react';
import { DocumentUpload } from './components/DocumentUpload';

function App() {
  return (
    <ChakraProvider>
      <CSSReset />
      <DocumentUpload />
    </ChakraProvider>
  );
}

export default App;
