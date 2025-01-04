import { VStack, Text, Circle, HStack } from '@chakra-ui/react';

export const ApiStatusIndicator = ({ isConnected, apiUrl }) => {
  console.log('ApiStatusIndicator: Received URL:', apiUrl);
  
  return (
    <VStack align="flex-end" spacing={1}>
      <HStack spacing={2}>
        <Text fontSize="sm" color={isConnected ? "green.600" : "red.600"}>
          API {isConnected ? "Connected" : "Disconnected"}
        </Text>
        <Circle size="10px" bg={isConnected ? "green.500" : "red.500"} />
      </HStack>
      <Text 
        fontSize="xs" 
        color="gray.500" 
        fontFamily="monospace"
        visibility={apiUrl ? "visible" : "hidden"}
      >
        {apiUrl}
      </Text>
    </VStack>
  );
}; 