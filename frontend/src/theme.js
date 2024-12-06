import { extendTheme } from '@chakra-ui/react';

const theme = extendTheme({
  colors: {
    cambridge: {
      blue: '#A3C1AD',
      darkBlue: '#1E2D24',
      gold: '#C49B33',
      cream: '#F5F2E9',
      brown: '#4A3C31',
      red: '#7A282D'
    }
  },
  styles: {
    global: {
      body: {
        bg: '#F5F2E9',
        color: '#1E2D24',
        fontFamily: 'Crimson Text, Georgia, serif'
      }
    }
  },
  components: {
    Button: {
      baseStyle: {
        fontFamily: 'Crimson Text, Georgia, serif',
      },
      variants: {
        solid: {
          bg: '#1E2D24',
          color: '#F5F2E9',
          _hover: {
            bg: '#4A3C31',
          }
        },
        outline: {
          border: '2px solid',
          borderColor: '#1E2D24',
          color: '#1E2D24',
          _hover: {
            bg: '#A3C1AD',
          }
        }
      }
    },
    Container: {
      baseStyle: {
        maxW: 'container.xl',
        px: 6,
        py: 8,
      }
    }
  }
});

export default theme;