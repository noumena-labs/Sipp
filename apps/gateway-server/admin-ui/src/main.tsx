import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.js';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 2000,
      refetchOnWindowFocus: false,
    },
  },
});

const root = document.getElementById('root');
if (root == null) {
  throw new Error('Dashboard root element is missing.');
}

ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>
);
