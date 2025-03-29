import { render, screen } from '@testing-library/react';
import App from './App'; // NOT USING
import AirMonitorDashboard from './components/AirMonitorDashboard';

test('renders Air Quality title', () => {
  render(<AirMonitorDashboard />);
  const titleElement = screen.getByText(/Air Quality/i); // Look for 'Air Quality'
  expect(titleElement).toBeInTheDocument();
});
