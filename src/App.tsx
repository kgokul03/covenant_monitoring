import React from 'react';
import './App.css';
import Header from './components/header.tsx';
import Footer from './components/footer.tsx';
import img from './covenant.jpg';

const App: React.FC = () => {
  const appStyle = {
    position: 'relative' as 'relative',
    backgroundImage: `url(${img})`,
    backgroundSize: 'cover',
    backgroundPosition: 'center center',
    backgroundAttachment: 'fixed',
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    color: 'white',  // White text for contrast against the background
  };

  // Overlay styles to darken the background for better contrast
  const overlayStyle = {
    position: 'absolute' as 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.4)',  // Black overlay with 40% opacity
  };

  return (
    <div style={appStyle}>
      <div style={overlayStyle}></div> {/* Overlay */}
      <Header />
      <main style={{ flex: 1, zIndex: 1 }}>
        <h2>Welcome to My App</h2>
        <p>This is a basic layout with a header and footer, and a background image.</p>
      </main>
      <Footer />
    </div>
  );
};

export default App;
