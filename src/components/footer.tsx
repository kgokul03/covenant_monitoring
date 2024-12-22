// src/Footer.tsx
import React from 'react';

const Footer: React.FC = () => {
  return (
    <footer style={styles.footer}>
      <p>My App Footer - &copy; 2024</p>
    </footer>
  );
};

const styles = {
  footer: {
    backgroundColor: '#333',
    color: 'white',
    padding: '10px 20px',
    textAlign: 'center' as 'center',
    position: 'fixed' as 'fixed',
    bottom: '0',
    width: '100%',
  },
};

export default Footer;
