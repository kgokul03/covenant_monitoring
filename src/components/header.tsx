// src/Header.tsx
import React from 'react';

const Header: React.FC = () => {
  return (
    <header style={styles.header}>
      <h1>Covenants Monitor</h1>
    </header>
  );
};

const styles = {
  header: {
    backgroundColor: '#D9D9D9',
    color: 'white',
    padding: '10px 20px',
    textAlign: 'right' as 'right',
  },
};

export default Header;
