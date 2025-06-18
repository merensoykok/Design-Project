import { useState } from 'react';
import './App.css';
import { GiBrainTentacle } from 'react-icons/gi';
import ChatBox from './components/ChatBox';

function App() {
  const [showChat, setShowChat] = useState(false);

  const toggleChat = () => {
    setShowChat(!showChat);
  };

  return (
    <div className="App">
      <header className="header">
        <GiBrainTentacle className="header-logo" />
        <h1>Fashion-Gen</h1>
      </header>
      
      <main className="main-content">
        {!showChat && (
          <button className="start-chat-btn" onClick={toggleChat}>
            Start Chat
          </button>
        )}
        {showChat && <ChatBox onClose={toggleChat} />}
      </main>
      
      <footer className="footer">
        <p>© {new Date().getFullYear()} Fashion-Gen. All Rights Reserved.</p>
      </footer>
    </div>
  );
}

export default App;
