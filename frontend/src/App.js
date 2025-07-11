import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [userId, setUserId] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    // Generate a random user ID if not set
    if (!userId) {
      setUserId(`user_${Math.random().toString(36).substr(2, 9)}`);
    }
    
    // Load initial stats
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const response = await axios.get('/stats');
      setStats(response.data);
    } catch (error) {
      console.error('Error loading stats:', error);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage = {
      id: Date.now(),
      text: inputMessage,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post('/ask', {
        user_id: userId,
        question: inputMessage
      });

      const botMessage = {
        id: Date.now() + 1,
        text: response.data.answer,
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString(),
        matchScore: response.data.match_score,
        usedLLM: response.data.used_llm,
        llmCost: response.data.llm_cost,
        matchedFAQ: response.data.matched_faq
      };

      setMessages(prev => [...prev, botMessage]);
      
      // Reload stats after new query
      loadStats();
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error. Please try again.',
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🤖 Hawala AI Assistant</h1>
        <p>Ask me about money transfers, savings, and financial planning!</p>
      </header>

      <div className="chat-container">
        <div className="messages">
          {messages.map((message) => (
            <div key={message.id} className={`message ${message.sender}`}>
              <div className="message-content">
                <p>{message.text}</p>
                {message.sender === 'bot' && !message.isError && (
                  <div className="message-meta">
                    <span className={`score ${message.usedLLM ? 'llm' : 'faq'}`}>
                      {message.usedLLM ? '🤖 LLM' : '📚 FAQ'} (Score: {message.matchScore.toFixed(2)})
                    </span>
                    {message.usedLLM && message.llmCost && (
                      <span className="cost">Cost: ${message.llmCost.toFixed(4)}</span>
                    )}
                  </div>
                )}
                <small className="timestamp">{message.timestamp}</small>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="message bot">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="input-container">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me anything about Hawala services..."
            disabled={isLoading}
          />
          <button onClick={sendMessage} disabled={isLoading || !inputMessage.trim()}>
            Send
          </button>
        </div>
      </div>

      {stats && (
        <div className="stats">
          <h3>📊 Usage Statistics</h3>
          <div className="stats-grid">
            <div className="stat">
              <span className="stat-label">Total Queries:</span>
              <span className="stat-value">{stats.total_queries}</span>
            </div>
            <div className="stat">
              <span className="stat-label">FAQ Matches:</span>
              <span className="stat-value">{stats.faq_queries}</span>
            </div>
            <div className="stat">
              <span className="stat-label">LLM Calls:</span>
              <span className="stat-value">{stats.llm_queries}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Total Cost:</span>
              <span className="stat-value">${stats.total_cost}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Cost Savings:</span>
              <span className="stat-value">${stats.cost_savings}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App; 