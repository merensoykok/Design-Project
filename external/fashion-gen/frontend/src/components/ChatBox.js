import React, { useState, useRef } from 'react';
import { FaUser, FaRobot, FaImage, FaTimes } from 'react-icons/fa';
import '../styles/ChatBox.css';
import GeneratedImage from './GeneratedImage';
import ImageMaskEditor from './ImageMaskEditor';

const ChatBox = ({ onClose }) => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState('');
  const [showMaskEditor, setShowMaskEditor] = useState(false);
  const [generatedMask, setGeneratedMask] = useState(null);
  const fileInputRef = useRef(null);

  const handleImageSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(file);
      
      // Create preview URL
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreviewUrl(reader.result);
        // Immediately open mask editor instead of showing preview
        setShowMaskEditor(true);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleRemoveImage = () => {
    setSelectedImage(null);
    setImagePreviewUrl('');
    setGeneratedMask(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleMaskSave = (maskDataUrl) => {
    setGeneratedMask(maskDataUrl);
    setShowMaskEditor(false);
    // The mask is now ready to be sent with the message
  };

  const handleMaskCancel = () => {
    setShowMaskEditor(false);
    // Clear the selected image since user cancelled
    handleRemoveImage();
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if ((!inputText.trim() && !generatedMask) || isLoading) return;

    // Create the request body
    const requestBody = {};
    
    // Only add properties if they have values
    if (inputText.trim()) {
      requestBody.prompt = inputText;
    }
    
    if (generatedMask) {
      requestBody.maskData = generatedMask;
      requestBody.originalImage = imagePreviewUrl;
    }

    const newUserMessage = {
      text: inputText || "Image with mask uploaded",
      sender: 'user',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      image: generatedMask || null
    };

    setMessages([...messages, newUserMessage]);
    setInputText('');
    handleRemoveImage();
    setIsLoading(true);

    try {
      // Send request to backend
      const response = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }
      
      const data = await response.json();
      
      // Log the generated prompt to console
      if (data.response) {
        console.log('Generated Prompt:', data.response);
      }
      
      // Create AI response - handle different types of responses
      let aiResponse;
      
      if (data.generatedImage) {
        // Handle different types of generated images
        let responseText;
        if (data.type === 'custom_controlnet_inpaint') {
          responseText = `Custom ControlNet inpainting completed: ${data.response}`;
        } else if (data.type === 'text2img') {
          responseText = `Generated fashion design: ${data.response}`;
        } else {
          responseText = `Generated fashion design: ${data.response}`;
        }
        
        aiResponse = {
          text: responseText,
          sender: 'ai',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          image: data.generatedImage
        };
      } else if (data.maskUrl) {
        // Mask testing response
        aiResponse = {
          text: "Here's your generated mask:",
          sender: 'ai',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          image: data.maskUrl
        };
      } else if (data.imageUrl) {
        // Legacy static image response
        aiResponse = {
          text: "Here's a fashion design based on your description.",
          sender: 'ai',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          image: data.imageUrl
        };
      } else {
        // Text-only response
        aiResponse = {
          text: generatedMask ? "Here's your generated mask:" : "Here's a fashion design based on your description.",
          sender: 'ai',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          image: generatedMask || (inputText ? 'https://placehold.co/400x500/3498db/ffffff?text=Fashion+Design' : null)
        };
      }
      
      setMessages(prev => [...prev, aiResponse]);
    } catch (error) {
      console.error('Error communicating with backend:', error);
      
      // Show error message if backend communication fails
      const errorResponse = {
        text: "Sorry, I couldn't process your request. Please try again later.",
        sender: 'ai',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      
      setMessages(prev => [...prev, errorResponse]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
    <div className="chat-box">
      <div className="chat-header">
        <h3>Fashion-Gen AI</h3>
        <button className="close-btn" onClick={onClose}>×</button>
      </div>
      
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="empty-chat">
            <p>Send a message to start generating fashion designs!</p>
          </div>
        ) : (
          messages.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}>
              <div className="message-avatar">
                {msg.sender === 'user' ? <FaUser /> : <FaRobot />}
              </div>
              <div className="message-content">
                <div className="message-text">{msg.text}</div>
                {msg.image && <GeneratedImage imageUrl={msg.image} />}
                <div className="message-time">{msg.timestamp}</div>
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="message ai">
            <div className="message-avatar">
              <FaRobot />
            </div>
            <div className="message-content">
              <div className="loading-indicator">
                <span>Generating design</span>
                <div className="loading-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
      
      <form className="chat-input" onSubmit={handleSendMessage}>
          {generatedMask && (
          <div className="image-preview-container">
              <img src={generatedMask} alt="Generated Mask" className="image-preview" />
            <button 
              type="button" 
              className="remove-image-btn"
              onClick={handleRemoveImage}
            >
              <FaTimes />
            </button>
              <span className="mask-label">Mask Ready</span>
          </div>
        )}
        
        <div className="input-container">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Describe a fashion item..."
            disabled={isLoading}
          />
          
          <label className="image-upload-btn">
            <FaImage />
            <input
              type="file"
              accept="image/*"
              onChange={handleImageSelect}
              disabled={isLoading}
              ref={fileInputRef}
              style={{ display: 'none' }}
            />
          </label>
          
          <button 
            type="submit" 
              disabled={isLoading || (!inputText.trim() && !generatedMask)}
          >
            Send
          </button>
        </div>
      </form>
    </div>
      
      {showMaskEditor && (
        <ImageMaskEditor
          imageUrl={imagePreviewUrl}
          onSave={handleMaskSave}
          onCancel={handleMaskCancel}
        />
      )}
    </>
  );
};

export default ChatBox; 