import React, { useState } from 'react';
import '../styles/GeneratedImage.css';

const GeneratedImage = ({ imageUrl }) => {
  const [imageError, setImageError] = useState(false);

  const handleImageError = () => {
    setImageError(true);
  };

  // All images should use the same class for consistent sizing
  return (
    <div className="generated-image">
      {!imageError ? (
        <img 
          src={imageUrl} 
          alt="Generated fashion design" 
          onError={handleImageError}
          className="chat-image"
        />
      ) : (
        <div className="image-error">
          <p>Image could not be loaded</p>
        </div>
      )}
    </div>
  );
};

export default GeneratedImage; 