import React, { useState } from 'react';
import '../styles/GeneratedImage.css';

const GeneratedImage = ({ imageUrl }) => {
  const [imageError, setImageError] = useState(false);

  const handleImageError = () => {
    setImageError(true);
  };

  // Check if the imageUrl is a base64 string or a regular URL
  const isBase64 = imageUrl && 
    (imageUrl.startsWith('data:image') || 
     imageUrl.startsWith('data:application/octet-stream'));

  return (
    <div className="generated-image">
      {!imageError ? (
        <img 
          src={imageUrl} 
          alt="Generated fashion design" 
          onError={handleImageError}
          className={isBase64 ? "user-uploaded-image" : ""}
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