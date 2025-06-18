import React, { useRef, useEffect, useState } from 'react';
import { FaUndo, FaTrash, FaCheck, FaTimes, FaPaintBrush, FaFill } from 'react-icons/fa';
import '../styles/ImageMaskEditor.css';

const ImageMaskEditor = ({ imageUrl, onSave, onCancel }) => {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [brushSize, setBrushSize] = useState(20);
  const [canvasContext, setCanvasContext] = useState(null);
  const [originalImageData, setOriginalImageData] = useState(null);
  const [tool, setTool] = useState('brush'); // 'brush' or 'fill'
  const [imageScale, setImageScale] = useState(1);
  const [imageOffset, setImageOffset] = useState({ x: 0, y: 0 });

  // Fixed canvas dimensions
  const CANVAS_MAX_WIDTH = 800;
  const CANVAS_MAX_HEIGHT = 600;

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    setCanvasContext(ctx);

    // Load and display the image
    const img = new Image();
    img.onload = () => {
      // Calculate scaling to fit within fixed canvas size
      const scaleX = CANVAS_MAX_WIDTH / img.width;
      const scaleY = CANVAS_MAX_HEIGHT / img.height;
      const scale = Math.min(scaleX, scaleY, 1); // Don't scale up, only down
      
      const scaledWidth = img.width * scale;
      const scaledHeight = img.height * scale;
      
      // Set fixed canvas size
      canvas.width = CANVAS_MAX_WIDTH;
      canvas.height = CANVAS_MAX_HEIGHT;
      
      // Calculate centering offset
      const offsetX = (CANVAS_MAX_WIDTH - scaledWidth) / 2;
      const offsetY = (CANVAS_MAX_HEIGHT - scaledHeight) / 2;
      
      setImageScale(scale);
      setImageOffset({ x: offsetX, y: offsetY });
      
      // Clear canvas with white background
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Draw the scaled and centered image
      ctx.drawImage(img, offsetX, offsetY, scaledWidth, scaledHeight);
      
      // Store original image data for undo functionality
      setOriginalImageData(ctx.getImageData(0, 0, canvas.width, canvas.height));
    };
    img.src = imageUrl;
  }, [imageUrl]);

  const getCanvasCoordinates = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  };

  const startDrawing = (e) => {
    const coords = getCanvasCoordinates(e);
    
    if (tool === 'fill') {
      floodFill(coords.x, coords.y);
    } else {
      setIsDrawing(true);
      draw(e);
    }
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    if (canvasContext) {
      canvasContext.beginPath();
    }
  };

  const draw = (e) => {
    if (!isDrawing || !canvasContext || tool !== 'brush') return;

    const coords = getCanvasCoordinates(e);

    canvasContext.lineWidth = brushSize;
    canvasContext.lineCap = 'round';
    canvasContext.strokeStyle = 'rgba(255, 0, 0, 0.7)'; // Semi-transparent red for drawing
    
    canvasContext.lineTo(coords.x, coords.y);
    canvasContext.stroke();
    canvasContext.beginPath();
    canvasContext.moveTo(coords.x, coords.y);
  };

  const floodFill = (startX, startY) => {
    if (!canvasContext) return;

    const imageData = canvasContext.getImageData(0, 0, canvasRef.current.width, canvasRef.current.height);
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    
    const startIndex = (Math.floor(startY) * width + Math.floor(startX)) * 4;
    const startR = data[startIndex];
    const startG = data[startIndex + 1];
    const startB = data[startIndex + 2];
    const startA = data[startIndex + 3];
    
    const fillR = 255;
    const fillG = 0;
    const fillB = 0;
    const fillA = 178; // 0.7 * 255 for semi-transparency
    
    // Check if we're starting on a red pixel (already drawn area) - don't fill
    if (startR > 200 && startG < 100 && startB < 100 && startA > 100) {
      return;
    }
    
    // Helper function to check if a pixel is red (drawn boundary)
    const isRedPixel = (x, y) => {
      if (x < 0 || x >= width || y < 0 || y >= height) return true; // Treat boundaries as walls
      const index = (y * width + x) * 4;
      const r = data[index];
      const g = data[index + 1];
      const b = data[index + 2];
      const a = data[index + 3];
      return r > 200 && g < 100 && b < 100 && a > 100;
    };
    
    // Helper function to check if a pixel should be filled (not red and not already filled)
    const shouldFill = (x, y) => {
      if (x < 0 || x >= width || y < 0 || y >= height) return false;
      const index = (y * width + x) * 4;
      const r = data[index];
      const g = data[index + 1];
      const b = data[index + 2];
      const a = data[index + 3];
      
      // Don't fill if it's already red (drawn area)
      if (r > 200 && g < 100 && b < 100 && a > 100) return false;
      
      return true;
    };
    
    const pixelStack = [[Math.floor(startX), Math.floor(startY)]];
    const visited = new Set();
    
    while (pixelStack.length > 0) {
      const [x, y] = pixelStack.pop();
      const key = `${x},${y}`;
      
      if (visited.has(key) || !shouldFill(x, y)) {
        continue;
      }
      
      visited.add(key);
      
      // Fill this pixel
      const index = (y * width + x) * 4;
      data[index] = fillR;
      data[index + 1] = fillG;
      data[index + 2] = fillB;
      data[index + 3] = fillA;
      
      // Add neighboring pixels to stack if they're not red boundaries
      if (!isRedPixel(x + 1, y)) pixelStack.push([x + 1, y]);
      if (!isRedPixel(x - 1, y)) pixelStack.push([x - 1, y]);
      if (!isRedPixel(x, y + 1)) pixelStack.push([x, y + 1]);
      if (!isRedPixel(x, y - 1)) pixelStack.push([x, y - 1]);
    }
    
    canvasContext.putImageData(imageData, 0, 0);
  };

  const clearDrawing = () => {
    if (originalImageData && canvasContext) {
      canvasContext.putImageData(originalImageData, 0, 0);
    }
  };

  const generateMask = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Create a new canvas for the mask with original image dimensions
    const img = new Image();
    img.src = imageUrl;
    
    return new Promise((resolve) => {
      img.onload = () => {
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = img.width;
        maskCanvas.height = img.height;
        const maskCtx = maskCanvas.getContext('2d');
        
        // Fill with black background
        maskCtx.fillStyle = 'black';
        maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
        
        // Get current canvas image data
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        
        // Create mask data at original resolution
        const maskImageData = maskCtx.createImageData(img.width, img.height);
        const maskData = maskImageData.data;
        
        // Scale coordinates back to original image size
        for (let y = 0; y < img.height; y++) {
          for (let x = 0; x < img.width; x++) {
            // Map original image coordinates to canvas coordinates
            const canvasX = Math.floor(x * imageScale + imageOffset.x);
            const canvasY = Math.floor(y * imageScale + imageOffset.y);
            
            if (canvasX >= 0 && canvasX < CANVAS_MAX_WIDTH && canvasY >= 0 && canvasY < CANVAS_MAX_HEIGHT) {
              const canvasIndex = (canvasY * CANVAS_MAX_WIDTH + canvasX) * 4;
              const maskIndex = (y * img.width + x) * 4;
              
              const r = data[canvasIndex];
              const g = data[canvasIndex + 1];
              const b = data[canvasIndex + 2];
              const a = data[canvasIndex + 3];
              
              // If pixel has red color (from drawing), make it white in mask
              if (r > 200 && g < 100 && b < 100 && a > 100) {
                maskData[maskIndex] = 255;     // R
                maskData[maskIndex + 1] = 255; // G
                maskData[maskIndex + 2] = 255; // B
                maskData[maskIndex + 3] = 255; // A
              } else {
                maskData[maskIndex] = 0;       // R
                maskData[maskIndex + 1] = 0;   // G
                maskData[maskIndex + 2] = 0;   // B
                maskData[maskIndex + 3] = 255; // A
              }
            }
          }
        }
        
        maskCtx.putImageData(maskImageData, 0, 0);
        resolve(maskCanvas.toDataURL());
      };
    });
  };

  const handleSave = async () => {
    const maskDataUrl = await generateMask();
    onSave(maskDataUrl);
  };

  return (
    <div className="image-mask-editor-overlay">
      <div className="image-mask-editor">
        <div className="editor-header">
          <h3>Draw on the outfit parts you want to modify</h3>
          <button className="close-editor-btn" onClick={onCancel}>
            <FaTimes />
          </button>
        </div>
        
        <div className="editor-controls">
          <div className="tool-selection">
            <button 
              className={`tool-btn ${tool === 'brush' ? 'active' : ''}`}
              onClick={() => setTool('brush')}
            >
              <FaPaintBrush /> Brush
            </button>
            <button 
              className={`tool-btn ${tool === 'fill' ? 'active' : ''}`}
              onClick={() => setTool('fill')}
            >
              <FaFill /> Fill
            </button>
          </div>
          
          {tool === 'brush' && (
            <div className="brush-control">
              <label>Brush Size: {brushSize}px</label>
              <input
                type="range"
                min="5"
                max="100"
                value={brushSize}
                onChange={(e) => setBrushSize(parseInt(e.target.value))}
              />
            </div>
          )}
          
          <div className="editor-buttons">
            <button className="editor-btn clear-btn" onClick={clearDrawing}>
              <FaTrash /> Clear
            </button>
            <button className="editor-btn save-btn" onClick={handleSave}>
              <FaCheck /> Create Mask
            </button>
          </div>
        </div>
        
        <div className="canvas-container">
          <canvas
            ref={canvasRef}
            onMouseDown={startDrawing}
            onMouseUp={stopDrawing}
            onMouseMove={draw}
            onMouseLeave={stopDrawing}
            className="drawing-canvas"
            style={{ cursor: tool === 'brush' ? 'crosshair' : 'pointer' }}
          />
        </div>
        
        <div className="instructions">
          <p>
            {tool === 'brush' 
              ? "Draw over the parts of the outfit you want to modify. The red areas will become the mask." 
              : "Click inside areas bounded by red lines to fill them. Draw boundaries with the brush first, then use fill to complete enclosed areas."
            }
          </p>
        </div>
      </div>
    </div>
  );
};

export default ImageMaskEditor; 