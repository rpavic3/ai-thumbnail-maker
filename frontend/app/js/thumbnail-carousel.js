/**
 * Thumbnail Carousel
 * Handles the sliding animation of thumbnails on both sides of the main content.
 */

class ThumbnailCarousel {
  constructor(options = {}) {
    this.options = {
      leftContainerId: 'left-carousel',
      rightContainerId: 'right-carousel',
      leftDirection: 'up', // Direction: 'up' or 'down'
      rightDirection: 'down', // Direction: 'up' or 'down'
      speed: 0.3, // Speed increased from 0.2 to 0.5 for faster animation
      gap: 15, // Gap between thumbnails
      thumbnailWidth: 410, // Width of thumbnails (increased from 240)
      thumbnailHeight: 260, // Height of thumbnails (increased from 140)
      columns: 1, // Number of columns on each side (changed from 2 to 1)
      columnGap: 15, // Gap between columns
      sideMargin: 63, // Margin from the edge of the screen
      ...options
    };
    
    this.leftContainer = document.getElementById(this.options.leftContainerId);
    this.rightContainer = document.getElementById(this.options.rightContainerId);
    
    this.leftThumbnails = [];
    this.rightThumbnails = [];
    
    // Animation frame IDs for left and right animations
    this.leftAnimationFrameId = null;
    this.rightAnimationFrameId = null;
    
    // Timestamp tracking for smooth animation
    this.lastLeftUpdateTime = 0;
    this.lastRightUpdateTime = 0;
    
    this.init();
  }
  
  init() {
    if (!this.leftContainer || !this.rightContainer) {
      console.error('Carousel containers not found');
      return;
    }
    
    // Apply styles to containers
    this.applyContainerStyles();
    
    // Load curated showcase thumbnails
    this.loadCuratedThumbnails();
    
    // Start animation
    this.startAnimation();
  }
  
  applyContainerStyles() {
    // Calculate total width based on number of columns, thumbnail width, and column gap
    const totalWidth = (this.options.thumbnailWidth * this.options.columns) + 
                      (this.options.columnGap * (this.options.columns - 1));
    
    const commonStyles = {
      position: 'absolute',
      top: '0',
      bottom: '0',
      width: `${totalWidth}px`,
      overflow: 'hidden'
    };
    
    Object.assign(this.leftContainer.style, {
      ...commonStyles,
      left: `${this.options.sideMargin}px` // Add margin from the left edge
    });
    
    Object.assign(this.rightContainer.style, {
      ...commonStyles,
      right: `${this.options.sideMargin}px` // Add margin from the right edge
    });
  }
  
  loadCuratedThumbnails() {
    // Images in the showcase directory - not using these currently
    const showcaseImages = [
      'images/showcase/thumbnail1.jpg',
      'images/showcase/thumbnail2.jpg',
      'images/showcase/thumbnail3.jpg',
      'images/showcase/thumbnail4.jpg'
    ];
    
    // Use images directly from the images directory
    const thumbnailImages = [
      'images/thumbnail1.jpg',
      'images/thumbnail2.jpg',
      'images/thumbnail3.jpg',
      'images/thumbnail4.jpg',
      'images/thumbnail5.jpg',
      'images/thumbnail6.jpg',
      'images/thumbnail7.jpg',
      'images/thumbnail1.png',
      'images/thumbnail2.png',
      'images/thumbnail3.png',
      'images/thumbnail8.jpg',
      'images/thumbnail9.jpg',
      'images/thumbnail10.jpg',
      'images/thumbnail11.jpg',
      'images/thumbnail12.jpg',
    ];
    
    // Use the thumbnail images from the main images directory
    const imagesToUse = thumbnailImages;
    
    // Create enough thumbnails to fill the containers with some duplication for continuous scrolling
    const containerHeight = window.innerHeight;
    const thumbnailTotalHeight = this.options.thumbnailHeight + this.options.gap;
    // Increase the number of thumbnails to ensure continuous scrolling
    const requiredThumbnailsPerColumn = Math.ceil(containerHeight / thumbnailTotalHeight) * 2 + 2;
    
    // Shuffle the images array for left side
    const leftImages = this.createNonDuplicateSequence(imagesToUse, requiredThumbnailsPerColumn * this.options.columns);
    
    // Create a different shuffle for right side (can contain same images as left, but no duplicates within itself)
    const rightImages = this.createNonDuplicateSequence(imagesToUse, requiredThumbnailsPerColumn * this.options.columns);
    
    // Create thumbnails for left container
    for (let col = 0; col < this.options.columns; col++) {
      for (let i = 0; i < requiredThumbnailsPerColumn; i++) {
        const imgIndex = (i + col * requiredThumbnailsPerColumn) % leftImages.length;
        const imgSrc = leftImages[imgIndex];
        const xOffset = col * (this.options.thumbnailWidth + this.options.columnGap);
        this.addThumbnail(this.leftContainer, imgSrc, this.leftThumbnails, xOffset);
      }
    }
    
    // Create thumbnails for right container
    for (let col = 0; col < this.options.columns; col++) {
      for (let i = 0; i < requiredThumbnailsPerColumn; i++) {
        const imgIndex = (i + col * requiredThumbnailsPerColumn) % rightImages.length;
        const imgSrc = rightImages[imgIndex];
        const xOffset = col * (this.options.thumbnailWidth + this.options.columnGap);
        this.addThumbnail(this.rightContainer, imgSrc, this.rightThumbnails, xOffset);
      }
    }
  }
  
  // Helper method to create a sequence with no duplicates next to each other
  createNonDuplicateSequence(sourceImages, length) {
    // First create a shuffled copy of the source images
    const shuffled = [...sourceImages];
    this.shuffleArray(shuffled);
    
    // If we need more images than we have unique ones, we need to repeat them
    const result = [];
    
    // Fill the result array to the required length
    for (let i = 0; i < length; i++) {
      let nextImageIndex = i % shuffled.length;
      
      // If this would create a duplicate with the previous image, try the next one
      if (result.length > 0 && shuffled[nextImageIndex] === result[result.length - 1]) {
        nextImageIndex = (nextImageIndex + 1) % shuffled.length;
      }
      
      result.push(shuffled[nextImageIndex]);
    }
    
    return result;
  }
  
  // Helper method to shuffle an array (Fisher-Yates algorithm)
  shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
  }
  
  addThumbnail(container, imgSrc, thumbnailsArray, xOffset = 0) {
    const thumbnailDiv = document.createElement('div');
    thumbnailDiv.className = 'carousel-thumbnail';
    thumbnailDiv.style.position = 'absolute';
    thumbnailDiv.style.width = `${this.options.thumbnailWidth}px`;
    thumbnailDiv.style.height = `${this.options.thumbnailHeight}px`;
    thumbnailDiv.style.borderRadius = 'var(--radius, 8px)';
    thumbnailDiv.style.overflow = 'hidden';
    thumbnailDiv.style.boxShadow = '0 4px 10px rgba(0,0,0,0.2)';
    thumbnailDiv.style.transition = 'transform 0.3s ease';
    thumbnailDiv.style.left = `${xOffset}px`;
    
    const img = document.createElement('img');
    img.src = imgSrc;
    img.alt = 'Thumbnail';
    img.style.width = '100%';
    img.style.height = '100%';
    img.style.objectFit = 'cover';
    
    // Add error handling for image loading
    img.onerror = () => {
      console.warn(`Failed to load image: ${imgSrc}, using fallback`);
      // Use a fallback image if the requested one fails to load
      img.src = 'images/thumbnail1.jpg';
    };
    
    thumbnailDiv.appendChild(img);
    container.appendChild(thumbnailDiv);
    
    // Add hover effect
    thumbnailDiv.addEventListener('mouseenter', () => {
      thumbnailDiv.style.transform = 'scale(1.05)';
      thumbnailDiv.style.zIndex = '1';
    });
    
    thumbnailDiv.addEventListener('mouseleave', () => {
      thumbnailDiv.style.transform = 'scale(1)';
      thumbnailDiv.style.zIndex = '0';
    });
    
    thumbnailsArray.push({
      element: thumbnailDiv,
      height: this.options.thumbnailHeight + this.options.gap,
      xOffset: xOffset,
      top: 0, // Initialize top position
      imgSrc: imgSrc // Store the image source for tracking duplicates
    });
  }
  
  // Method to update carousel with new thumbnails if needed
  updateThumbnails(images) {
    if (!images || images.length === 0) return;
    
    // Stop animation while we update the thumbnails
    this.stopAnimation();
    
    // Clear existing thumbnails
    this.leftContainer.innerHTML = '';
    this.rightContainer.innerHTML = '';
    this.leftThumbnails = [];
    this.rightThumbnails = [];
    
    // Create enough thumbnails to fill the containers with some duplication for continuous scrolling
    const containerHeight = window.innerHeight;
    const thumbnailTotalHeight = this.options.thumbnailHeight + this.options.gap;
    // Increase the number of thumbnails to ensure continuous scrolling
    const requiredThumbnailsPerColumn = Math.ceil(containerHeight / thumbnailTotalHeight) * 2 + 2;
    
    // Create non-duplicate sequences for both sides
    const leftImages = this.createNonDuplicateSequence(images, requiredThumbnailsPerColumn * this.options.columns);
    const rightImages = this.createNonDuplicateSequence(images, requiredThumbnailsPerColumn * this.options.columns);
    
    // Create thumbnails for left container
    for (let col = 0; col < this.options.columns; col++) {
      for (let i = 0; i < requiredThumbnailsPerColumn; i++) {
        const imgIndex = (i + col * requiredThumbnailsPerColumn) % leftImages.length;
        const imgSrc = leftImages[imgIndex];
        const xOffset = col * (this.options.thumbnailWidth + this.options.columnGap);
        this.addThumbnail(this.leftContainer, imgSrc, this.leftThumbnails, xOffset);
      }
    }
    
    // Create thumbnails for right container
    for (let col = 0; col < this.options.columns; col++) {
      for (let i = 0; i < requiredThumbnailsPerColumn; i++) {
        const imgIndex = (i + col * requiredThumbnailsPerColumn) % rightImages.length;
        const imgSrc = rightImages[imgIndex];
        const xOffset = col * (this.options.thumbnailWidth + this.options.columnGap);
        this.addThumbnail(this.rightContainer, imgSrc, this.rightThumbnails, xOffset);
      }
    }
    
    // Initialize positions
    this.initializePositions();
    
    // Restart animation with fresh state
    this.startAnimation();
  }
  
  initializePositions() {
    // Group thumbnails by their x-offset (column)
    const leftColumnMap = new Map();
    const rightColumnMap = new Map();
    
    // Group left thumbnails by column
    this.leftThumbnails.forEach(thumbnail => {
      const column = leftColumnMap.get(thumbnail.xOffset) || [];
      column.push(thumbnail);
      leftColumnMap.set(thumbnail.xOffset, column);
    });
    
    // Group right thumbnails by column
    this.rightThumbnails.forEach(thumbnail => {
      const column = rightColumnMap.get(thumbnail.xOffset) || [];
      column.push(thumbnail);
      rightColumnMap.set(thumbnail.xOffset, column);
    });
    
    // Position left thumbnails by column
    for (const [xOffset, columnThumbnails] of leftColumnMap.entries()) {
      // Sort thumbnails by position
      columnThumbnails.sort((a, b) => a.top - b.top);
      
      // Distribute thumbnails from top to bottom with some above the viewport
      const totalHeight = columnThumbnails.reduce((sum, t) => sum + t.height, 0);
      let currentYPos = -totalHeight / 4; // Start some thumbnails above the viewport
      
      columnThumbnails.forEach(thumbnail => {
        thumbnail.top = currentYPos;
        thumbnail.element.style.top = `${currentYPos}px`;
        thumbnail.element.style.left = `${xOffset}px`;
        currentYPos += thumbnail.height;
      });
    }
    
    // Position right thumbnails by column
    for (const [xOffset, columnThumbnails] of rightColumnMap.entries()) {
      // Sort thumbnails by position
      columnThumbnails.sort((a, b) => a.top - b.top);
      
      // Distribute thumbnails from top to bottom with some above the viewport
      const totalHeight = columnThumbnails.reduce((sum, t) => sum + t.height, 0);
      let currentYPos = -totalHeight / 4; // Start some thumbnails above the viewport
      
      columnThumbnails.forEach(thumbnail => {
        thumbnail.top = currentYPos;
        thumbnail.element.style.top = `${currentYPos}px`;
        thumbnail.element.style.left = `${xOffset}px`;
        currentYPos += thumbnail.height;
      });
    }
  }
  
  startAnimation() {
    // Clear any existing animations
    this.stopAnimation();
    
    // Initialize positions
    this.initializePositions();
    
    // Reset timestamps
    this.lastLeftUpdateTime = performance.now();
    this.lastRightUpdateTime = performance.now();
    
    // Start animations using requestAnimationFrame for smoother rendering
    this.animateLeftCarousel();
    this.animateRightCarousel();
  }
  
  stopAnimation() {
    // Cancel any pending animation frames
    if (this.leftAnimationFrameId) {
      cancelAnimationFrame(this.leftAnimationFrameId);
      this.leftAnimationFrameId = null;
    }
    
    if (this.rightAnimationFrameId) {
      cancelAnimationFrame(this.rightAnimationFrameId);
      this.rightAnimationFrameId = null;
    }
  }
  
  animateLeftCarousel() {
    const now = performance.now();
    const deltaTime = now - this.lastLeftUpdateTime;
    
    // Only update if enough time has passed (for consistent animation speed)
    if (deltaTime > 16) { // ~60fps target
      this.moveLeftCarousel(deltaTime / 16); // Normalize by expected frame time
      this.lastLeftUpdateTime = now;
    }
    
    // Continue animation loop
    this.leftAnimationFrameId = requestAnimationFrame(() => this.animateLeftCarousel());
  }
  
  animateRightCarousel() {
    const now = performance.now();
    const deltaTime = now - this.lastRightUpdateTime;
    
    // Only update if enough time has passed (for consistent animation speed)
    if (deltaTime > 16) { // ~60fps target
      this.moveRightCarousel(deltaTime / 16); // Normalize by expected frame time
      this.lastRightUpdateTime = now;
    }
    
    // Continue animation loop
    this.rightAnimationFrameId = requestAnimationFrame(() => this.animateRightCarousel());
  }
  
  moveLeftCarousel(deltaFactor = 1) {
    // Group thumbnails by their x-offset (column)
    const columnMap = new Map();
    
    // Group thumbnails by column
    this.leftThumbnails.forEach(thumbnail => {
      const column = columnMap.get(thumbnail.xOffset) || [];
      column.push(thumbnail);
      columnMap.set(thumbnail.xOffset, column);
    });
    
    // Process each column separately
    for (const [xOffset, columnThumbnails] of columnMap.entries()) {
      // Sort thumbnails by position
      columnThumbnails.sort((a, b) => a.top - b.top);
      
      // Calculate movement based on speed and delta factor for smooth animation
      const movement = this.options.speed * deltaFactor;
      
      // Move thumbnails based on direction
      if (this.options.leftDirection === 'up') {
        // Move all thumbnails up
        columnThumbnails.forEach(thumbnail => {
          thumbnail.top -= movement;
          thumbnail.element.style.top = `${thumbnail.top}px`;
        });
        
        // Check if the first thumbnail is completely off-screen
        const firstThumbnail = columnThumbnails[0];
        if (firstThumbnail.top < -firstThumbnail.height) {
          // Get the last thumbnail
          const lastThumbnail = columnThumbnails[columnThumbnails.length - 1];
          
          // Move the first thumbnail to after the last one
          firstThumbnail.top = lastThumbnail.top + lastThumbnail.height;
          firstThumbnail.element.style.top = `${firstThumbnail.top}px`;
          
          // Reorder the array (move first to last)
          columnThumbnails.push(columnThumbnails.shift());
        }
      } else {
        // Move all thumbnails down
        columnThumbnails.forEach(thumbnail => {
          thumbnail.top += movement;
          thumbnail.element.style.top = `${thumbnail.top}px`;
        });
        
        // Check if the last thumbnail is completely off-screen at the bottom
        const lastThumbnail = columnThumbnails[columnThumbnails.length - 1];
        const viewportHeight = window.innerHeight;
        
        if (lastThumbnail.top > viewportHeight) {
          // Get the first thumbnail
          const firstThumbnail = columnThumbnails[0];
          
          // Move the last thumbnail to before the first one
          lastThumbnail.top = firstThumbnail.top - lastThumbnail.height;
          lastThumbnail.element.style.top = `${lastThumbnail.top}px`;
          
          // Reorder the array (move last to first)
          columnThumbnails.unshift(columnThumbnails.pop());
        }
      }
    }
  }
  
  moveRightCarousel(deltaFactor = 1) {
    // Group thumbnails by their x-offset (column)
    const columnMap = new Map();
    
    // Group thumbnails by column
    this.rightThumbnails.forEach(thumbnail => {
      const column = columnMap.get(thumbnail.xOffset) || [];
      column.push(thumbnail);
      columnMap.set(thumbnail.xOffset, column);
    });
    
    // Process each column separately
    for (const [xOffset, columnThumbnails] of columnMap.entries()) {
      // Sort thumbnails by position
      columnThumbnails.sort((a, b) => a.top - b.top);
      
      // Calculate movement based on speed and delta factor for smooth animation
      const movement = this.options.speed * deltaFactor;
      
      // Move thumbnails based on direction
      if (this.options.rightDirection === 'up') {
        // Move all thumbnails up
        columnThumbnails.forEach(thumbnail => {
          thumbnail.top -= movement;
          thumbnail.element.style.top = `${thumbnail.top}px`;
        });
        
        // Check if the first thumbnail is completely off-screen
        const firstThumbnail = columnThumbnails[0];
        if (firstThumbnail.top < -firstThumbnail.height) {
          // Get the last thumbnail
          const lastThumbnail = columnThumbnails[columnThumbnails.length - 1];
          
          // Move the first thumbnail to after the last one
          firstThumbnail.top = lastThumbnail.top + lastThumbnail.height;
          firstThumbnail.element.style.top = `${firstThumbnail.top}px`;
          
          // Reorder the array (move first to last)
          columnThumbnails.push(columnThumbnails.shift());
        }
      } else {
        // Move all thumbnails down
        columnThumbnails.forEach(thumbnail => {
          thumbnail.top += movement;
          thumbnail.element.style.top = `${thumbnail.top}px`;
        });
        
        // Check if the last thumbnail is completely off-screen at the bottom
        const lastThumbnail = columnThumbnails[columnThumbnails.length - 1];
        const viewportHeight = window.innerHeight;
        
        if (lastThumbnail.top > viewportHeight) {
          // Get the first thumbnail
          const firstThumbnail = columnThumbnails[0];
          
          // Move the last thumbnail to before the first one
          lastThumbnail.top = firstThumbnail.top - lastThumbnail.height;
          lastThumbnail.element.style.top = `${lastThumbnail.top}px`;
          
          // Reorder the array (move last to first)
          columnThumbnails.unshift(columnThumbnails.pop());
        }
      }
    }
  }
  
  // Method to handle window resize
  handleResize() {
    // Stop animation while we recalculate
    this.stopAnimation();
    
    // Recalculate and reposition thumbnails
    this.initializePositions();
    
    // Restart animation
    this.startAnimation();
  }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ThumbnailCarousel;
}

// Initialize carousels when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  // Check if containers exist before initializing
  if (document.getElementById('left-carousel') && document.getElementById('right-carousel')) {
    // Create a single global instance
    if (!window.thumbnailCarousel) {
      window.thumbnailCarousel = new ThumbnailCarousel();
      
      // Add resize handler
      window.addEventListener('resize', () => {
        if (window.thumbnailCarousel) {
          window.thumbnailCarousel.handleResize();
        }
      });
      
      // Add page visibility handler to pause animation when tab is not visible
      document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
          if (window.thumbnailCarousel) {
            window.thumbnailCarousel.stopAnimation();
          }
        } else {
          if (window.thumbnailCarousel) {
            window.thumbnailCarousel.startAnimation();
          }
        }
      });
      
      // Add scroll handler to keep carousel in view
      window.addEventListener('scroll', () => {
        if (window.thumbnailCarousel) {
          const scrollTop = window.scrollY || document.documentElement.scrollTop;
          const leftCarousel = document.getElementById('left-carousel');
          const rightCarousel = document.getElementById('right-carousel');
          
          if (leftCarousel) {
            leftCarousel.style.top = `${scrollTop}px`;
          }
          
          if (rightCarousel) {
            rightCarousel.style.top = `${scrollTop}px`;
          }
        }
      });
    }
  }
}); 