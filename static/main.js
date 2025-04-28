document.addEventListener('DOMContentLoaded', () => {
  // Initialize Particles.js
  particlesJS('particles-js', {
    particles: {
      number: { value: 80, density: { enable: true, value_area: 800 } },
      color: { value: '#ffcc00' },
      shape: { type: 'circle' },
      opacity: { value: 0.5, random: true },
      size: { value: 3, random: true },
      line_linked: { enable: true, distance: 150, color: '#ff6f61', opacity: 0.4, width: 1 },
      move: { enable: true, speed: 6, direction: 'none', random: false }
    },
    interactivity: {
      detect_on: 'canvas',
      events: { onhover: { enable: true, mode: 'repulse' }, onclick: { enable: true, mode: 'push' } },
      modes: { repulse: { distance: 100 }, push: { particles_nb: 4 } }
    },
    retina_detect: true
  });

  // Splash Screen Animation
  gsap.to('#app-title', { scale: 1.1, duration: 1.5, yoyo: true, repeat: -1 });
  setTimeout(() => {
    gsap.to('#splash-screen', {
      opacity: 0,
      duration: 1,
      onComplete: () => {
        document.querySelector('#splash-screen').style.display = 'none';
        document.querySelector('#app-container').classList.remove('hidden');
        gsap.from('#app-container', { opacity: 0, y: 50, duration: 1 });
        showTutorial();
      }
    });
  }, 3000);

  // Elements
  const filterCarousel = document.querySelector('#filters');
  const videoFeed = document.querySelector('#video-feed');
  const snapBtn = document.querySelector('#snap-btn');
  const videoBtn = document.querySelector('#video-btn');
  const boomerangBtn = document.querySelector('#boomerang-btn');
  const snapsContainer = document.querySelector('#snaps');
  const themeToggle = document.querySelector('#theme-toggle');
  const flashEffect = document.querySelector('#flash-effect');
  const editorModal = document.querySelector('#editor-modal');
  const editorCanvas = document.querySelector('#editor-canvas');
  const drawTool = document.querySelector('#draw-tool');
  const stickerTool = document.querySelector('#sticker-tool');
  const textTool = document.querySelector('#text-tool');
  const eraseStickerTool = document.querySelector('#erase-sticker-tool');
  const drawColor = document.querySelector('#draw-color');
  const stickerSelect = document.querySelector('#sticker-select');
  const textInput = document.querySelector('#text-input');
  const editorSave = document.querySelector('#editor-save');
  const editorCancel = document.querySelector('#editor-cancel');
  const viewAllSnaps = document.querySelector('#view-all-snaps');
  const gallery = document.querySelector('#gallery');
  const tutorialModal = document.querySelector('#tutorial-modal');
  const tutorialPrev = document.querySelector('#tutorial-prev');
  const tutorialNext = document.querySelector('#tutorial-next');
  const tutorialSkip = document.querySelector('#tutorial-skip');

  // Filter Management
  let filters = [];
  let currentFilter = '';

  const loadFilters = async () => {
    try {
      const response = await fetch('/filters');
      const data = await response.json();
      filters = data.filters;
      currentFilter = data.current;

      filterCarousel.innerHTML = '';
      filters.forEach((filter) => {
        const item = document.createElement('div');
        item.className = 'filter-item';
        item.innerHTML = `
          <button class="filter-btn${filter === currentFilter ? ' active' : ''}" data-filter="${filter}">
            ${filter}
          </button>
        `;
        filterCarousel.appendChild(item);
      });

      gsap.from('.filter-btn', { opacity: 0, x: 20, stagger: 0.1, duration: 0.5 });
    } catch (error) {
      console.error('Error loading filters:', error);
      alert('Failed to load filters. Please check the server.');
    }
  };

  const setFilter = async (name, button) => {
    try {
      gsap.to(videoFeed, {
        opacity: 0,
        duration: 0.3,
        onComplete: async () => {
          await fetch(`/set_filter/${name}`, { method: 'POST' });
          gsap.to(videoFeed, { opacity: 1, duration: 0.3 });
        }
      });

      document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.filter === name);
      });
      currentFilter = name;

      if (button) {
        gsap.to(button, {
          scale: 1.2,
          duration: 0.2,
          yoyo: true,
          repeat: 1,
          onComplete: () => gsap.to(button, { scale: 1, duration: 0.2 })
        });
      }
    } catch (error) {
      console.error('Error setting filter:', error);
      alert('Failed to set filter. Please try again.');
    }
  };

  // Snap/Video Management
  let snapUrl = null;
  const takeSnap = async () => {
    const maxAttempts = 3;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        gsap.to(flashEffect, { opacity: 0.7, duration: 0.1, onComplete: () => gsap.to(flashEffect, { opacity: 0, duration: 0.3 }) });
        const response = await fetch('/save_snap', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({})
        });
        if (!response.ok) {
          const errorText = await response.text();
          console.error(`takeSnap attempt ${attempt} failed with status ${response.status}: ${errorText}`);
          throw new Error(`HTTP ${response.status}: ${errorText}`);
        }
        const data = await response.json();
        snapUrl = data.url;
        openEditor();
        return; // Success, exit loop
      } catch (error) {
        console.error(`Attempt ${attempt} to take snap failed:`, error);
        if (attempt === maxAttempts) {
          alert(`Failed to take snap after ${maxAttempts} attempts: ${error.message}`);
        }
        await new Promise(resolve => setTimeout(resolve, 500)); // Wait before retry
      }
    }
  };

  const recordVideo = async () => {
    try {
      videoBtn.disabled = true;
      gsap.to(flashEffect, { opacity: 0.3, duration: 0.1, repeat: 5, yoyo: true });
      const response = await fetch('/record_video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ duration: 5 })
      });
      if (!response.ok) throw new Error('Failed to record video');
      const data = await response.json();
      addSnap(data.url, true, data.type);
    } catch (error) {
      console.error('Error recording video:', error);
      alert('Failed to record video. Please try again.');
    } finally {
      videoBtn.disabled = false;
    }
  };

  const recordBoomerang = async () => {
    try {
      boomerangBtn.disabled = true;
      gsap.to(flashEffect, { opacity: 0.3, duration: 0.1, repeat: 3, yoyo: true });
      const response = await fetch('/record_boomerang', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      if (!response.ok) throw new Error('Failed to record boomerang');
      const data = await response.json();
      addSnap(data.url, true, data.type);
    } catch (error) {
      console.error('Error recording boomerang:', error);
      alert('Failed to record boomerang. Please try again.');
    } finally {
      boomerangBtn.disabled = false;
    }
  };

  const loadSnaps = async () => {
    try {
      const response = await fetch('/list_snaps');
      if (!response.ok) throw new Error('Failed to load snaps');
      const data = await response.json();
      snapsContainer.innerHTML = ''; // Clear existing snaps
      const snapsToShow = gallery.classList.contains('full-screen') ? data.snaps : data.snaps.slice(0, 4);
      console.log(`Loading ${snapsToShow.length} snaps (full-screen: ${gallery.classList.contains('full-screen')})`);
      snapsToShow.forEach(snap => addSnap(snap.url, false, snap.type));
    } catch (error) {
      console.error('Error loading snaps:', error);
      alert('Failed to load snaps. Please try again.');
    }
  };

  const addSnap = (url, animate = true, type = 'image') => {
    const snap = document.createElement('div');
    snap.className = 'snap-item';
    snap.innerHTML = type === 'image' ?
      `<img src="${url}" alt="Snap" />` :
      `<video src="${url}" controls loop muted></video>`;
    snap.innerHTML += `<button class="download-btn" data-url="${url}">⬇️</button>`;
    snapsContainer.prepend(snap);

    if (animate) {
      gsap.from(snap, { opacity: 0, scale: 0.8, duration: 0.5 });
    }

    // Only limit to 4 snaps if not in full-screen mode
    if (!gallery.classList.contains('full-screen')) {
      const snaps = snapsContainer.querySelectorAll('.snap-item');
      if (snaps.length > 4) {
        const oldestSnap = snaps[snaps.length - 1];
        gsap.to(oldestSnap, {
          opacity: 0,
          scale: 0.8,
          duration: 0.3,
          onComplete: () => oldestSnap.remove()
        });
      }
    }
  };

  // Editor Logic
  let ctx, baseImage, isDrawing = false, lastX = 0, lastY = 0, currentTool = 'draw';
  let textObjects = [], stickerObjects = [], selectedText = null, isMovingText = false;

  const openEditor = () => {
    editorModal.classList.remove('hidden');
    const img = new Image();
    img.src = snapUrl;
    img.onload = () => {
      editorCanvas.width = img.width;
      editorCanvas.height = img.height;
      ctx = editorCanvas.getContext('2d');
      baseImage = img;
      textObjects = [];
      stickerObjects = [];
      selectedText = null;
      redrawCanvas();
      ctx.lineWidth = 5;
      ctx.lineCap = 'round';
      ctx.strokeStyle = drawColor.value;
      ctx.font = '30px Poppins';
    };
    drawTool.classList.add('active');
    stickerTool.classList.remove('active');
    textTool.classList.remove('active');
    eraseStickerTool.classList.remove('active');
    stickerSelect.classList.add('hidden');
    textInput.classList.add('hidden');
    currentTool = 'draw';
  };

  const redrawCanvas = () => {
    ctx.clearRect(0, 0, editorCanvas.width, editorCanvas.height);
    ctx.drawImage(baseImage, 0, 0);

    // Draw stickers
    stickerObjects.forEach(sticker => {
      ctx.font = '40px Arial';
      ctx.fillText(sticker.emoji, sticker.x, sticker.y);
    });

    // Draw text objects
    textObjects.forEach((text, index) => {
      ctx.font = '30px Poppins';
      ctx.fillStyle = text.color;
      ctx.fillText(text.content, text.x, text.y);
      if (index === selectedText) {
        ctx.strokeStyle = '#ff6f61';
        ctx.lineWidth = 1;
        ctx.strokeRect(text.x - 5, text.y - 30, ctx.measureText(text.content).width + 10, 35);
      }
    });
  };

  const startDrawing = (e) => {
    if (currentTool !== 'draw') return;
    isDrawing = true;
    const rect = editorCanvas.getBoundingClientRect();
    lastX = (e.clientX || e.touches[0].clientX) - rect.left;
    lastY = (e.clientY || e.touches[0].clientY) - rect.top;
  };

  const draw = (e) => {
    if (!isDrawing) return;
    e.preventDefault();
    const rect = editorCanvas.getBoundingClientRect();
    const x = (e.clientX || e.touches[0].clientX) - rect.left;
    const y = (e.clientY || e.touches[0].clientY) - rect.top;
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    lastX = x;
    lastY = y;
  };

  const stopDrawing = () => {
    isDrawing = false;
  };

  const addSticker = (e) => {
    if (currentTool !== 'sticker') return;
    const rect = editorCanvas.getBoundingClientRect();
    const x = (e.clientX || e.touches[0].clientX) - rect.left;
    const y = (e.clientY || e.touches[0].clientY) - rect.top;
    stickerObjects.push({ emoji: stickerSelect.value, x, y });
    redrawCanvas();
  };

  const eraseSticker = (e) => {
    if (currentTool !== 'erase-sticker') return;
    const rect = editorCanvas.getBoundingClientRect();
    const x = (e.clientX || e.touches[0].clientX) - rect.left;
    const y = (e.clientY || e.touches[0].clientY) - rect.top;
    stickerObjects = stickerObjects.filter(sticker => {
      const stickerWidth = ctx.measureText(sticker.emoji).width;
      const stickerHeight = 40;
      return !(x >= sticker.x && x <= sticker.x + stickerWidth && y >= sticker.y - stickerHeight && y <= sticker.y);
    });
    redrawCanvas();
  };

  const selectText = (e) => {
    if (currentTool !== 'text') return;
    const rect = editorCanvas.getBoundingClientRect();
    const x = (e.clientX || e.touches[0].clientX) - rect.left;
    const y = (e.clientY || e.touches[0].clientY) - rect.top;
    selectedText = null;
    textObjects.forEach((text, index) => {
      const textWidth = ctx.measureText(text.content).width;
      if (x >= text.x && x <= text.x + textWidth && y >= text.y - 30 && y <= text.y) {
        selectedText = index;
      }
    });
    redrawCanvas();
  };

  const startMovingText = (e) => {
    if (currentTool !== 'text' || selectedText === null) return;
    isMovingText = true;
    const rect = editorCanvas.getBoundingClientRect();
    lastX = (e.clientX || e.touches[0].clientX) - rect.left;
    lastY = (e.clientY || e.touches[0].clientY) - rect.top;
  };

  const moveText = (e) => {
    if (!isMovingText || selectedText === null) return;
    e.preventDefault();
    const rect = editorCanvas.getBoundingClientRect();
    const x = (e.clientX || e.touches[0].clientX) - rect.left;
    const y = (e.clientY || e.touches[0].clientY) - rect.top;
    textObjects[selectedText].x += x - lastX;
    textObjects[selectedText].y += y - lastY;
    lastX = x;
    lastY = y;
    redrawCanvas();
  };

  const stopMovingText = () => {
    isMovingText = false;
  };

  const addText = () => {
    if (currentTool !== 'text' || !textInput.value) return;
    textObjects.push({
      content: textInput.value,
      x: 50,
      y: 50,
      color: drawColor.value
    });
    textInput.value = '';
    selectedText = textObjects.length - 1;
    redrawCanvas();
  };

  const saveSnap = async () => {
    const maxAttempts = 3;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        const overlay = editorCanvas.toDataURL('image/png');
        const response = await fetch('/save_snap', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ overlay })
        });
        if (!response.ok) {
          const errorText = await response.text();
          console.error(`saveSnap attempt ${attempt} failed with status ${response.status}: ${errorText}`);
          throw new Error(`HTTP ${response.status}: ${errorText}`);
        }
        const data = await response.json();
        addSnap(data.url, true, data.type);
        editorModal.classList.add('hidden');
        return; // Success, exit loop
      } catch (error) {
        console.error(`Attempt ${attempt} to save snap failed:`, error);
        if (attempt === maxAttempts) {
          alert(`Failed to save edited snap after ${maxAttempts} attempts: ${error.message}`);
        }
        await new Promise(resolve => setTimeout(resolve, 500)); // Wait before retry
      }
    }
  };

  // Tutorial Logic
  let currentStep = 1;
  const showTutorial = () => {
    if (localStorage.getItem('tutorialSeen')) return;
    tutorialModal.classList.remove('hidden');
    updateTutorialStep();
  };

  const updateTutorialStep = () => {
    document.querySelectorAll('.tutorial-step').forEach(step => step.classList.add('hidden'));
    document.querySelector(`#tutorial-step-${currentStep}`).classList.remove('hidden');
    tutorialPrev.classList.toggle('hidden', currentStep === 1);
    tutorialNext.textContent = currentStep === 3 ? 'Finish' : 'Next';
  };

  // Event Listeners
  snapBtn.addEventListener('click', takeSnap);
  videoBtn.addEventListener('click', recordVideo);
  boomerangBtn.addEventListener('click', recordBoomerang);

  filterCarousel.addEventListener('click', (e) => {
    const btn = e.target.closest('.filter-btn');
    if (btn) {
      setFilter(btn.dataset.filter, btn);
    }
  });

  videoFeed.addEventListener('dblclick', () => {
    const currentIndex = filters.indexOf(currentFilter);
    const nextIndex = (currentIndex + 1) % filters.length;
    setFilter(filters[nextIndex]);
  });

  let startX = 0;
  filterCarousel.addEventListener('touchstart', (e) => {
    startX = e.touches[0].clientX;
  });

  filterCarousel.addEventListener('touchmove', (e) => {
    e.preventDefault();
    const moveX = e.touches[0].clientX;
    const diff = startX - moveX;
    filterCarousel.scrollLeft += diff;
    startX = moveX;
  });

  themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    themeToggle.textContent = document.body.classList.contains('dark-mode') ? '☀️' : '🌙';
  });

  viewAllSnaps.addEventListener('click', () => {
    gallery.classList.toggle('full-screen');
    viewAllSnaps.textContent = gallery.classList.contains('full-screen') ? 'Close Gallery' : 'View All Snaps';
    loadSnaps(); // Refresh gallery
  });

  snapsContainer.addEventListener('click', (e) => {
    const btn = e.target.closest('.download-btn');
    if (btn) {
      const link = document.createElement('a');
      link.href = btn.dataset.url;
      link.download = btn.dataset.url.split('/').pop();
      link.click();
    }
  });

  editorCanvas.addEventListener('mousedown', (e) => {
    startDrawing(e);
    selectText(e);
    startMovingText(e);
  });
  editorCanvas.addEventListener('mousemove', (e) => {
    draw(e);
    moveText(e);
  });
  editorCanvas.addEventListener('mouseup', () => {
    stopDrawing();
    stopMovingText();
  });
  editorCanvas.addEventListener('touchstart', (e) => {
    startDrawing(e);
    selectText(e);
    startMovingText(e);
  });
  editorCanvas.addEventListener('touchmove', (e) => {
    draw(e);
    moveText(e);
  });
  editorCanvas.addEventListener('touchend', () => {
    stopDrawing();
    stopMovingText();
  });
  editorCanvas.addEventListener('click', (e) => {
    addSticker(e);
    eraseSticker(e);
  });

  drawTool.addEventListener('click', () => {
    currentTool = 'draw';
    drawTool.classList.add('active');
    stickerTool.classList.remove('active');
    textTool.classList.remove('active');
    eraseStickerTool.classList.remove('active');
    stickerSelect.classList.add('hidden');
    textInput.classList.add('hidden');
  });

  stickerTool.addEventListener('click', () => {
    currentTool = 'sticker';
    drawTool.classList.remove('active');
    stickerTool.classList.add('active');
    textTool.classList.remove('active');
    eraseStickerTool.classList.remove('active');
    stickerSelect.classList.remove('hidden');
    textInput.classList.add('hidden');
  });

  textTool.addEventListener('click', () => {
    currentTool = 'text';
    drawTool.classList.remove('active');
    stickerTool.classList.remove('active');
    textTool.classList.add('active');
    eraseStickerTool.classList.remove('active');
    stickerSelect.classList.add('hidden');
    textInput.classList.remove('hidden');
  });

  eraseStickerTool.addEventListener('click', () => {
    currentTool = 'erase-sticker';
    drawTool.classList.remove('active');
    stickerTool.classList.remove('active');
    textTool.classList.remove('active');
    eraseStickerTool.classList.add('active');
    stickerSelect.classList.add('hidden');
    textInput.classList.add('hidden');
  });

  drawColor.addEventListener('input', () => {
    ctx.strokeStyle = drawColor.value;
    ctx.fillStyle = drawColor.value;
  });

  textInput.addEventListener('change', addText);
  editorSave.addEventListener('click', saveSnap);
  editorCancel.addEventListener('click', () => {
    editorModal.classList.add('hidden');
  });

  tutorialPrev.addEventListener('click', () => {
    if (currentStep > 1) {
      currentStep--;
      updateTutorialStep();
    }
  });

  tutorialNext.addEventListener('click', () => {
    if (currentStep < 3) {
      currentStep++;
      updateTutorialStep();
    } else {
      tutorialModal.classList.add('hidden');
      localStorage.setItem('tutorialSeen', 'true');
    }
  });

  tutorialSkip.addEventListener('click', () => {
    tutorialModal.classList.add('hidden');
    localStorage.setItem('tutorialSeen', 'true');
  });

  // Initialize
  loadFilters();
  loadSnaps();
});