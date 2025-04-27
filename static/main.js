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
      }
    });
  }, 3000);

  // Elements
  const filterCarousel = document.querySelector('#filters');
  const videoFeed = document.querySelector('#video-feed');
  const snapBtn = document.querySelector('#snap-btn');
  const snapsContainer = document.querySelector('#snaps');
  const togglePanel = document.querySelector('#toggle-panel');
  const themeToggle = document.querySelector('#theme-toggle');

  // Filter Management
  let filters = [];
  let currentFilter = '';

  const loadFilters = async () => {
    const response = await fetch('/filters');
    const data = await response.json();
    filters = data.filters;
    currentFilter = data.current;

    filterCarousel.innerHTML = '';
    filters.forEach((filter, index) => {
      const item = document.createElement('div');
      item.className = 'filter-item';
      item.innerHTML = `
        <button class="filter-btn${filter === currentFilter ? ' active' : ''}" data-filter="${filter}">
          ${filter}
        </button>
      `;
      filterCarousel.appendChild(item);
    });

    // Animate filter buttons on load
    gsap.from('.filter-btn', { opacity: 0, x: 20, stagger: 0.1, duration: 0.5 });
  };

  const setFilter = async (name, button) => {
    await fetch(`/set_filter/${name}`, { method: 'POST' });
    document.querySelectorAll('.filter-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.filter === name);
    });
    currentFilter = name;

    // Click animation
    if (button) {
      gsap.to(button, {
        scale: 1.2,
        duration: 0.2,
        yoyo: true,
        repeat: 1,
        onComplete: () => {
          gsap.to(button, { scale: 1, duration: 0.2 });
        }
      });
      // Particle burst effect
      const rect = button.getBoundingClientRect();
      particlesJS('particles-js', {
        particles: {
          number: { value: 20, density: { enable: false } },
          color: { value: '#ffcc00' },
          shape: { type: 'circle' },
          opacity: { value: 0.8, random: true },
          size: { value: 5, random: true },
          move: {
            enable: true,
            speed: 10,
            direction: 'none',
            random: true,
            out_mode: 'out'
          }
        },
        interactivity: { detect_on: 'canvas', events: { onhover: { enable: false }, onclick: { enable: false } } },
        retina_detect: true
      });
      setTimeout(() => {
        document.querySelector('#particles-js').innerHTML = ''; // Clear particles
      }, 1000);
    }
  };

  // Snap Management
  const takeSnap = async () => {
    try {
      const response = await fetch('/save_snap', { method: 'POST' });
      if (!response.ok) throw new Error('Failed to save snap');
      const data = await response.json();
      addSnap(data.url);
    } catch (error) {
      console.error('Error saving snap:', error);
      alert('Failed to save snap. Please try again.');
    }
  };

  const loadSnaps = async () => {
    try {
      const response = await fetch('/list_snaps');
      const data = await response.json();
      snapsContainer.innerHTML = '';
      // Display only the latest 4 snaps
      const latestSnaps = data.snaps.slice(0, 4);
      latestSnaps.forEach(url => addSnap(url, false));
    } catch (error) {
      console.error('Error loading snaps:', error);
    }
  };

  const addSnap = (url, animate = true) => {
    // Add new snap
    const snap = document.createElement('div');
    snap.className = 'snap-item';
    snap.innerHTML = `<img src="${url}" alt="Snap" />`;
    snapsContainer.prepend(snap);

    // Animate new snap if requested
    if (animate) {
      gsap.from(snap, { opacity: 0, scale: 0.8, duration: 0.5 });
    }

    // Remove oldest snap if more than 4
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
  };

  // Event Listeners
  snapBtn.addEventListener('click', takeSnap);

  filterCarousel.addEventListener('click', (e) => {
    const btn = e.target.closest('.filter-btn');
    if (btn) {
      setFilter(btn.dataset.filter, btn);
    }
  });

  // Double-Click to Cycle Filters
  videoFeed.addEventListener('dblclick', () => {
    const currentIndex = filters.indexOf(currentFilter);
    const nextIndex = (currentIndex + 1) % filters.length;
    setFilter(filters[nextIndex]);
  });

  // Touch Swipe for Filter Carousel
  let startX = 0;
  filterCarousel.addEventListener('touchstart', (e) => {
    startX = e.touches[0].clientX;
  });

  filterCarousel.addEventListener('touchmove', (e) => {
    const moveX = e.touches[0].clientX;
    const diff = startX - moveX;
    filterCarousel.scrollLeft += diff;
    startX = moveX;
  });

  // Theme Toggle
  themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    themeToggle.textContent = document.body.classList.contains('dark-mode') ? '☀️' : '🌙';
  });

  // Toggle Filter Panel
  togglePanel.addEventListener('click', () => {
    filterCarousel.classList.toggle('open');
  });

  // Initial Load
  loadFilters();
  loadSnaps();
});