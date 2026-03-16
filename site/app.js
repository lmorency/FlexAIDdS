/* ==========================================================================
   FlexAID∆S — App Logic (Enhanced)
   ========================================================================== */

(function() {
  'use strict';

  /* --- Theme Toggle --- */
  const toggle = document.querySelector('[data-theme-toggle]');
  const root = document.documentElement;

  let theme = matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  root.setAttribute('data-theme', theme);

  if (toggle) {
    toggle.addEventListener('click', () => {
      theme = theme === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', theme);
      toggle.setAttribute('aria-label', 'Switch to ' + (theme === 'dark' ? 'light' : 'dark') + ' mode');
    });
  }

  /* --- Mobile Menu --- */
  const menuToggle = document.querySelector('.mobile-menu-toggle');
  const mainNav = document.querySelector('.main-nav');

  if (menuToggle && mainNav) {
    menuToggle.addEventListener('click', () => {
      const isOpen = menuToggle.getAttribute('aria-expanded') === 'true';
      menuToggle.setAttribute('aria-expanded', !isOpen);
      mainNav.classList.toggle('open');
    });
    mainNav.querySelectorAll('a').forEach(a => {
      a.addEventListener('click', () => {
        menuToggle.setAttribute('aria-expanded', 'false');
        mainNav.classList.remove('open');
      });
    });
  }

  /* --- Header Scroll + Scroll Progress --- */
  const header = document.getElementById('site-header');
  const scrollProgress = document.querySelector('.scroll-progress');

  window.addEventListener('scroll', () => {
    header.classList.toggle('scrolled', window.scrollY > 60);

    // Update scroll progress bar
    if (scrollProgress) {
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      const progress = docHeight > 0 ? Math.min(window.scrollY / docHeight, 1) : 0;
      scrollProgress.style.transform = 'scaleX(' + progress + ')';
    }
  }, { passive: true });

  /* --- Back to Top --- */
  const backToTop = document.querySelector('.back-to-top');
  if (backToTop) {
    window.addEventListener('scroll', () => {
      backToTop.classList.toggle('visible', window.scrollY > 400);
    }, { passive: true });
    backToTop.addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  /* --- Tabs --- */
  const tabs = document.querySelectorAll('.tab-btn');
  const panels = document.querySelectorAll('.tab-panel');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => { t.classList.remove('active'); t.setAttribute('aria-selected', 'false'); });
      panels.forEach(p => p.classList.add('hidden'));
      tab.classList.add('active');
      tab.setAttribute('aria-selected', 'true');
      const panel = document.getElementById(tab.getAttribute('aria-controls'));
      if (panel) panel.classList.remove('hidden');
    });
  });

  /* --- Copy Buttons --- */
  document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const text = btn.getAttribute('data-copy');
      if (!text) return;
      navigator.clipboard.writeText(text).then(() => {
        btn.classList.add('copied');
        const original = btn.innerHTML;
        btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>';
        setTimeout(() => { btn.classList.remove('copied'); btn.innerHTML = original; }, 2000);
      }).catch(() => {});
    });
  });

  /* --- Number Count Up --- */
  const countEls = document.querySelectorAll('[data-count]');
  const countObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const el = entry.target;
        const target = parseInt(el.getAttribute('data-count'), 10);
        if (isNaN(target)) return;
        animateCount(el, 0, target, 800);
        countObserver.unobserve(el);
      }
    });
  }, { threshold: 0.3 });
  countEls.forEach(el => countObserver.observe(el));

  function animateCount(el, start, end, duration) {
    const startTime = performance.now();
    function step(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.floor(start + (end - start) * eased);
      if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  /* --- Hero Stat Counter Animation --- */
  function animateHeroStats() {
    const heroStats = document.querySelectorAll('.hero-stat-value');
    heroStats.forEach(el => {
      const text = el.textContent.trim();
      const hasPercent = text.endsWith('%');
      const num = parseFloat(text);
      if (isNaN(num)) return;

      const decimals = text.includes('.') ? (text.replace('%', '').split('.')[1] || '').length : 0;
      const startTime = performance.now();
      const duration = 1200;

      function step(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = eased * num;
        el.textContent = current.toFixed(decimals) + (hasPercent ? '%' : '');
        if (progress < 1) requestAnimationFrame(step);
      }
      el.textContent = (0).toFixed(decimals) + (hasPercent ? '%' : '');
      requestAnimationFrame(step);
    });
  }

  // Trigger hero stat animation when hero is visible
  const heroSection = document.querySelector('.hero');
  if (heroSection) {
    const heroObserver = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting) {
        animateHeroStats();
        heroObserver.disconnect();
      }
    }, { threshold: 0.3 });
    heroObserver.observe(heroSection);
  }

  /* --- Hero Background: Mol* "Drug of the Day" --- */

  // Curated set of iconic drug–target complexes
  // Each rotates daily as the hero background, showcasing a famous drug
  const drugOfTheDay = [
    { pdb: '1hsg', drug: 'Indinavir',      indication: 'HIV protease inhibitor' },
    { pdb: '3ert', drug: 'Tamoxifen',       indication: 'Estrogen receptor antagonist' },
    { pdb: '1iep', drug: 'Imatinib',        indication: 'BCR-Abl kinase inhibitor (CML)' },
    { pdb: '1m17', drug: 'Erlotinib',       indication: 'EGFR inhibitor (lung cancer)' },
    { pdb: '3nss', drug: 'Oseltamivir',     indication: 'Neuraminidase inhibitor (influenza)' },
    { pdb: '6lu7', drug: 'N3 Inhibitor',    indication: 'SARS-CoV-2 main protease' },
    { pdb: '4cox', drug: 'Celecoxib',       indication: 'COX-2 selective NSAID' },
    { pdb: '1hwi', drug: 'Donepezil',       indication: 'Acetylcholinesterase inhibitor (Alzheimer\'s)' },
    { pdb: '2rh1', drug: 'Carazolol',       indication: 'Beta-2 adrenergic antagonist' },
    { pdb: '3htb', drug: 'Dabigatran',      indication: 'Thrombin inhibitor (anticoagulant)' },
    { pdb: '2src', drug: 'Dasatinib',       indication: 'Src/Abl kinase inhibitor (CML)' },
    { pdb: '3eml', drug: 'Crizotinib',      indication: 'ALK inhibitor (lung cancer)' },
    { pdb: '4dkl', drug: 'Sorafenib',       indication: 'RAF kinase inhibitor (liver/kidney cancer)' },
    { pdb: '1fin', drug: 'ATP analog',      indication: 'CDK2-Cyclin A complex' },
    { pdb: '2f4j', drug: 'SB-203580',       indication: 'p38 MAP kinase inhibitor' },
    { pdb: '1n8z', drug: 'Insulin mimic',   indication: 'Insulin receptor kinase activator' },
    { pdb: '3kf4', drug: 'Ceftaroline',     indication: 'Penicillin-binding protein (MRSA)' },
    { pdb: '4lde', drug: 'Oxamate',         indication: 'Lactate dehydrogenase inhibitor' },
    { pdb: '1g9v', drug: 'Glutathione',     indication: 'GST conjugation substrate' },
    { pdb: '2pgh', drug: 'Flurbiprofen',    indication: 'COX-1/2 NSAID (inflammation)' },
    { pdb: '1cbs', drug: 'Retinoic acid',   indication: 'Cellular retinoic acid binding' },
    { pdb: '1tup', drug: 'DNA fragment',    indication: 'p53 tumor suppressor–DNA complex' },
    { pdb: '4hhb', drug: 'Oxygen (O₂)',     indication: 'Hemoglobin oxygen transport' },
    { pdb: '1mbn', drug: 'Oxygen (O₂)',     indication: 'Myoglobin oxygen storage' },
    { pdb: '1lyz', drug: 'NAG trimer',      indication: 'Lysozyme substrate binding' },
    { pdb: '1brs', drug: 'Barstar',         indication: 'Barnase–Barstar protein interaction' },
    { pdb: '3pth', drug: 'Phosphoramidon',  indication: 'Thermolysin metalloprotease inhibitor' },
    { pdb: '1crn', drug: 'Crambin',         indication: 'Plant seed protein (docking benchmark)' },
    { pdb: '1bna', drug: 'B-DNA',           indication: 'Canonical DNA dodecamer' },
    { pdb: '1gpn', drug: 'GPN tripeptide',  indication: 'Loop conformation benchmark' },
    { pdb: '1a2b', drug: 'Deoxy-Hb',        indication: 'T-state hemoglobin (allosteric)' },
  ];

  function getTodaysDrug() {
    const now = new Date();
    const dayOfYear = Math.floor((now - new Date(now.getFullYear(),0,0)) / 86400000);
    return drugOfTheDay[dayOfYear % drugOfTheDay.length];
  }

  /* --- Mol* BindingMode Representations --- */
  function applyBindingModeRepresentations(viewer) {
    try {
      const plugin = viewer.plugin;
      const structures = plugin.managers.structure.hierarchy.current.structures;
      if (!structures.length) return;

      const struct = structures[0];
      const components = struct.components;

      // Remove all default representations
      components.forEach(function(comp) {
        comp.representations.forEach(function(repr) {
          plugin.managers.structure.representation.remove(repr);
        });
      });

      // Add cartoon for polymer (protein/nucleic)
      var polymerSel = { name: 'static', params: 'polymer' };
      plugin.managers.structure.component.add(
        { key: 'polymer-cartoon', ref: struct.cell.transform.ref },
        polymerSel
      ).then(function(polyComp) {
        if (polyComp) {
          plugin.managers.structure.representation.addRepresentation(polyComp, {
            type: 'cartoon',
            color: 'chain-id'
          });
        }
      });

      // Add ball-and-stick for ligand
      var ligandSel = { name: 'static', params: 'ligand' };
      plugin.managers.structure.component.add(
        { key: 'ligand-sticks', ref: struct.cell.transform.ref },
        ligandSel
      ).then(function(ligComp) {
        if (ligComp) {
          plugin.managers.structure.representation.addRepresentation(ligComp, {
            type: 'ball-and-stick',
            color: 'element-symbol',
            size: 'physical'
          });
        }
      });
    } catch(e) {
      // Mol* API may differ across versions — fall back to defaults silently
    }
  }

  function initMolstar() {
    if (typeof molstar === 'undefined') return;

    const viewerEl = document.getElementById('molstar-viewer');
    if (!viewerEl) return;

    const drug = getTodaysDrug();

    molstar.Viewer.create('molstar-viewer', {
      layoutIsExpanded: false,
      layoutShowControls: false,
      layoutShowRemoteState: false,
      layoutShowSequence: false,
      layoutShowLog: false,
      layoutShowLeftPanel: false,
      viewportShowExpand: false,
      viewportShowSelectionMode: false,
      viewportShowAnimation: false,
      viewportShowControls: false,
      pdbProvider: 'rcsb',
      canvas3d: {
        transparentBackground: true,
        renderer: { backgroundColor: 0x000000, backgroundAlpha: 0 },
      },
    }).then(viewer => {
      viewer.loadPdb(drug.pdb).then(() => {
        // Apply BindingMode representations (cartoon + sticks)
        setTimeout(() => applyBindingModeRepresentations(viewer), 1500);
      });

      // Enable auto-rotate once structure settles
      setTimeout(() => {
        try {
          if (viewer.plugin.canvas3d) {
            viewer.plugin.canvas3d.setProps({
              trackball: { animate: { name: 'spin', params: { speed: 0.5 } } }
            });
          }
        } catch(e) { /* WebGL may not be available in headless environments */ }
      }, 3500);
    }).catch(() => { /* Mol* init failed silently — hero stays clean */ });

    // Show drug of the day label
    const drugLabel = document.getElementById('drug-of-day-label');
    if (drugLabel) {
      drugLabel.innerHTML = '<strong>' + drug.drug + '</strong> <span class="drug-indication">' + drug.indication + '</span>';
    }
  }

  if (document.readyState === 'complete') {
    setTimeout(initMolstar, 300);
  } else {
    window.addEventListener('load', () => setTimeout(initMolstar, 300));
  }

  /* --- Smooth Scroll for Nav Links --- */
  document.querySelectorAll('a[href^="#"]').forEach(link => {
    link.addEventListener('click', (e) => {
      const href = link.getAttribute('href');
      if (href === '#') return;
      const target = document.querySelector(href);
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  /* --- GitHub Stats (enhanced) --- */
  function fetchGitHubStats() {
    const starEl = document.getElementById('stat-stars');
    const commitEl = document.querySelector('[data-count]');

    fetch('https://api.github.com/repos/lmorency/FlexAIDdS')
      .then(r => r.json())
      .then(data => {
        if (starEl && data.stargazers_count !== undefined) {
          starEl.textContent = data.stargazers_count;
        }
        // Update commit count if available via size heuristic
        if (commitEl && data.size) {
          // Keep the hardcoded count — GitHub API doesn't expose commit count directly
        }
      })
      .catch(() => {
        if (starEl) starEl.textContent = '—';
      });
  }
  fetchGitHubStats();

  /* --- Reveal on Scroll (sections) --- */
  const revealSections = document.querySelectorAll('main > section');
  if (revealSections.length && 'IntersectionObserver' in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('revealed');
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.08 });
    revealSections.forEach(s => {
      s.classList.add('reveal');
      io.observe(s);
    });
  }

  /* --- Staggered Card Animations --- */
  const cardSelectors = '.feature-card, .stat-card, .pub-card, .module-card, .cns-stat-card, .install-card, .pymol-install-card, .pymol-commands-card, .license-card, .arch-step, .arch-sub-card';
  const cardContainers = document.querySelectorAll('.features-grid, .stats-grid, .publications-grid, .modules-grid, .cns-stats-grid, .install-grid, .pymol-grid, .contributing-grid, .arch-pipeline, .arch-sub-row');

  if (cardContainers.length && 'IntersectionObserver' in window) {
    const cardIO = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const cards = entry.target.querySelectorAll(cardSelectors);
          cards.forEach((card, i) => {
            card.style.setProperty('--card-delay', (i * 80) + 'ms');
            card.classList.add('reveal-card');
            // Trigger reflow then add revealed state
            requestAnimationFrame(() => {
              requestAnimationFrame(() => {
                card.classList.add('revealed-card');
              });
            });
          });
          cardIO.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1 });
    cardContainers.forEach(c => cardIO.observe(c));
  }

})();
