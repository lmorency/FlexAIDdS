/* ==========================================================================
   FlexAID∆S — App Logic
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

  /* --- Header Scroll --- */
  const header = document.getElementById('site-header');
  window.addEventListener('scroll', () => {
    header.classList.toggle('scrolled', window.scrollY > 60);
  }, { passive: true });

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
      viewer.loadPdb(drug.pdb);

      // Enable auto-rotate once structure settles
      setTimeout(() => {
        try {
          if (viewer.plugin.canvas3d) {
            viewer.plugin.canvas3d.setProps({
              trackball: { animate: { name: 'spin', params: { speed: 0.5 } } }
            });
          }
        } catch(e) { /* WebGL may not be available in headless environments */ }
      }, 2500);
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

  /* --- GitHub Stars --- */
  const starEl = document.getElementById('stat-stars');
  if (starEl) {
    fetch('https://api.github.com/repos/LeBonhommePharma/FlexAIDdS')
      .then(r => r.json())
      .then(data => {
        if (data.stargazers_count !== undefined) {
          starEl.textContent = data.stargazers_count;
        }
      })
      .catch(() => { starEl.textContent = '—'; });
  }

  /* --- Reveal on Scroll --- */
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

})();
