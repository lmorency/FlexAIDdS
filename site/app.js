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

  /* --- Scroll-Triggered Reveal Animations --- */
  const revealElements = document.querySelectorAll('.reveal');
  const prefersReducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (!prefersReducedMotion && revealElements.length > 0) {
    const revealObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('revealed');
          revealObserver.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });
    revealElements.forEach(el => revealObserver.observe(el));
  } else {
    revealElements.forEach(el => el.classList.add('revealed'));
  }

  /* --- Dynamic GitHub Stats --- */
  function fetchGitHubStats() {
    fetch('https://api.github.com/repos/lmorency/FlexAIDdS')
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(data => {
        var starsEl = document.getElementById('stat-stars');
        if (starsEl && typeof data.stargazers_count === 'number') {
          starsEl.textContent = data.stargazers_count;
          starsEl.setAttribute('data-count', data.stargazers_count);
        }
      })
      .catch(function() {
        var starsEl = document.getElementById('stat-stars');
        if (starsEl) starsEl.textContent = '\u2014';
      });
  }
  fetchGitHubStats();

  /* --- Hero Background: Mol* "Drug of the Day" --- */

  var drugOfTheDay = [
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
    { pdb: '1tup', drug: 'DNA fragment',    indication: 'p53 tumor suppressor\u2013DNA complex' },
    { pdb: '4hhb', drug: 'Oxygen (O\u2082)',indication: 'Hemoglobin oxygen transport' },
    { pdb: '1mbn', drug: 'Oxygen (O\u2082)',indication: 'Myoglobin oxygen storage' },
    { pdb: '1lyz', drug: 'NAG trimer',      indication: 'Lysozyme substrate binding' },
    { pdb: '1brs', drug: 'Barstar',         indication: 'Barnase\u2013Barstar protein interaction' },
    { pdb: '3pth', drug: 'Phosphoramidon',  indication: 'Thermolysin metalloprotease inhibitor' },
    { pdb: '1crn', drug: 'Crambin',         indication: 'Plant seed protein (docking benchmark)' },
    { pdb: '1bna', drug: 'B-DNA',           indication: 'Canonical DNA dodecamer' },
    { pdb: '1gpn', drug: 'GPN tripeptide',  indication: 'Loop conformation benchmark' },
    { pdb: '1a2b', drug: 'Deoxy-Hb',        indication: 'T-state hemoglobin (allosteric)' },
  ];

  function getTodaysDrug() {
    var now = new Date();
    var dayOfYear = Math.floor((now - new Date(now.getFullYear(),0,0)) / 86400000);
    return drugOfTheDay[dayOfYear % drugOfTheDay.length];
  }

  function applyBindingModeRepresentations(viewer) {
    try {
      var plugin = viewer.plugin;
      var structures = plugin.managers.structure.hierarchy.current.structures;
      if (!structures || structures.length === 0) return;

      var struct = structures[0];

      // Clear existing representations
      plugin.managers.structure.component.clear(struct).then(function() {
        // Add polymer as cartoon
        plugin.managers.structure.component.add(
          { structure: struct },
          { type: { name: 'static', params: 'polymer' } }
        ).then(function(polymerComp) {
          if (polymerComp) {
            plugin.managers.structure.representation.addRepresentation(
              polymerComp,
              { type: 'cartoon', color: 'chain-id' }
            );
          }
        });

        // Add ligand as ball-and-stick
        plugin.managers.structure.component.add(
          { structure: struct },
          { type: { name: 'static', params: 'ligand' } }
        ).then(function(ligandComp) {
          if (ligandComp) {
            plugin.managers.structure.representation.addRepresentation(
              ligandComp,
              { type: 'ball-and-stick', color: 'element-symbol' }
            );
          }
        });
      });
    } catch(e) {
      // Representation customization failed silently
    }
  }

  function initMolstar() {
    if (typeof molstar === 'undefined') return;

    var viewerEl = document.getElementById('molstar-viewer');
    if (!viewerEl) return;

    var drug = getTodaysDrug();

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
    }).then(function(viewer) {
      viewer.loadPdb(drug.pdb).then(function() {
        // Wait for structure to load, then apply binding mode representations
        setTimeout(function() {
          applyBindingModeRepresentations(viewer);
        }, 1500);

        // Enable auto-rotate
        setTimeout(function() {
          try {
            if (viewer.plugin.canvas3d) {
              viewer.plugin.canvas3d.setProps({
                trackball: { animate: { name: 'spin', params: { speed: 0.5 } } }
              });
            }
          } catch(e) {}
        }, 3000);
      });
    }).catch(function() { /* Mol* init failed silently */ });

    // Show drug of the day label
    var drugLabel = document.getElementById('drug-of-day-label');
    if (drugLabel) {
      drugLabel.innerHTML = '<strong>' + drug.drug + '</strong> <span class="drug-indication">' + drug.indication + '</span>';
    }
  }

  if (document.readyState === 'complete') {
    setTimeout(initMolstar, 300);
  } else {
    window.addEventListener('load', function() { setTimeout(initMolstar, 300); });
  }

  /* --- Smooth Scroll for Nav Links --- */
  document.querySelectorAll('a[href^="#"]').forEach(function(link) {
    link.addEventListener('click', function(e) {
      var href = link.getAttribute('href');
      if (href === '#') return;
      var target = document.querySelector(href);
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

})();
