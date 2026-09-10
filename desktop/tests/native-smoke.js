(async () => {
  const checks = [];
  const waitFor = async (predicate, label) => {
    const deadline = Date.now() + 15000;
    while (!predicate()) {
      if (Date.now() > deadline) throw new Error(`Timed out: ${label}`);
      await new Promise(resolve => setTimeout(resolve, 80));
    }
    checks.push(label);
  };
  const click = selector => {
    const element = document.querySelector(selector);
    if (!element) throw new Error(`Missing control: ${selector}`);
    element.click();
  };
  const select = (label, value) => {
    const element = document.querySelector(`select[aria-label="${label}"]`);
    if (!element) throw new Error(`Missing select: ${label}`);
    element.value = value;
    element.dispatchEvent(new Event('change', { bubbles: true }));
  };
  if (!window.__TAURI_INTERNALS__) throw new Error('Native Tauri bridge is missing');
  const start = performance.now();
  await waitFor(() => document.querySelector('[data-country="SE"][data-ready="true"]'), 'Sweden data');
  await waitFor(() => document.querySelector('[data-map-ready="true"]'), 'Native WebGL map');
  if (document.querySelector('.map-error')) throw new Error('Map failed');
  await waitFor(() => document.querySelectorAll('.echart canvas').length > 0 || document.querySelectorAll('canvas').length > 2, 'Native charts');
  const readyMs = Math.round(performance.now() - start);
  select('Comparison country', 'DE');
  await waitFor(() => document.querySelector('.chart-key')?.textContent.includes('Germany'), 'Country comparison');
  click('[aria-label="History & outlook"]');
  await waitFor(() => document.querySelector('select[aria-label="Map historical indicator"]'), 'History controls');
  select('Map historical indicator', 'gov_debt_pct_gdp');
  await waitFor(() => document.querySelector('.metric-readout')?.textContent.includes('Historical observation'), 'Historical map and observation');
  click('[aria-label="Trade connections"]');
  await waitFor(() => document.querySelectorAll('.trade-partners button').length === 6, 'Trade chart');
  if (document.querySelector('.trade-partners')?.textContent.includes('Euro area')) throw new Error('Overlapping trade aggregate');
  click('[aria-label="Fundamentals"]');
  click('[aria-label="Open data library"]');
  await waitFor(() => document.querySelector('[role="dialog"]'), 'Data library');
  document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
  await waitFor(() => !document.querySelector('[role="dialog"]'), 'Close library');
  const notices = await fetch('/THIRD-PARTY-NOTICES.txt').then(response => response.text());
  if (!notices.includes('MACRO ATLAS')) throw new Error('Packaged notices are missing');
  checks.push('Packaged notices');
  return { status: 'PASS', checks, readyMs, origin: location.origin, userAgent: navigator.userAgent, nativeBridge: true };
})()
