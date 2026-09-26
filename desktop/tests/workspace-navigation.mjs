// Exercise the visible navigation at either viewport size. The domain select
// remains available on narrow screens; desktop exposes named domain buttons.
export async function selectObservatory(page, value) {
  await page.locator('.app[data-active-observatory]').waitFor();
  const names = { companies: 'Companies', sectors: 'Industries', macro: 'Macro' };
  const button = page.getByRole('navigation', { name: 'Main navigation', exact: true }).getByRole('button', { name: names[value], exact: true });
  if (await button.isVisible()) await button.click();
  else await page.getByLabel('Observatory', { exact: true }).selectOption(value);
  await page.locator(`[data-active-observatory="${value}"]`).waitFor();
}

export async function openListPanel(page, name) {
  const labels = { filters: 'Show list filters', columns: 'Show column sets', tools: 'Show list tools' };
  const toggle = page.getByRole('button', { name: labels[name], exact: true });
  if (await toggle.getAttribute('aria-expanded') !== 'true') await toggle.click();
}

export async function closeListPanel(page, name) {
  const labels = { filters: 'Show list filters', columns: 'Show column sets', tools: 'Show list tools' };
  const toggle = page.getByRole('button', { name: labels[name], exact: true });
  if (await toggle.getAttribute('aria-expanded') === 'true') await toggle.click();
}

export async function openCompanyFinancials(page) {
  await page.getByLabel('Company financials', { exact: true }).click();
  await page.locator('[data-business-view="financials"]').waitFor();
}

export async function openValuationModel(page) {
  const summary = page.getByText('Model, sources & historical baseline', { exact: true });
  if (await summary.locator('..').getAttribute('open') === null) await summary.click();
}

export async function openValuationPayback(page) {
  for (const name of ['Low', 'Mid', 'High']) {
    const summary = page.getByText(`${name} payback & breakdown`, { exact: true });
    if (await summary.count() && await summary.locator('..').getAttribute('open') === null) await summary.click();
  }
}

export async function openFinancialCoverage(page) {
  const summary = page.getByText('Report coverage & source dates', { exact: true });
  if (!await summary.count()) return;
  if (await summary.locator('..').getAttribute('open') === null) await summary.click();
}
