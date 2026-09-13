import { configDefaults, defineConfig, mergeConfig } from 'vitest/config';
import viteConfig from './vite.config';

// This research suite uses Node's test runner; execute it with test:backtest.
export default mergeConfig(viteConfig, defineConfig({ test: {
  exclude: [...configDefaults.exclude, 'tests/cash-flow-backtest.test.mjs'],
} }));
