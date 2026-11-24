# Macro Atlas desktop

Work from this directory for the offline Windows app. Check `package.json` for the app version and available commands. Use the matching release section in [README.md](README.md) for intended behavior. Consult [HANDOFF.md](HANDOFF.md) when continuing a checkpoint or checking what was installed and tested. Match its version and date to the task.

- Edit the React UI, financial calculations and their unit tests in `src/`. For changes to forecasts or saved inputs, keep source observations distinguishable from user assumptions and check that existing saved drafts still open correctly.
- Trace native data access from `src-tauri/src/main.rs` into `financial-store/` or `research-store/`. For a bundled data change, start with the relevant exporter in `scripts/` and its source inputs. Preserve offline operation and the data’s source and date.
- Run `npm test` here for the usual unit check. For changes to an exporter, a Rust store, a browser interaction or Windows packaging, use the corresponding checks in the [README verification section](README.md#verification). Report Windows installation or launch only when verified on Windows.
- For changes affecting the historical cash-flow backtest in `scripts/cash-flow-backtest.mjs`, run `npm run test:backtest`. It checks forecast calculations, whether training and calibration use only data available at the time, and the backtest’s SQLite output.
