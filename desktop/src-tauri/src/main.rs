#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::Manager;

#[tauri::command]
fn export_csv(app: tauri::AppHandle, filename: String, contents: String) -> Result<String, String> {
    if !filename.starts_with("Macro-Atlas-")
        || !filename.ends_with(".csv")
        || filename.len() > 160
        || filename.chars().any(|c| !(c.is_ascii_alphanumeric() || "-_.".contains(c)))
        || contents.len() > 8_000_000
    {
        return Err("Invalid export".into());
    }
    let folder = app.path().download_dir().map_err(|e| e.to_string())?;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).map_err(|e| e.to_string())?.as_millis();
    let path = folder.join(format!("{}-{}.csv", filename.trim_end_matches(".csv"), timestamp));
    // Exports always create a new file. Existing user files cannot be overwritten.
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new().write(true).create_new(true)
        .open(&path).map_err(|e| e.to_string())?;
    file.write_all(contents.as_bytes()).map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().into_owned())
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![export_csv])
        .run(tauri::generate_context!())
        .expect("Macro Atlas could not start");
}
