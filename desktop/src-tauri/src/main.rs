#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use atlas_research_store::{Archive, Library, Release};
use tauri::Manager;

fn archive(app: &tauri::AppHandle) -> Result<Archive, String> {
    // Native tests set this to their temporary directory, isolating the live library.
    // IPC callers cannot select paths; the normal app always uses its stable data directory.
    let root = match std::env::var_os("ATLAS_RESEARCH_DIR") {
        Some(path) => std::path::PathBuf::from(path),
        None => app
            .path()
            .app_local_data_dir()
            .map_err(|e| e.to_string())?
            .join("research-v1"),
    };
    Ok(Archive::new(root))
}
#[tauri::command]
fn research_list(app: tauri::AppHandle) -> Result<Library, String> {
    archive(&app)?.list()
}
#[tauri::command]
fn research_inspect(contents: String) -> Result<Release, String> {
    atlas_research_store::inspect(&contents)
}
#[tauri::command]
fn research_import(app: tauri::AppHandle, contents: String) -> Result<Release, String> {
    archive(&app)?.import(&contents)
}
#[tauri::command]
fn research_resource(
    app: tauri::AppHandle,
    id: String,
    resource: String,
) -> Result<serde_json::Value, String> {
    archive(&app)?.resource(&id, &resource)
}
#[tauri::command]
fn research_package(app: tauri::AppHandle, id: String) -> Result<String, String> {
    archive(&app)?.read(&id)
}
#[tauri::command]
fn research_export(app: tauri::AppHandle, contents: String) -> Result<String, String> {
    let release = atlas_research_store::inspect(&contents)?;
    let folder = app.path().download_dir().map_err(|e| e.to_string())?;
    let path = folder.join(format!(
        "Macro-Atlas-Research-{}-{}.atlas.json",
        release.as_of,
        &release.id[..12]
    ));
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
    {
        Ok(mut file) => {
            use std::io::Write;
            file.write_all(contents.as_bytes())
                .map_err(|e| e.to_string())?;
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
            if atlas_research_store::inspect(&existing)?.id != release.id {
                return Err("A different file already has this name.".into());
            }
        }
        Err(e) => return Err(e.to_string()),
    }
    Ok(path.to_string_lossy().into_owned())
}

#[tauri::command]
fn export_csv(app: tauri::AppHandle, filename: String, contents: String) -> Result<String, String> {
    if !filename.starts_with("Macro-Atlas-")
        || !filename.ends_with(".csv")
        || filename.len() > 160
        || filename
            .chars()
            .any(|c| !(c.is_ascii_alphanumeric() || "-_.".contains(c)))
        || contents.len() > 8_000_000
    {
        return Err("Invalid export".into());
    }
    let folder = app.path().download_dir().map_err(|e| e.to_string())?;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_millis();
    let path = folder.join(format!(
        "{}-{}.csv",
        filename.trim_end_matches(".csv"),
        timestamp
    ));
    // Exports always create a new file. Existing user files cannot be overwritten.
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| e.to_string())?;
    file.write_all(contents.as_bytes())
        .map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().into_owned())
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            export_csv,
            research_list,
            research_inspect,
            research_import,
            research_resource,
            research_package,
            research_export
        ])
        .run(tauri::generate_context!())
        .expect("Macro Atlas could not start");
}
