use atlas_research_store::{inspect, Archive};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

fn fixture() -> String {
    let categories = [
        "real_stuff",
        "production",
        "exchange",
        "promises",
        "enforcer",
    ];
    let mut country = json!({"name":"Test Sweden", "iso3":"SWE", "on_map":true,
        "currency":"SEK", "data_quality":{"flag":"test","note":null},
        "categories":{}, "indicators":{}, "history":{}, "pressures":[]});
    let mut indicators = vec![];
    for key in categories {
        indicators.push(json!({"name":key,"category":key,"label":key,"unit":"%",
            "scored":true,"higher_is_better":false,"uncertainty":"A",
            "description":"Synthetic archive test", "cadence":"A","sources":["TEST"],"forward":false}));
        country["categories"][key] = json!({"score":50.0,"n_available":1,"n_total":1});
        country["indicators"][key] = json!({"value":0.0,"pct":50.0,"date":"2020-12-31",
            "source":"TEST","is_forecast":false,"uncertainty":"A","trend":null});
        country["history"][key] = json!([{"year":2020,"value":null,"is_forecast":false}]);
    }
    let content =
        json!({"version":1,"as_of":"2021-01-01","generated_at":"2021-01-01T12:00:00+00:00",
        "ranking_population":["SE"],"categories":categories,"indicators":indicators,
        "countries":{"SE":country},"trade":[]})
        .to_string();
    json!({"format":"macro-atlas-research","schema_version":1,
        "fundamentals":{"source_file":"synthetic.json","sha256":format!("{:x}",Sha256::digest(content.as_bytes())),"content":content},
        "liquidity":null}).to_string()
}

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "atlas-store-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn import_survives_reopen_and_duplicate_is_idempotent() {
    let dir = Temp::new();
    let store = Archive::new(dir.0.clone());
    let first = store.import(&fixture()).unwrap();
    assert_eq!(store.import(&fixture()).unwrap().id, first.id);
    let reopened = Archive::new(dir.0.clone());
    assert_eq!(reopened.list().unwrap().releases.len(), 1);
    let country = reopened.resource(&first.id, "country:SE").unwrap();
    assert_eq!(
        country["indicators"]["promises"]["value"].as_f64(),
        Some(0.0)
    );
    assert!(country["history"]["promises"][0]["value"].is_null());
    assert!(
        reopened.resource(&first.id, "index").unwrap()["countries"]["SE"]
            .get("history")
            .is_none()
    );
}

#[test]
fn tampered_payload_cannot_change_the_archive() {
    let dir = Temp::new();
    let store = Archive::new(dir.0.clone());
    store.import(&fixture()).unwrap();
    let mut bad: Value = serde_json::from_str(&fixture()).unwrap();
    bad["fundamentals"]["content"] = json!("{}");
    assert!(store.import(&bad.to_string()).is_err());
    assert_eq!(store.list().unwrap().releases.len(), 1);
}

#[test]
fn schema_validation_rejects_unsupported_and_invalid_scored_data() {
    let mut bad: Value = serde_json::from_str(&fixture()).unwrap();
    bad["schema_version"] = json!(2);
    assert!(inspect(&bad.to_string()).is_err());
    bad["schema_version"] = json!(1);
    let mut raw: Value =
        serde_json::from_str(bad["fundamentals"]["content"].as_str().unwrap()).unwrap();
    raw["countries"]["SE"]["categories"]["promises"]["score"] = json!(150);
    let content = raw.to_string();
    bad["fundamentals"]["sha256"] = json!(format!("{:x}", Sha256::digest(content.as_bytes())));
    bad["fundamentals"]["content"] = json!(content);
    assert!(inspect(&bad.to_string()).is_err());
}

#[test]
fn ids_and_resources_cannot_read_outside_the_archive() {
    let dir = Temp::new();
    let store = Archive::new(dir.0.clone());
    let release = store.import(&fixture()).unwrap();
    assert!(store.resource("../private", "index").is_err());
    assert!(store.resource(&release.id, "../../private").is_err());
    assert!(store.resource(&release.id, "country:../").is_err());
}

#[test]
fn damaged_saved_release_is_reported_and_other_releases_remain_usable() {
    let dir = Temp::new();
    let store = Archive::new(dir.0.clone());
    let release = store.import(&fixture()).unwrap();
    fs::write(
        dir.0.join(format!("{}.atlas.json", "a".repeat(64))),
        "broken",
    )
    .unwrap();
    let list = store.list().unwrap();
    assert_eq!(list.releases.len(), 1);
    assert_eq!(list.unreadable, 1);
    assert_eq!(list.releases[0].id, release.id);
}
