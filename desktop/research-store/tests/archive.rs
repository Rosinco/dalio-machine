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
    bad["schema_version"] = json!(3);
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

fn business_package(raw: &Value) -> Value {
    let mut package: Value = serde_json::from_str(&fixture()).unwrap();
    let content = raw.to_string();
    package["schema_version"] = json!(2);
    package["business"] = json!({"source_file":"business.json", "sha256":atlas_research_store::digest(content.as_bytes()), "content":content});
    package
}

#[test]
fn business_survives_restart_and_projects_without_loading_history_into_index() {
    let raw: Value =
        serde_json::from_str(include_str!("../../tests/fixtures/business.json")).unwrap();
    let package = business_package(&raw);
    let dir = Temp::new();
    let store = Archive::new(dir.0.clone());
    let legacy = store.import(&fixture()).unwrap();
    assert!(store
        .resource(&legacy.id, "business-index")
        .unwrap()
        .is_null());
    let release = store.import(&package.to_string()).unwrap();
    assert_eq!(release.company_count, 1);
    let expected = format!(
        "macro-atlas-research-v2\n{}\n\n{}",
        package["fundamentals"]["sha256"].as_str().unwrap(),
        package["business"]["sha256"].as_str().unwrap()
    );
    assert_eq!(
        release.id,
        atlas_research_store::digest(expected.as_bytes())
    );
    let reopened = Archive::new(dir.0.clone());
    let index = reopened.resource(&release.id, "business-index").unwrap();
    assert_eq!(index["common_year"], 2020);
    assert!(index["companies"]["102"].get("annual").is_none());
    assert!(index.get("research").is_none());
    assert_eq!(
        reopened.resource(&release.id, "company:102").unwrap()["annual"][0]["values"]["revenues"],
        100
    );
    assert!(reopened.resource(&release.id, "company:../").is_err());
    assert_eq!(
        reopened.resource(&release.id, "business-research").unwrap(),
        raw["research"]
    );
}

#[test]
fn malformed_business_and_v1_business_collisions_are_rejected() {
    let mut raw: Value =
        serde_json::from_str(include_str!("../../tests/fixtures/business.json")).unwrap();
    let mut old = business_package(&raw);
    old["schema_version"] = json!(1);
    assert!(inspect(&old.to_string()).is_err());
    let mut damaged = business_package(&raw);
    damaged["business"]["content"] = json!("{}");
    assert!(inspect(&damaged.to_string()).is_err());
    let first = raw["companies"]["102"]["annual"][0].clone();
    raw["companies"]["102"]["annual"]
        .as_array_mut()
        .unwrap()
        .push(first);
    assert!(inspect(&business_package(&raw).to_string()).is_err());
}

#[test]
fn archived_research_accepts_full_length_prose_with_a_separate_size_limit() {
    let mut raw: Value =
        serde_json::from_str(include_str!("../../tests/fixtures/business.json")).unwrap();
    raw["research"][0]["text"] = json!("Å".repeat(25000));
    assert!(inspect(&business_package(&raw).to_string()).is_ok());
    raw["research"][0]["text"] = json!("x".repeat(1000001));
    assert!(inspect(&business_package(&raw).to_string()).is_err());
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
