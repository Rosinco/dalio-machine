//! Release check: validate actual exported packages and verify exact resource round trips.
use atlas_research_store::{inspect, Archive};
use serde_json::Value;

fn main() {
    let root = std::env::temp_dir().join(format!("atlas-release-check-{}", std::process::id()));
    let store = Archive::new(root.clone());
    for path in std::env::args().skip(1) {
        let text = std::fs::read_to_string(&path).unwrap();
        let package: Value = serde_json::from_str(&text).unwrap();
        let fund: Value =
            serde_json::from_str(package["fundamentals"]["content"].as_str().unwrap()).unwrap();
        let release = store.import(&text).unwrap();
        assert_eq!(release.id, inspect(&text).unwrap().id);
        assert_eq!(store.read(&release.id).unwrap(), text);
        for (code, country) in fund["countries"].as_object().unwrap() {
            assert_eq!(
                &store
                    .resource(&release.id, &format!("country:{code}"))
                    .unwrap(),
                country
            );
        }
        let expected_liquidity: Value = package["liquidity"]["content"]
            .as_str()
            .map(|s| serde_json::from_str(s).unwrap())
            .unwrap_or(Value::Null);
        assert_eq!(
            store.resource(&release.id, "liquidity").unwrap(),
            expected_liquidity
        );
        if let Some(content) = package["business"]["content"].as_str() {
            let business: Value = serde_json::from_str(content).unwrap();
            for (id, company) in business["companies"].as_object().unwrap() {
                assert_eq!(
                    &store
                        .resource(&release.id, &format!("company:{id}"))
                        .unwrap(),
                    company
                );
            }
            let projection = std::path::Path::new(&path)
                .parent()
                .unwrap()
                .join("business-index.json");
            if projection.exists() {
                let expected: Value =
                    serde_json::from_str(&std::fs::read_to_string(projection).unwrap()).unwrap();
                assert_eq!(
                    store.resource(&release.id, "business-index").unwrap(),
                    expected
                );
            }
            assert_eq!(
                store.resource(&release.id, "business-research").unwrap(),
                business["research"]
            );
        } else {
            assert!(store
                .resource(&release.id, "business-index")
                .unwrap()
                .is_null());
        }
        println!(
            "Verified {}: {} economies, exact saved resources, {}",
            release.as_of, release.country_count, release.id
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}
