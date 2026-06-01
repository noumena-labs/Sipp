use super::*;

fn classified_json() -> &'static str {
    r#"[
            {
                "assetId":"asset-model",
                "name":"base.gguf",
                "inspection":{
                    "version":1,
                    "role":"model",
                    "architecture":"lfm2",
                    "visionCapable":true,
                    "compatibleVisionProjectorTypes":["lfm2"],
                    "providedVisionProjectorType":null
                }
            },
            {
                "assetId":"asset-projector",
                "name":"mmproj.gguf",
                "inspection":{
                    "version":1,
                    "role":"projector",
                    "architecture":"clip",
                    "visionCapable":false,
                    "compatibleVisionProjectorTypes":[],
                    "providedVisionProjectorType":"lfm2"
                }
            }
        ]"#
}

#[test]
fn validates_explicit_projector_pairing() {
    let classified = classified_json();

    let response = read_owned(pairing_validate_json(classified, "asset-projector"));

    assert_eq!(response["ok"], true);
    assert_eq!(response["plan"]["projectorAssetId"], "asset-projector");
    assert_eq!(response["plan"]["status"], "ready");
}

#[test]
fn returns_typed_error_for_invalid_json() {
    let response = read_owned(pairing_validate_json("{", ""));

    assert_eq!(response["ok"], false);
    assert_eq!(response["error"]["code"], CODE_INVALID_MODEL_SOURCE);
}

fn read_owned(text: String) -> serde_json::Value {
    serde_json::from_str(&text).expect("response json")
}
