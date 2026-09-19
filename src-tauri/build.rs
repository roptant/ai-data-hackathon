fn main() {
    // Declaring the app commands makes each one a permission that must be
    // granted per window in `capabilities/`; nothing is allowed implicitly.
    tauri_build::try_build(tauri_build::Attributes::new().app_manifest(
        tauri_build::AppManifest::new().commands(&[
            "get_status",
            "get_settings",
            "update_settings",
            "capability_matrix",
            "recording_action",
            "do_not_contribute",
            "get_result",
            "copy_result",
            "dismiss_result",
            "list_models",
            "install_model",
            "cancel_model_download",
            "import_model",
            "install_custom_model",
            "choose_model_file",
            "get_disclosure",
            "grant_consent",
            "set_contribution_paused",
            "withdraw_consent",
            "delete_training_data",
            "contribution_history",
            "list_pairings",
            "approve_pairing",
            "deny_pairing",
            "list_clients",
            "revoke_client",
            "retry_shortcuts",
        ]),
    ))
    .expect("failed to run the Tauri build script");
}
