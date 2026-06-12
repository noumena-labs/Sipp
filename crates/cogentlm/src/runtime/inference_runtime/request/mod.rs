mod api;
mod lifecycle;

#[cfg(test)]
#[path = "../../../tests/runtime/inference_runtime/request/api_tests.rs"]
mod api_tests;

#[cfg(test)]
#[path = "../../../tests/runtime/inference_runtime/request/lifecycle_tests.rs"]
mod lifecycle_tests;
