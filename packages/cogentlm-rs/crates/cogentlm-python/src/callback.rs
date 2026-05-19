//! Python `on_tokens` callback bridge and core/model error → PyErr mapping.
//!
//! When a Python callback raises, we stash the `PyErr` in a shared slot, drive
//! the core to fail the request with a sentinel message, then translate that
//! sentinel back into the original `PyErr` at the outer boundary.

use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use cogentlm_core::{RequestResult, TokenBatch};

use super::dicts::token_batch_to_dict;
use super::PY_CALLBACK_FAILED_MESSAGE;

pub(super) fn require_callable(py: Python<'_>, callback: &PyObject) -> PyResult<()> {
    if callback.bind(py).is_callable() {
        Ok(())
    } else {
        Err(PyTypeError::new_err("on_tokens must be callable"))
    }
}

pub(super) fn make_python_tokens_callback(
    callback: PyObject,
    callback_error: Arc<Mutex<Option<PyErr>>>,
) -> impl FnMut(&TokenBatch) -> cogentlm_core::Result<()> + Send + 'static {
    move |batch| {
        if has_callback_error(&callback_error) {
            return Err(cogentlm_core::Error::RuntimeCommand(
                PY_CALLBACK_FAILED_MESSAGE.to_string(),
            ));
        }

        Python::with_gil(|py| {
            let batch = token_batch_to_dict(py, batch.clone()).map_err(|error| {
                store_callback_error(&callback_error, error);
                cogentlm_core::Error::RuntimeCommand(PY_CALLBACK_FAILED_MESSAGE.to_string())
            })?;
            callback.call1(py, (batch,)).map(|_| ()).map_err(|error| {
                store_callback_error(&callback_error, error);
                cogentlm_core::Error::RuntimeCommand(PY_CALLBACK_FAILED_MESSAGE.to_string())
            })
        })
    }
}

pub(super) fn py_token_result_to_request_result(
    result: cogentlm_core::Result<RequestResult>,
    callback_error: Arc<Mutex<Option<PyErr>>>,
) -> PyResult<RequestResult> {
    result.map_err(|error| callback_error_or_core_error(error, callback_error))
}

pub(super) fn py_model_token_result_to_request_result(
    result: Result<RequestResult, cogentlm_core::ModelError>,
    callback_error: Arc<Mutex<Option<PyErr>>>,
) -> PyResult<RequestResult> {
    result.map_err(|error| callback_error_or_model_error(error, callback_error))
}

fn callback_error_or_model_error(
    error: cogentlm_core::ModelError,
    callback_error: Arc<Mutex<Option<PyErr>>>,
) -> PyErr {
    if is_python_model_callback_error(&error) {
        if let Some(stored) = take_callback_error(&callback_error) {
            return stored;
        }
    }
    to_py_model_error(error)
}

fn is_python_model_callback_error(error: &cogentlm_core::ModelError) -> bool {
    matches!(
        error,
        cogentlm_core::ModelError::Runtime(message) if message.contains(PY_CALLBACK_FAILED_MESSAGE)
    )
}

fn callback_error_or_core_error(
    error: cogentlm_core::Error,
    callback_error: Arc<Mutex<Option<PyErr>>>,
) -> PyErr {
    if is_python_callback_error(&error) {
        if let Some(stored) = take_callback_error(&callback_error) {
            return stored;
        }
    }
    to_py_error(error)
}

fn is_python_callback_error(error: &cogentlm_core::Error) -> bool {
    matches!(
        error,
        cogentlm_core::Error::RuntimeCommand(message) if message == PY_CALLBACK_FAILED_MESSAGE
    )
}

fn has_callback_error(callback_error: &Arc<Mutex<Option<PyErr>>>) -> bool {
    match callback_error.lock() {
        Ok(error) => error.is_some(),
        Err(poisoned) => poisoned.into_inner().is_some(),
    }
}

fn store_callback_error(callback_error: &Arc<Mutex<Option<PyErr>>>, error: PyErr) {
    let mut stored = match callback_error.lock() {
        Ok(stored) => stored,
        Err(poisoned) => poisoned.into_inner(),
    };
    if stored.is_none() {
        *stored = Some(error);
    }
}

fn take_callback_error(callback_error: &Arc<Mutex<Option<PyErr>>>) -> Option<PyErr> {
    match callback_error.lock() {
        Ok(mut error) => error.take(),
        Err(poisoned) => poisoned.into_inner().take(),
    }
}

pub(super) fn to_py_error(error: cogentlm_core::Error) -> PyErr {
    match error {
        cogentlm_core::Error::InvalidRequest(message)
        | cogentlm_core::Error::InvalidConfig(message) => PyValueError::new_err(message),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

pub(super) fn to_py_model_error(error: cogentlm_core::ModelError) -> PyErr {
    match error {
        cogentlm_core::ModelError::InvalidModelSource(message)
        | cogentlm_core::ModelError::InvalidModelPairing(message) => PyValueError::new_err(message),
        cogentlm_core::ModelError::UnsupportedGgufVersion(version) => {
            PyValueError::new_err(format!("unsupported GGUF version {version}"))
        }
        other => PyRuntimeError::new_err(other.to_string()),
    }
}
