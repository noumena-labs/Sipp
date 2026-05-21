use std::time::Duration;

use crate::engine::protocol::EngineEvent;
use crate::error::Error;
use crate::runtime::request::GenerateResponseStatus;
use crate::runtime::RequestStepResult;

use super::super::events::emit_event;
use super::super::stats::request_result_from_response;
use super::EngineThreadState;

fn cancel_and_consume_request(runtime: &mut crate::runtime::InferenceRuntime, request_id: u32) {
    let _ = runtime.cancel_request(request_id);

    loop {
        if runtime.try_peek_completed_response(request_id).is_some() {
            runtime.consume_completed_response(request_id);
            return;
        }
        if !runtime.has_request(request_id) {
            return;
        }

        let burst = runtime.run_scheduler_loop(256, 1, 0, Duration::ZERO);

        if matches!(
            burst.status,
            RequestStepResult::Invalid | RequestStepResult::FatalNoProgress
        ) {
            return;
        }
        if burst.status == RequestStepResult::Waiting
            && runtime.try_peek_completed_response(request_id).is_none()
        {
            return;
        }
    }
}

trait EmptyStringFallback {
    fn if_empty(self, fallback: &'static str) -> String;
}

impl EmptyStringFallback for String {
    fn if_empty(self, fallback: &'static str) -> String {
        if self.is_empty() {
            fallback.to_string()
        } else {
            self
        }
    }
}

impl EngineThreadState {
    pub(super) fn fail_requests_with_sink_errors(&mut self) {
        let errored_ids: Vec<_> = self
            .token_sinks
            .iter_mut()
            .filter_map(|(&request_id, sink)| {
                sink.try_recv_error().map(|error| {
                    let message = error.to_string();
                    (request_id, error, message)
                })
            })
            .collect();

        for (request_id, error, error_msg) in errored_ids {
            self.cancel_and_cleanup_request(request_id);
            if let Some(response_tx) = self.active_requests.remove(&request_id) {
                let _ = response_tx.send(Err(error));
            }
            emit_event(
                &self.event_subscribers,
                EngineEvent::RequestFailed {
                    request_id: request_id.to_string(),
                    error: error_msg,
                },
            );
        }
    }

    pub(super) fn complete_finished_requests(&mut self) {
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        let completed: Vec<_> = self
            .active_requests
            .keys()
            .copied()
            .filter_map(|request_id| {
                runtime
                    .take_completed_response(request_id)
                    .map(|response| (request_id, response))
            })
            .collect();

        for (request_id, response) in completed {
            self.close_token_sink(request_id);
            if let Some(runtime) = self.runtime.as_mut() {
                runtime.remove_token_ring_producer(request_id);
            }

            let response_tx = self.active_requests.remove(&request_id);
            let result = match response.status {
                GenerateResponseStatus::Completed => {
                    let result = request_result_from_response(&response);
                    emit_event(
                        &self.event_subscribers,
                        EngineEvent::RequestCompleted {
                            result: result.clone(),
                        },
                    );
                    Ok(response)
                }
                GenerateResponseStatus::Cancelled => {
                    let message = response.error_message.if_empty("request cancelled");
                    self.emit_request_failed(request_id, message.clone());
                    Err(Error::RuntimeCommand(message))
                }
                GenerateResponseStatus::Failed => {
                    let message = response.error_message.if_empty("request failed");
                    self.emit_request_failed(request_id, message.clone());
                    Err(Error::RuntimeCommand(message))
                }
                GenerateResponseStatus::Pending => Err(Error::RuntimeCommand(
                    "request returned pending response".to_string(),
                )),
            };

            if let Some(tx) = response_tx {
                let _ = tx.send(result);
            }
        }
    }

    pub(super) fn fail_all_active_requests(&mut self, error_msg: String) {
        let remaining_ids: Vec<u32> = self.active_requests.keys().copied().collect();
        for request_id in remaining_ids {
            self.cancel_and_cleanup_request(request_id);
            if let Some(response_tx) = self.active_requests.remove(&request_id) {
                let _ = response_tx.send(Err(Error::RuntimeCommand(error_msg.clone())));
            }
            self.emit_request_failed(request_id, error_msg.clone());
        }
    }

    pub(super) fn close_active_requests(&mut self) {
        let remaining_ids: Vec<u32> = self.active_requests.keys().copied().collect();
        for request_id in remaining_ids {
            self.cancel_and_cleanup_request(request_id);
            if let Some(response_tx) = self.active_requests.remove(&request_id) {
                let _ = response_tx.send(Err(Error::RuntimeCommand("engine closed".to_string())));
            }
        }
    }

    fn cancel_and_cleanup_request(&mut self, request_id: u32) {
        if let Some(runtime) = self.runtime.as_mut() {
            cancel_and_consume_request(runtime, request_id);
            runtime.remove_token_ring_producer(request_id);
        }
        self.close_token_sink(request_id);
    }

    fn close_token_sink(&mut self, request_id: u32) {
        if let Some(mut sink) = self.token_sinks.remove(&request_id) {
            sink.close();
            let _ = sink.join();
        }
    }

    fn emit_request_failed(&self, request_id: u32, error: String) {
        emit_event(
            &self.event_subscribers,
            EngineEvent::RequestFailed {
                request_id: request_id.to_string(),
                error,
            },
        );
    }
}
