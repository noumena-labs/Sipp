use std::time::Duration;

use crate::engine::protocol::EngineEvent;
use crate::error::Result;
use crate::runtime::request::{GenerateResponse, GenerateResponseStatus};
use crate::runtime::RequestStepResult;

use super::super::events::emit_event;
use super::super::runtime_command;
use super::super::stats::generation_result_from_response;
use super::EngineThreadState;

const REQUEST_CANCELLED_FALLBACK: &str = "request cancelled";
const REQUEST_FAILED_FALLBACK: &str = "request failed";
const REQUEST_PENDING_RESPONSE: &str = "request returned pending response";
const ENGINE_CLOSED: &str = "engine closed";

fn cancel_and_consume_request(runtime: &mut crate::runtime::InferenceRuntime, request_id: u32) {
    let _ = runtime.cancel_request(request_id);

    loop {
        if runtime
            .request_queue
            .completed_responses
            .contains_key(&request_id)
        {
            let _ = runtime.take_completed_response(request_id);
            return;
        }
        if !runtime.request_queue.requests.contains_key(&request_id) {
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
            && !runtime
                .request_queue
                .completed_responses
                .contains_key(&request_id)
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
            self.send_active_response(request_id, Err(error));
            self.emit_request_failed(request_id, error_msg);
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
                runtime
                    .request_queue
                    .token_ring_producers
                    .remove(&request_id);
            }

            let result = match response.status {
                GenerateResponseStatus::Completed => {
                    match generation_result_from_response(&response) {
                        Ok(result) => {
                            emit_event(
                                &self.event_subscribers,
                                EngineEvent::RequestCompleted {
                                    result: Box::new(result.clone()),
                                },
                            );
                            Ok(response)
                        }
                        Err(error) => {
                            self.emit_request_failed(request_id, error.to_string());
                            Err(error)
                        }
                    }
                }
                GenerateResponseStatus::Cancelled => self.failed_completion_result(
                    request_id,
                    response.error_message,
                    REQUEST_CANCELLED_FALLBACK,
                ),
                GenerateResponseStatus::Failed => self.failed_completion_result(
                    request_id,
                    response.error_message,
                    REQUEST_FAILED_FALLBACK,
                ),
                GenerateResponseStatus::Pending => Err(runtime_command(REQUEST_PENDING_RESPONSE)),
            };

            self.send_active_response(request_id, result);
        }
    }

    pub(super) fn fail_all_active_requests(&mut self, error_msg: String) {
        for request_id in self.active_request_ids() {
            self.cancel_and_cleanup_request(request_id);
            self.send_active_response(request_id, Err(runtime_command(error_msg.clone())));
            self.emit_request_failed(request_id, error_msg.clone());
        }
    }

    pub(super) fn close_active_requests(&mut self) {
        for request_id in self.active_request_ids() {
            self.cancel_and_cleanup_request(request_id);
            self.send_active_response(request_id, Err(runtime_command(ENGINE_CLOSED)));
        }
    }

    fn active_request_ids(&self) -> Vec<u32> {
        self.active_requests.keys().copied().collect()
    }

    fn cancel_and_cleanup_request(&mut self, request_id: u32) {
        if let Some(runtime) = self.runtime.as_mut() {
            cancel_and_consume_request(runtime, request_id);
            runtime
                .request_queue
                .token_ring_producers
                .remove(&request_id);
        }
        self.close_token_sink(request_id);
    }

    fn close_token_sink(&mut self, request_id: u32) {
        if let Some(mut sink) = self.token_sinks.remove(&request_id) {
            sink.producer.close();
            let _ = sink.join();
        }
    }

    fn send_active_response(&mut self, request_id: u32, response: Result<GenerateResponse>) {
        if let Some(response_tx) = self.active_requests.remove(&request_id) {
            let _ = response_tx.send(response);
        }
    }

    fn failed_completion_result(
        &self,
        request_id: u32,
        error_message: String,
        fallback: &'static str,
    ) -> Result<GenerateResponse> {
        let message = error_message.if_empty(fallback);
        self.emit_request_failed(request_id, message.clone());
        Err(runtime_command(message))
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
