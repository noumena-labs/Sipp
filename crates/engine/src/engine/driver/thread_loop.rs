use std::collections::HashMap;
use std::sync::mpsc;
use std::time::Duration;

use futures_channel::{mpsc as futures_mpsc, oneshot};

use cogentlm_core::TokenBatch;

use crate::engine::protocol::{EmbedRequest, EngineEvent, EngineState, EngineStatus, ModelState};
use crate::error::Result;
use crate::runtime::request::GenerateResponse;
use crate::runtime::{InferenceRuntime, RequestStepResult};

use super::events::{build_engine_state_with_status, emit_event, emit_state_event};
use super::request::{start_chat, start_embed, start_query, ChatRequest, QueryRequest};
use super::token_stream::{drain_ring_into_sender, ActiveTokenStream};
use super::{runtime_command, EngineEventSubscribers};

mod completion;

const RUNTIME_CLOSED: &str = "runtime is closed";
const ENGINE_INVALID_DURING_EXECUTION: &str = "Engine became invalid during execution.";
const ENGINE_NO_PROGRESS: &str = "Engine execution failed with no progress.";

pub(super) enum EngineThreadCommand {
    Generate(
        QueryRequest,
        oneshot::Sender<Result<GenerateResponse>>,
        Option<futures_mpsc::Sender<TokenBatch>>,
    ),
    GenerateChat(
        ChatRequest,
        oneshot::Sender<Result<GenerateResponse>>,
        Option<futures_mpsc::Sender<TokenBatch>>,
    ),
    Embed(EmbedRequest, oneshot::Sender<Result<GenerateResponse>>),
    GetState(oneshot::Sender<Result<EngineState>>),
    Close(Option<oneshot::Sender<Result<()>>>),
}

pub(super) fn run_engine_thread(
    runtime: InferenceRuntime,
    command_rx: mpsc::Receiver<EngineThreadCommand>,
    model_state: ModelState,
    event_subscribers: EngineEventSubscribers,
) {
    let mut state = EngineThreadState {
        runtime: Some(runtime),
        active_requests: HashMap::new(),
        model_state,
        event_subscribers,
    };

    loop {
        if state.active_requests.is_empty() {
            let Ok(command) = command_rx.recv() else {
                break;
            };
            if !state.process_command(command) {
                break;
            }
            continue;
        }

        let mut stop = false;
        while let Ok(command) = command_rx.try_recv() {
            if !state.process_command(command) {
                stop = true;
                break;
            }
        }
        if stop {
            break;
        }
        state.step_active_requests();
    }
}

pub(super) struct EngineThreadState {
    runtime: Option<InferenceRuntime>,
    active_requests: HashMap<u32, ActiveRequest>,
    model_state: ModelState,
    event_subscribers: EngineEventSubscribers,
}

pub(super) struct ActiveRequest {
    pub output: ActiveRequestOutput,
    pub response_tx: oneshot::Sender<Result<GenerateResponse>>,
    pub token: Option<ActiveTokenStream>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ActiveRequestOutput {
    Text,
    Embedding,
}

impl EngineThreadState {
    fn process_command(&mut self, command: EngineThreadCommand) -> bool {
        match command {
            EngineThreadCommand::Generate(request, response_tx, token_tx) => {
                self.start_request(
                    response_tx,
                    ActiveRequestOutput::Text,
                    |runtime, subscribers| start_query(runtime, request, token_tx, subscribers),
                );
            }
            EngineThreadCommand::GenerateChat(request, response_tx, token_tx) => {
                self.start_request(
                    response_tx,
                    ActiveRequestOutput::Text,
                    |runtime, subscribers| start_chat(runtime, request, token_tx, subscribers),
                );
            }
            EngineThreadCommand::Embed(request, response_tx) => {
                self.start_request(
                    response_tx,
                    ActiveRequestOutput::Embedding,
                    |runtime, subscribers| start_embed(runtime, request, subscribers),
                );
            }
            EngineThreadCommand::GetState(response_tx) => {
                let _ = response_tx.send(self.current_state());
            }
            EngineThreadCommand::Close(ack_tx) => {
                self.close_active_requests();
                drop(self.runtime.take());
                emit_event(&self.event_subscribers, EngineEvent::Closed);
                if let Some(ack_tx) = ack_tx {
                    let _ = ack_tx.send(Ok(()));
                }
                return false;
            }
        }
        true
    }

    fn start_request(
        &mut self,
        response_tx: oneshot::Sender<Result<GenerateResponse>>,
        output: ActiveRequestOutput,
        start: impl FnOnce(
            &mut InferenceRuntime,
            &EngineEventSubscribers,
        ) -> Result<(u32, Option<ActiveTokenStream>)>,
    ) {
        let Some(runtime) = self.runtime.as_mut() else {
            let _ = response_tx.send(Err(runtime_command(RUNTIME_CLOSED)));
            return;
        };

        match start(runtime, &self.event_subscribers) {
            Ok((request_id, token_stream)) => {
                self.active_requests.insert(
                    request_id,
                    ActiveRequest {
                        output,
                        response_tx,
                        token: token_stream,
                    },
                );
                emit_state_event(
                    runtime,
                    &self.model_state,
                    &self.event_subscribers,
                    EngineStatus::Running,
                );
            }
            Err(error) => {
                let _ = response_tx.send(Err(error));
            }
        }
    }

    fn current_state(&self) -> Result<EngineState> {
        let Some(runtime) = self.runtime.as_ref() else {
            return Err(runtime_command(RUNTIME_CLOSED));
        };
        Ok(build_engine_state_with_status(
            runtime,
            &self.model_state,
            Some(active_request_status(!self.active_requests.is_empty())),
        ))
    }

    fn step_active_requests(&mut self) {
        if self.active_requests.is_empty() {
            return;
        }

        let dropped: Vec<_> = self
            .active_requests
            .iter()
            .filter(|(_, request)| request.response_tx.is_canceled())
            .map(|(&request_id, _)| request_id)
            .collect();
        for request_id in dropped {
            self.cancel_and_cleanup_request(request_id);
            self.active_requests.remove(&request_id);
            self.emit_request_failed(request_id, "request cancelled".to_string());
        }
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        if self.active_requests.is_empty() {
            return;
        }

        let burst = runtime.run_scheduler_loop(1, 0, 0, Duration::ZERO);
        for request in self.active_requests.values_mut() {
            if let Some(token) = request.token.as_mut() {
                drain_ring_into_sender(token);
            }
        }
        self.complete_finished_requests();

        if matches!(
            burst.status,
            RequestStepResult::Invalid | RequestStepResult::FatalNoProgress
        ) {
            let error_msg = if burst.status == RequestStepResult::Invalid {
                ENGINE_INVALID_DURING_EXECUTION.to_string()
            } else {
                ENGINE_NO_PROGRESS.to_string()
            };
            self.fail_all_active_requests(error_msg);
        }

        if self.active_requests.is_empty() {
            if let Some(runtime) = self.runtime.as_mut() {
                emit_state_event(
                    runtime,
                    &self.model_state,
                    &self.event_subscribers,
                    EngineStatus::Ready,
                );
            }
        }
    }
}

fn active_request_status(has_active_requests: bool) -> EngineStatus {
    if has_active_requests {
        EngineStatus::Running
    } else {
        EngineStatus::Ready
    }
}
