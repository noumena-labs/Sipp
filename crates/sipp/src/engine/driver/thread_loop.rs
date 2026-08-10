use std::collections::{HashMap, VecDeque};
use std::sync::mpsc;
use std::time::Duration;

use futures_channel::{mpsc as futures_mpsc, oneshot};

use crate::core::TokenBatch;

use crate::engine::protocol::{EmbedRequest, EngineEvent, EngineState, EngineStatus, ModelState};
use crate::error::Result;
use crate::runtime::request::GenerateResponse;
use crate::runtime::{InferenceRuntime, RequestStepResult};

use super::events::{build_engine_state_with_status, emit_event, emit_state_event};
use super::request::{
    start_chat, start_embed, start_listen, start_query, start_speak, ChatRequest, ListenRequest,
    QueryRequest, SpeakRequest,
};
use super::token_emission::{drain_ring_into_sender, ActiveTokenEmission};
use super::{runtime_command, EngineCancellation, EngineEventSubscribers};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../tests/engine/driver/thread_loop_tests.rs"]
mod thread_loop_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////
mod completion;

const RUNTIME_CLOSED: &str = "runtime is closed";
const ENGINE_INVALID_DURING_EXECUTION: &str = "Engine became invalid during execution.";
const ENGINE_NO_PROGRESS: &str = "Engine execution failed with no progress.";

pub(super) enum EngineThreadCommand {
    Generate(
        QueryRequest,
        oneshot::Sender<Result<GenerateResponse>>,
        Option<futures_mpsc::UnboundedSender<TokenBatch>>,
    ),
    GenerateChat(
        ChatRequest,
        oneshot::Sender<Result<GenerateResponse>>,
        Option<futures_mpsc::UnboundedSender<TokenBatch>>,
    ),
    Embed(EmbedRequest, oneshot::Sender<Result<GenerateResponse>>),
    Listen(ListenRequest, oneshot::Sender<Result<GenerateResponse>>),
    Speak(
        SpeakRequest,
        EngineCancellation,
        oneshot::Sender<Result<GenerateResponse>>,
    ),
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

    let mut pending_commands = VecDeque::new();
    loop {
        if state.has_active_speech() {
            if !drain_commands(&mut state, &command_rx, &mut pending_commands) {
                break;
            }
            state.step_active_requests();
            continue;
        }

        let can_process_pending = pending_commands.front().is_some_and(|command| {
            state.active_requests.is_empty() || !matches!(command, EngineThreadCommand::Speak(..))
        });
        if can_process_pending {
            if let Some(command) = pending_commands.pop_front() {
                if !state.process_command(command) {
                    break;
                }
            }
            continue;
        }

        if state.active_requests.is_empty() {
            let Ok(command) = command_rx.recv() else {
                break;
            };
            if !state.process_command(command) {
                break;
            }
            continue;
        }

        if !drain_commands(&mut state, &command_rx, &mut pending_commands) {
            break;
        }
        state.step_active_requests();
    }
}

fn drain_commands(
    state: &mut EngineThreadState,
    command_rx: &mpsc::Receiver<EngineThreadCommand>,
    pending_commands: &mut VecDeque<EngineThreadCommand>,
) -> bool {
    while let Ok(command) = command_rx.try_recv() {
        if is_control_command(&command) {
            if !state.process_command(command) {
                return false;
            }
        } else {
            pending_commands.push_back(command);
        }
    }
    true
}

fn is_control_command(command: &EngineThreadCommand) -> bool {
    matches!(
        command,
        EngineThreadCommand::GetState(_) | EngineThreadCommand::Close(_)
    )
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
    pub token: Option<ActiveTokenEmission>,
    pub cancellation: Option<EngineCancellation>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum ActiveRequestOutput {
    Text,
    Transcription { language: Option<String> },
    Embedding,
    Speech,
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
            EngineThreadCommand::Listen(request, response_tx) => {
                let output = ActiveRequestOutput::Transcription {
                    language: request.language.clone(),
                };
                self.start_request(response_tx, output, |runtime, subscribers| {
                    start_listen(runtime, request, subscribers)
                });
            }
            EngineThreadCommand::Speak(request, cancellation, response_tx) => {
                if cancellation.is_cancelled() || response_tx.is_canceled() {
                    return true;
                }
                self.start_request_with_cancellation(
                    response_tx,
                    ActiveRequestOutput::Speech,
                    Some(cancellation),
                    |runtime, subscribers| start_speak(runtime, request, subscribers),
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
        ) -> Result<(u32, Option<ActiveTokenEmission>)>,
    ) {
        self.start_request_with_cancellation(response_tx, output, None, start);
    }

    fn start_request_with_cancellation(
        &mut self,
        response_tx: oneshot::Sender<Result<GenerateResponse>>,
        output: ActiveRequestOutput,
        cancellation: Option<EngineCancellation>,
        start: impl FnOnce(
            &mut InferenceRuntime,
            &EngineEventSubscribers,
        ) -> Result<(u32, Option<ActiveTokenEmission>)>,
    ) {
        let Some(runtime) = self.runtime.as_mut() else {
            let _ = response_tx.send(Err(runtime_command(RUNTIME_CLOSED)));
            return;
        };

        match start(runtime, &self.event_subscribers) {
            Ok((request_id, token_emission)) => {
                self.active_requests.insert(
                    request_id,
                    ActiveRequest {
                        output,
                        response_tx,
                        token: token_emission,
                        cancellation,
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

    fn has_active_speech(&self) -> bool {
        self.active_requests
            .values()
            .any(|request| request.output == ActiveRequestOutput::Speech)
    }

    fn step_active_requests(&mut self) {
        if self.active_requests.is_empty() {
            return;
        }

        let dropped: Vec<_> = self
            .active_requests
            .iter()
            .filter(|(_, request)| {
                request.response_tx.is_canceled()
                    || request
                        .cancellation
                        .as_ref()
                        .is_some_and(EngineCancellation::is_cancelled)
            })
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
