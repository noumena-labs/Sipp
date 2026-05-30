mod bindings;
mod cmake;
mod context;
mod ide;
mod link;
mod pipeline;
mod targets;
mod util;

pub(crate) fn run() {
    pipeline::run();
}
