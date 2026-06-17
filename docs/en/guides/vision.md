# Vision

Vision workflows send image data alongside chat messages. They are local-only
unless an application gateway deliberately implements an equivalent media
profile.

## Local Vision

Local vision examples typically require:

- A compatible vision-capable GGUF model.
- A projector GGUF when required by the model family.
- An image path, browser canvas export, or image byte payload.

The Rust, Node.js, and Python example directories include `vision_chat`
examples. Browser demos show canvas and file-oriented workflows.

## Browser Vision

Browser applications keep captured image payloads small and grounded in the
task. The proactive drawing demo uses cropped JPEG captures so the model sees
the relevant ink instead of a full page screenshot.

## Related Docs

- [Local Inference](local-inference.md)
- [Models And Backends](../getting-started/models-backends.md)
- [Examples And Demos](../examples-demos.md)
